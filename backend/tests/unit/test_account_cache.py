"""AccountCache serialization and AccountService cache-aside (fakes, no server)."""

import io
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from app.core.cache import NullCache
from app.core.exceptions import NotFoundError
from app.core.logging import configure_logging
from app.models import Account
from app.repositories.account import AccountRepository
from app.repositories.user import UserRepository
from app.schemas.account import AccountRead
from app.services.account_cache import KEY_PREFIX, AccountCache
from app.services.account_service import AccountService

TTL = 300
ACCOUNT_ID = uuid.uuid4()
NOW = datetime(2026, 10, 23, 12, 0, 0, 123456, tzinfo=UTC)


class FakeCache:
    """In-memory Cache that can simulate an unavailable backend."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.ttls: dict[str, int] = {}
        self.available = True

    def get(self, key: str) -> bytes | None:
        return self.store.get(key) if self.available else None

    def set(self, key: str, value: bytes, ttl_seconds: int) -> None:
        if self.available:
            self.store[key] = value
            self.ttls[key] = ttl_seconds

    def delete(self, *keys: str) -> None:
        if self.available:
            for key in keys:
                self.store.pop(key, None)


def _account() -> Account:
    return Account(
        id=ACCOUNT_ID,
        name="RevIntel Test Co",
        industry="Software",
        employee_count=250,
        annual_revenue=Decimal("1234567.50"),
        website=None,
        owner_id=None,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def cache() -> FakeCache:
    return FakeCache()


@pytest.fixture
def accounts() -> MagicMock:
    repo = MagicMock(spec=AccountRepository)
    repo.get.return_value = _account()
    return repo


@pytest.fixture
def service(cache: FakeCache, accounts: MagicMock) -> AccountService:
    return AccountService(
        MagicMock(spec=Session),
        accounts,
        MagicMock(spec=UserRepository),
        cache=AccountCache(cache, ttl_seconds=TTL),
    )


@pytest.fixture
def logs() -> io.StringIO:
    buffer = io.StringIO()
    configure_logging(environment="test", log_level="DEBUG", stream=buffer)
    return buffer


def _events(logs: io.StringIO) -> list[str]:
    return [json.loads(line)["event"] for line in logs.getvalue().splitlines() if line.strip()]


# --- AccountCache -------------------------------------------------------------------------


def test_key_format_is_namespaced_and_versioned() -> None:
    assert AccountCache.key(ACCOUNT_ID) == f"revintel:v1:account:{ACCOUNT_ID}"
    assert KEY_PREFIX == "revintel:v1:account:"


def test_round_trip_preserves_money_and_timestamps(cache: FakeCache) -> None:
    account_cache = AccountCache(cache, ttl_seconds=TTL)
    original = AccountRead.model_validate(_account())

    account_cache.set(original)
    restored = account_cache.get(ACCOUNT_ID)

    assert restored == original
    assert restored is not None and restored.annual_revenue == Decimal("1234567.50")
    assert restored.model_dump_json() == original.model_dump_json()
    stored = json.loads(cache.store[AccountCache.key(ACCOUNT_ID)])
    assert stored["annual_revenue"] == "1234567.50"  # JSON string, never a float
    assert cache.ttls[AccountCache.key(ACCOUNT_ID)] == TTL


@pytest.mark.parametrize(
    "raw",
    [b"not json{", b'{"id": "not-a-uuid"}', b'{"unexpected": "shape"}', b"\xff\xfe\x00"],
    ids=["invalid-json", "invalid-field", "wrong-schema", "invalid-utf8"],
)
def test_corrupt_entries_are_misses_and_removed(
    cache: FakeCache, logs: io.StringIO, raw: bytes
) -> None:
    key = AccountCache.key(ACCOUNT_ID)
    cache.store[key] = raw

    assert AccountCache(cache, ttl_seconds=TTL).get(ACCOUNT_ID) is None
    assert key not in cache.store
    assert "cache_entry_invalid" in _events(logs)


def test_invalidate_removes_the_entry(cache: FakeCache) -> None:
    account_cache = AccountCache(cache, ttl_seconds=TTL)
    account_cache.set(AccountRead.model_validate(_account()))

    account_cache.invalidate(ACCOUNT_ID)

    assert cache.store == {}


def test_zero_ttl_disables_writes(cache: FakeCache) -> None:
    AccountCache(cache, ttl_seconds=0).set(AccountRead.model_validate(_account()))

    assert cache.store == {}


# --- AccountService.get_account (cache-aside) ---------------------------------------------


def test_miss_loads_from_postgres_and_populates_cache(
    service: AccountService, cache: FakeCache, accounts: MagicMock, logs: io.StringIO
) -> None:
    result = service.get_account(ACCOUNT_ID)

    assert isinstance(result, AccountRead)
    assert result.name == "RevIntel Test Co"
    accounts.get.assert_called_once_with(ACCOUNT_ID)
    assert AccountCache.key(ACCOUNT_ID) in cache.store
    assert "cache_miss" in _events(logs)


def test_hit_does_not_touch_postgres(
    service: AccountService, accounts: MagicMock, logs: io.StringIO
) -> None:
    first = service.get_account(ACCOUNT_ID)
    accounts.get.reset_mock()

    second = service.get_account(ACCOUNT_ID)

    accounts.get.assert_not_called()
    assert second == first
    assert second.model_dump_json() == first.model_dump_json()
    assert "cache_hit" in _events(logs)


def test_not_found_is_not_cached(
    service: AccountService, cache: FakeCache, accounts: MagicMock
) -> None:
    accounts.get.return_value = None

    with pytest.raises(NotFoundError):
        service.get_account(ACCOUNT_ID)

    assert cache.store == {}


def test_unavailable_cache_falls_back_to_postgres(
    service: AccountService, cache: FakeCache, accounts: MagicMock
) -> None:
    cache.available = False

    for _ in range(2):
        assert service.get_account(ACCOUNT_ID).name == "RevIntel Test Co"

    assert accounts.get.call_count == 2


def test_cache_write_failure_still_returns_the_account(accounts: MagicMock) -> None:
    # RedisCache swallows write errors (see test_cache.py); model that as a cache whose
    # writes are silently lost.
    silent = FakeCache()
    silent.set = lambda *_: None  # type: ignore[method-assign]
    service = AccountService(
        MagicMock(spec=Session),
        accounts,
        MagicMock(spec=UserRepository),
        cache=AccountCache(silent, ttl_seconds=TTL),
    )

    assert service.get_account(ACCOUNT_ID).name == "RevIntel Test Co"
    assert service.get_account(ACCOUNT_ID).name == "RevIntel Test Co"
    assert accounts.get.call_count == 2


def test_corrupt_entry_is_replaced_from_postgres(
    service: AccountService, cache: FakeCache, accounts: MagicMock
) -> None:
    key = AccountCache.key(ACCOUNT_ID)
    cache.store[key] = b"garbage"

    result = service.get_account(ACCOUNT_ID)

    accounts.get.assert_called_once()
    assert AccountRead.model_validate_json(cache.store[key]) == result


def test_disabled_cache_always_reads_postgres(accounts: MagicMock) -> None:
    service = AccountService(
        MagicMock(spec=Session),
        accounts,
        MagicMock(spec=UserRepository),
        cache=AccountCache(NullCache(), ttl_seconds=0),
    )

    service.get_account(ACCOUNT_ID)
    service.get_account(ACCOUNT_ID)

    assert accounts.get.call_count == 2
