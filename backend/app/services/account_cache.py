"""Account read cache: AccountRead JSON in Redis, keyed by account id.

Only the public response model is cached (never ORM objects, never pickle). Entries that
cannot be parsed are treated as misses and removed.

Invalidation: any future account mutation must call `invalidate()` only AFTER its
database transaction has committed; invalidating earlier lets a concurrent read put the
old value back.
"""

import uuid

import structlog
from pydantic import ValidationError

from app.core.cache import Cache
from app.schemas.account import AccountRead

# Bump the version when AccountRead changes incompatibly; old entries are then ignored
# and expire on their own.
KEY_PREFIX = "revintel:v1:account:"

logger = structlog.get_logger(__name__)


class AccountCache:
    def __init__(self, cache: Cache, *, ttl_seconds: int) -> None:
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def key(account_id: uuid.UUID) -> str:
        return f"{KEY_PREFIX}{account_id}"

    def get(self, account_id: uuid.UUID) -> AccountRead | None:
        key = self.key(account_id)
        raw = self._cache.get(key)
        if raw is None:
            logger.debug("cache_miss", key=key)
            return None
        try:
            account = AccountRead.model_validate_json(raw)
        except (ValidationError, ValueError):
            # Corrupt or outdated entry: never trust it; reload from PostgreSQL.
            logger.warning("cache_entry_invalid", key=key)
            self._cache.delete(key)
            return None
        logger.debug("cache_hit", key=key)
        return account

    def set(self, account: AccountRead) -> None:
        if self._ttl_seconds <= 0:
            return
        self._cache.set(
            self.key(account.id), account.model_dump_json().encode(), self._ttl_seconds
        )

    def invalidate(self, account_id: uuid.UUID) -> None:
        """Remove the cached account. Call only after a successful commit."""
        self._cache.delete(self.key(account_id))
