"""Password hashing and JWT access tokens."""

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pydantic import SecretStr

from app.core.emails import normalize_user_email
from app.core.exceptions import AuthenticationError
from app.core.security import (
    JWT_AUDIENCE,
    JWT_ISSUER,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_dummy_password,
    verify_password,
)

SECRET = SecretStr("unit-test-jwt-secret-0123456789abcdef")
OTHER_SECRET = SecretStr("another-test-secret-0123456789abcdefgh")
USER_ID = uuid.uuid4()


def test_hash_is_argon2id_and_verifies() -> None:
    password_hash = hash_password("correct horse battery staple")

    assert password_hash.startswith("$argon2id$")
    assert "correct horse" not in password_hash
    assert verify_password("correct horse battery staple", password_hash) == (True, None)
    assert verify_password("wrong", password_hash) == (False, None)


def test_hashes_are_salted() -> None:
    assert hash_password("same password") != hash_password("same password")


def test_outdated_hash_is_upgraded() -> None:
    weak_hasher = Argon2Hasher(time_cost=1, memory_cost=1024, parallelism=1)
    weak = PasswordHash((weak_hasher,)).hash("pw-123456789")

    valid, upgraded = verify_password("pw-123456789", weak)

    assert valid is True
    assert upgraded is not None and upgraded != weak


def test_non_argon2_hash_never_matches() -> None:
    assert verify_password("anything", "not-a-real-hash") == (False, None)


def test_dummy_verification_runs_without_error() -> None:
    verify_dummy_password("whatever")


def test_user_email_normalization() -> None:
    assert normalize_user_email("  Ada.Lovelace@Example.COM \n") == "ada.lovelace@example.com"


def test_token_round_trip_and_claims() -> None:
    now = datetime.now(UTC)
    token = create_access_token(USER_ID, secret=SECRET, expires_minutes=60, now=now)

    assert token.expires_in == 3600
    assert decode_access_token(token.token, secret=SECRET) == USER_ID
    claims = jwt.decode(
        token.token, SECRET.get_secret_value(), algorithms=["HS256"], audience=JWT_AUDIENCE
    )
    assert set(claims) == {"sub", "iat", "exp", "iss", "aud", "type"}
    assert claims["sub"] == str(USER_ID)
    assert claims["iss"] == JWT_ISSUER and claims["type"] == "access"
    assert claims["exp"] - claims["iat"] == 3600
    assert jwt.get_unverified_header(token.token)["alg"] == "HS256"


def test_token_is_hidden_in_repr() -> None:
    token = create_access_token(USER_ID, secret=SECRET, expires_minutes=5)

    assert token.token not in repr(token)


def _claims(**overrides: object) -> dict:
    now = datetime.now(UTC)
    claims: dict = {
        "sub": str(USER_ID),
        "iat": now,
        "exp": now + timedelta(minutes=5),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "type": "access",
    }
    claims.update(overrides)
    return {key: value for key, value in claims.items() if value is not None}


def _sign(claims: dict, *, secret: SecretStr = SECRET, algorithm: str = "HS256") -> str:
    return jwt.encode(claims, secret.get_secret_value(), algorithm=algorithm)


@pytest.mark.parametrize(
    "token",
    [
        pytest.param(lambda: "not-a-jwt", id="garbage"),
        pytest.param(lambda: "", id="empty"),
        pytest.param(lambda: _sign(_claims(), secret=OTHER_SECRET), id="other-secret"),
        pytest.param(lambda: _sign(_claims()) + "x", id="tampered-signature"),
        pytest.param(
            lambda: _sign(_claims(), algorithm="HS512"),
            id="hs512",
            # The attacker-side token is signed with a short key; expected for this case.
            marks=pytest.mark.filterwarnings("ignore::jwt.warnings.InsecureKeyLengthWarning"),
        ),
        pytest.param(lambda: jwt.encode(_claims(), None, algorithm="none"), id="alg-none"),
        pytest.param(
            lambda: _sign(_claims(exp=datetime.now(UTC) - timedelta(seconds=1))), id="expired"
        ),
        pytest.param(
            lambda: _sign(_claims(iat=datetime.now(UTC) + timedelta(hours=1))), id="future-iat"
        ),
        pytest.param(lambda: _sign(_claims(sub=None)), id="missing-sub"),
        pytest.param(lambda: _sign(_claims(exp=None)), id="missing-exp"),
        pytest.param(lambda: _sign(_claims(iat=None)), id="missing-iat"),
        pytest.param(lambda: _sign(_claims(type=None)), id="missing-type"),
        pytest.param(lambda: _sign(_claims(iss="someone-else")), id="wrong-issuer"),
        pytest.param(lambda: _sign(_claims(aud="other-api")), id="wrong-audience"),
        pytest.param(lambda: _sign(_claims(type="refresh")), id="wrong-type"),
        pytest.param(lambda: _sign(_claims(sub="not-a-uuid")), id="sub-not-uuid"),
    ],
)
def test_invalid_tokens_are_rejected_with_a_generic_error(token: object) -> None:
    raw = token()  # type: ignore[operator]

    with pytest.raises(AuthenticationError) as exc_info:
        decode_access_token(raw, secret=SECRET)

    assert exc_info.value.message == "Could not validate credentials"
    assert exc_info.value.__cause__ is None
    if raw:
        assert raw not in str(exc_info.value)
