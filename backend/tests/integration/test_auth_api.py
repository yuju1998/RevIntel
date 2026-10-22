"""Authentication against PostgreSQL: login, /users/me, token lifecycle, CLI, protection."""

import io
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.cli import create_user as create_user_cli
from app.core.config import Settings
from app.core.exceptions import BusinessRuleError
from app.core.security import verify_password
from app.models import Account, Activity, Contact, Lead, Note, Opportunity, User
from app.repositories.user import UserRepository
from app.schemas.user import UserCreate
from app.services.user_service import UserService
from tests.integration.conftest import AUTH_USER_PASSWORD, bearer_headers
from tests.integration.factories import create_account, create_lead, create_user

pytestmark = pytest.mark.integration

LOGIN = "/api/v1/auth/login"
ME = "/api/v1/users/me"


def _login(client: TestClient, email: str, password: str):  # noqa: ANN202
    return client.post(LOGIN, json={"email": email, "password": password})


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_login_then_users_me(anon_client: TestClient, auth_user: User) -> None:
    response = _login(anon_client, "  API-User@Example.COM ", AUTH_USER_PASSWORD)

    assert response.status_code == 200
    token = response.json()["access_token"]
    assert response.json()["expires_in"] == 3600

    me = anon_client.get(ME, headers=_bearer(token))
    assert me.status_code == 200
    body = me.json()
    assert body["id"] == str(auth_user.id)
    assert body["email"] == "api-user@example.com"
    assert body["role"] == "SALES_REP" and body["is_active"] is True
    assert "password_hash" not in body and "$argon2" not in me.text


@pytest.mark.parametrize("case", ["unknown-email", "wrong-password", "inactive", "no-real-hash"])
def test_failed_logins_are_indistinguishable(
    anon_client: TestClient, db_session: Session, auth_user: User, case: str
) -> None:
    if case == "inactive":
        create_user(db_session, email="gone@example.com", password="gone password 123",
                    is_active=False)
    if case == "no-real-hash":
        create_user(db_session, email="placeholder@example.com")
    email, password = {
        "unknown-email": ("nobody@example.com", AUTH_USER_PASSWORD),
        "wrong-password": ("api-user@example.com", "not the password"),
        "inactive": ("gone@example.com", "gone password 123"),
        "no-real-hash": ("placeholder@example.com", "not-a-real-hash"),
    }[case]

    response = _login(anon_client, email, password)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json() == {"detail": "Incorrect email or password"}


def test_deactivated_user_token_stops_working(
    anon_client: TestClient, db_session: Session, auth_user: User
) -> None:
    token = _login(anon_client, "api-user@example.com", AUTH_USER_PASSWORD).json()["access_token"]
    assert anon_client.get(ME, headers=_bearer(token)).status_code == 200

    db_session.execute(text("UPDATE users SET is_active = false WHERE id = :id"),
                       {"id": auth_user.id})
    db_session.commit()

    assert anon_client.get(ME, headers=_bearer(token)).status_code == 401
    response = anon_client.post(
        "/api/v1/accounts", json={"name": "Blocked"}, headers=_bearer(token)
    )
    assert response.status_code == 401


def test_deleted_user_token_stops_working(
    anon_client: TestClient, db_session: Session, auth_user: User
) -> None:
    token = _login(anon_client, "api-user@example.com", AUTH_USER_PASSWORD).json()["access_token"]

    db_session.execute(text("DELETE FROM users WHERE id = :id"), {"id": auth_user.id})
    db_session.commit()

    assert anon_client.get(ME, headers=_bearer(token)).status_code == 401


def test_role_change_takes_effect_without_a_new_token(
    anon_client: TestClient, db_session: Session, auth_user: User
) -> None:
    token = _login(anon_client, "api-user@example.com", AUTH_USER_PASSWORD).json()["access_token"]

    db_session.execute(text("UPDATE users SET role = 'ADMIN' WHERE id = :id"),
                       {"id": auth_user.id})
    db_session.commit()

    # Role is read from PostgreSQL on every request, not from the token.
    assert anon_client.get(ME, headers=_bearer(token)).json()["role"] == "ADMIN"


def test_expired_token_is_rejected(
    anon_client: TestClient, auth_user: User, integration_settings: Settings
) -> None:
    from app.core.security import create_access_token

    token = create_access_token(
        auth_user.id,
        secret=integration_settings.jwt_secret,
        expires_minutes=1,
        now=datetime.now(UTC) - timedelta(minutes=5),
    )

    assert anon_client.get(ME, headers=_bearer(token.token)).status_code == 401


@pytest.fixture
def crm_ids(db_session: Session) -> dict[str, str]:
    account = create_account(db_session)
    opportunity = Opportunity(account_id=account.id, name="Deal", stage="PROSPECTING")
    db_session.add(opportunity)
    db_session.commit()
    return {
        "account": str(account.id),
        "lead": str(create_lead(db_session).id),
        "opportunity": str(opportunity.id),
    }


def _mutations(ids: dict[str, str]) -> list[tuple[str, str, dict]]:
    hour_ago = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    return [
        ("POST", "/api/v1/accounts", {"name": "Acme"}),
        ("POST", "/api/v1/contacts",
         {"account_id": ids["account"], "first_name": "A", "last_name": "B"}),
        ("POST", "/api/v1/leads",
         {"company_name": "X", "contact_name": "Y", "lead_source": "WEB"}),
        ("POST", "/api/v1/opportunities",
         {"account_id": ids["account"], "name": "Deal 2", "stage": "PROPOSAL"}),
        ("PATCH", f"/api/v1/opportunities/{ids['opportunity']}", {"stage": "NEGOTIATION"}),
        ("POST", "/api/v1/activities",
         {"lead_id": ids["lead"], "activity_type": "CALL", "direction": "INBOUND",
          "occurred_at": hour_ago}),
        ("POST", "/api/v1/notes", {"lead_id": ids["lead"], "body": "Hello"}),
    ]


def _row_counts(session: Session) -> dict[str, int]:
    return {
        model.__tablename__: session.scalar(select(func.count()).select_from(model)) or 0
        for model in (Account, Contact, Lead, Opportunity, Activity, Note)
    }


def test_every_mutation_requires_a_token_and_writes_nothing(
    anon_client: TestClient, db_session: Session, crm_ids: dict[str, str]
) -> None:
    before = _row_counts(db_session)

    for method, path, body in _mutations(crm_ids):
        response = anon_client.request(method, path, json=body)
        assert response.status_code == 401, (method, path)

    db_session.expire_all()
    assert _row_counts(db_session) == before
    stage = db_session.scalar(
        select(Opportunity.stage).where(Opportunity.id == uuid.UUID(crm_ids["opportunity"]))
    )
    assert stage == "PROSPECTING"


def test_every_mutation_succeeds_with_a_token(
    api_client: TestClient, crm_ids: dict[str, str]
) -> None:
    for method, path, body in _mutations(crm_ids):
        response = api_client.request(method, path, json=body)
        assert response.status_code in {200, 201}, (method, path, response.json())


def test_unauthenticated_invalid_body_gets_401_not_422(anon_client: TestClient) -> None:
    assert anon_client.post("/api/v1/accounts", json={"name": ""}).status_code == 401


def test_public_reads_work_anonymously(
    anon_client: TestClient, crm_ids: dict[str, str]
) -> None:
    for path in [
        "/api/v1/accounts",
        f"/api/v1/accounts/{crm_ids['account']}",
        f"/api/v1/accounts/{crm_ids['account']}/timeline",
        "/api/v1/leads",
        f"/api/v1/leads/{crm_ids['lead']}",
        f"/api/v1/leads/{crm_ids['lead']}/activities",
    ]:
        assert anon_client.get(path).status_code == 200, path


def _secrets(user: User, settings: Settings, session: Session, *extra: str) -> list[str]:
    stored_hash = session.scalar(select(User.password_hash).where(User.id == user.id))
    assert stored_hash
    return [AUTH_USER_PASSWORD, stored_hash, settings.jwt_secret.get_secret_value(), *extra]


def test_secrets_never_reach_info_level_logs(
    anon_client: TestClient,
    auth_user: User,
    integration_settings: Settings,
    crm_ids: dict[str, str],
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # INFO on sqlalchemy.engine = SQL statement logging (like echo=True); alembic.ini
    # normally keeps it at WARNING, so raise it to make sure SQL logging is exercised.
    with caplog.at_level(logging.DEBUG), caplog.at_level(logging.INFO, "sqlalchemy.engine"):
        token = _login(anon_client, "api-user@example.com", AUTH_USER_PASSWORD).json()[
            "access_token"
        ]
        _login(anon_client, "api-user@example.com", "a wrong secret password")
        anon_client.post(
            "/api/v1/notes", json={"lead_id": crm_ids["lead"], "body": "x"},
            headers=_bearer(token),
        )
        anon_client.get(ME, headers={"Authorization": "Bearer forged.token.value"})

    logged = caplog.text
    # Guard against a vacuous pass: SQL statements were logged, with parameters hidden.
    assert "INSERT INTO notes" in logged and "parameters hidden" in logged
    for secret in _secrets(
        auth_user, integration_settings, db_session,
        token, "a wrong secret password", "forged.token.value",
    ):
        assert secret not in logged


def test_authenticated_requests_never_load_the_password_hash(
    anon_client: TestClient,
    auth_user: User,
    integration_settings: Settings,
    crm_ids: dict[str, str],
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = _login(anon_client, "api-user@example.com", AUTH_USER_PASSWORD).json()["access_token"]

    # DEBUG on sqlalchemy.engine also logs result rows. password_hash is deferred, so
    # resolving the current user on every request never selects it.
    with caplog.at_level(logging.DEBUG), caplog.at_level(logging.DEBUG, "sqlalchemy.engine"):
        assert anon_client.get(ME, headers=_bearer(token)).status_code == 200
        anon_client.post(
            "/api/v1/notes", json={"lead_id": crm_ids["lead"], "body": "x"},
            headers=_bearer(token),
        )

    logged = caplog.text
    assert "FROM users" in logged and "Row (" in logged
    assert "password_hash" not in logged
    for secret in _secrets(auth_user, integration_settings, db_session, token):
        assert secret not in logged


# --- user creation (service + CLI) -------------------------------------------------------


def test_duplicate_email_in_other_casing_is_rejected(db_session: Session, auth_user: User) -> None:
    data = UserCreate(
        email="API-USER@example.com",
        full_name="Dup",
        role="ADMIN",
        password="another long password",
    )

    with pytest.raises(BusinessRuleError, match="already exists"):
        UserService(db_session, UserRepository(db_session)).create_user(data)


def test_cli_creates_a_user_who_can_log_in(
    anon_client: TestClient,
    db_session: Session,
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", migrated_database)
    password = "cli created password"
    answers = iter([password, password])

    exit_code = create_user_cli.main(
        ["--email", "  New.Admin@Example.com ", "--full-name", "New Admin", "--role", "ADMIN"],
        stdin=io.StringIO(""),
        prompt=lambda _: next(answers),
    )

    assert exit_code == 0
    assert password not in capsys.readouterr().out
    user = db_session.scalar(select(User).where(User.email == "new.admin@example.com"))
    assert user is not None and user.role == "ADMIN"
    assert verify_password(password, user.password_hash)[0] is True
    assert _login(anon_client, "NEW.ADMIN@example.com", password).status_code == 200


def test_cli_rejects_duplicate_email(
    db_session: Session,
    auth_user: User,
    migrated_database: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", migrated_database)

    exit_code = create_user_cli.main(
        ["--email", "Api-User@example.com", "--full-name", "Dup", "--role", "SALES_REP",
         "--password-stdin"],
        stdin=io.StringIO("some long password\n"),
    )

    assert exit_code == 1
    assert "already exists" in capsys.readouterr().err


def test_bearer_headers_helper_matches_login(
    anon_client: TestClient, auth_user: User, integration_settings: Settings
) -> None:
    # The fixture-issued token is accepted like a login-issued one.
    assert anon_client.get(ME, headers=bearer_headers(auth_user, integration_settings)).json()[
        "id"
    ] == str(auth_user.id)
