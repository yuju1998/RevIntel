import uuid
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import inspect, text

from app.api.deps import SessionDep
from app.core.config import Settings
from app.main import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
def table_name() -> str:
    return f"get_db_probe_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def app(integration_settings: Settings, table_name: str) -> FastAPI:
    application = create_app(integration_settings)

    @application.get("/_test/current-database")
    def current_database(db: SessionDep) -> dict[str, str]:
        return {"database": db.execute(text("SELECT current_database()")).scalar_one()}

    # PostgreSQL DDL is transactional, so table existence shows whether work was committed.
    @application.post("/_test/create-table")
    def create_table(db: SessionDep) -> dict[str, bool]:
        db.execute(text(f"CREATE TABLE {table_name} (id integer)"))
        return {"created": True}

    @application.post("/_test/create-table-then-fail")
    def create_table_then_fail(db: SessionDep) -> None:
        db.execute(text(f"CREATE TABLE {table_name} (id integer)"))
        raise RuntimeError("handler failed")

    return application


@pytest.fixture
def client(app: FastAPI, table_name: str) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
        # Clean up in case a regression ever lets the table be committed.
        with app.state.engine.begin() as connection:
            connection.execute(text(f"DROP TABLE IF EXISTS {table_name}"))


def _table_exists(app: FastAPI, table_name: str) -> bool:
    return inspect(app.state.engine).has_table(table_name)


def test_session_queries_test_database(client: TestClient) -> None:
    response = client.get("/_test/current-database")

    assert response.status_code == 200
    assert response.json() == {"database": "revintel_test"}


def test_successful_request_is_not_committed(
    client: TestClient, app: FastAPI, table_name: str
) -> None:
    response = client.post("/_test/create-table")

    assert response.status_code == 200
    assert not _table_exists(app, table_name)


def test_failed_request_is_rolled_back(client: TestClient, app: FastAPI, table_name: str) -> None:
    response = client.post("/_test/create-table-then-fail")

    assert response.status_code == 500
    assert not _table_exists(app, table_name)
