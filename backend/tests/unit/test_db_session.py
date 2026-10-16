from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import SessionDep


@pytest.fixture
def session() -> MagicMock:
    return MagicMock(spec=Session)


@pytest.fixture
def session_factory(session: MagicMock) -> MagicMock:
    return MagicMock(return_value=session)


@pytest.fixture
def client(session_factory: MagicMock) -> TestClient:
    app = FastAPI()
    app.state.session_factory = session_factory

    @app.get("/ok")
    def ok(db: SessionDep) -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    def boom(db: SessionDep) -> None:
        raise RuntimeError("handler failed")

    @app.get("/same-session")
    def same_session(first: SessionDep, second: SessionDep) -> dict[str, bool]:
        return {"same": first is second}

    return TestClient(app, raise_server_exceptions=False)


def test_successful_request_closes_without_commit_or_rollback(
    client: TestClient, session: MagicMock
) -> None:
    response = client.get("/ok")

    assert response.status_code == 200
    session.close.assert_called_once()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


def test_failed_request_rolls_back_and_closes(client: TestClient, session: MagicMock) -> None:
    response = client.get("/boom")

    assert response.status_code == 500
    session.rollback.assert_called_once()
    session.close.assert_called_once()
    session.commit.assert_not_called()


def test_one_session_per_request(client: TestClient, session_factory: MagicMock) -> None:
    response = client.get("/same-session")
    assert response.json() == {"same": True}
    assert session_factory.call_count == 1

    client.get("/ok")
    assert session_factory.call_count == 2
