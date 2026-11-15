"""The RAG evaluation configures OpenAI exactly as the API does (mocked client, no network)."""

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.ai.copilot import build_prompt
from app.ai.llm import CopilotAnswer, create_llm_provider
from app.core.config import Settings, get_llm_settings
from ml.evaluation import rag_eval

LLM_ENV_VARS = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TEMPERATURE",
    "OPENAI_MAX_OUTPUT_TOKENS",
    "OPENAI_TIMEOUT_SECONDS",
    "OPENAI_MAX_RETRIES",
)
BASE_ENV = "OPENAI_API_KEY=sk-eval-key\nOPENAI_MODEL=configured-model\n"
APP_ENV = (
    "DATABASE_URL=postgresql+psycopg://u:p@localhost:5432/revintel\n"
    "REDIS_URL=redis://localhost:6379/0\n"
    f"JWT_SECRET={'x' * 40}\n"
    "LLM_PROVIDER=openai\n"
)


@pytest.fixture
def write_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Callable[[str], None]]:
    """Run in a directory whose `.env` is the given text; no LLM variables in os.environ."""
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_llm_settings.cache_clear()
    yield lambda text: (tmp_path / ".env").write_text(text)
    get_llm_settings.cache_clear()


@pytest.fixture
def openai_class() -> Iterator[MagicMock]:
    answer = CopilotAnswer(answer="a", cited_source_ids=[], insufficient_evidence=True)
    with patch("app.ai.openai_provider.openai.OpenAI") as client_class:
        client_class.return_value.responses.parse.return_value = SimpleNamespace(
            output_parsed=answer, status="completed"
        )
        yield client_class


def _request(provider: Any, openai_class: MagicMock) -> dict[str, Any]:
    """The responses.parse arguments the provider sends (prompt and schema left out)."""
    openai_class.return_value.responses.parse.reset_mock()
    provider.generate(build_prompt("q", []))
    kwargs = openai_class.return_value.responses.parse.call_args.kwargs
    return {k: v for k, v in kwargs.items() if k not in ("instructions", "input", "text_format")}


def test_empty_temperature_is_omitted(
    write_env: Callable[[str], None], openai_class: MagicMock
) -> None:
    write_env(BASE_ENV + "OPENAI_TEMPERATURE=\n")

    request = _request(rag_eval._llm_provider("openai", None), openai_class)

    assert "temperature" not in request


def test_explicit_temperature_is_sent(
    write_env: Callable[[str], None], openai_class: MagicMock
) -> None:
    write_env(BASE_ENV + "OPENAI_TEMPERATURE=0.3\n")

    request = _request(rag_eval._llm_provider("openai", None), openai_class)

    assert request["temperature"] == 0.3


@pytest.mark.parametrize("temperature", ["", "0", "0.7"], ids=["omitted", "zero", "set"])
def test_api_and_evaluation_send_identical_requests(
    write_env: Callable[[str], None], openai_class: MagicMock, temperature: str
) -> None:
    write_env(APP_ENV + BASE_ENV + f"OPENAI_TEMPERATURE={temperature}\n")

    api = create_llm_provider(Settings())  # what the FastAPI lifespan builds
    api_client_kwargs = openai_class.call_args.kwargs
    evaluation = rag_eval._llm_provider("openai", None)

    assert openai_class.call_args.kwargs == api_client_kwargs  # key, timeout, retries
    assert _request(evaluation, openai_class) == _request(api, openai_class)


def test_evaluation_reads_every_openai_setting(
    write_env: Callable[[str], None], openai_class: MagicMock
) -> None:
    write_env(
        BASE_ENV
        + "OPENAI_MAX_OUTPUT_TOKENS=321\nOPENAI_TIMEOUT_SECONDS=12.5\nOPENAI_MAX_RETRIES=4\n"
    )

    request = _request(rag_eval._llm_provider("openai", None), openai_class)

    openai_class.assert_called_once_with(api_key="sk-eval-key", timeout=12.5, max_retries=4)
    assert request["model"] == "configured-model"
    assert request["max_output_tokens"] == 321
    assert request["temperature"] == 0.0  # the settings default when not configured


def test_openai_model_flag_overrides_only_the_model(
    write_env: Callable[[str], None], openai_class: MagicMock
) -> None:
    write_env(BASE_ENV + "OPENAI_TEMPERATURE=\nOPENAI_MAX_OUTPUT_TOKENS=321\n")

    request = _request(rag_eval._llm_provider("openai", "override-model"), openai_class)

    assert request["model"] == "override-model"
    assert request["max_output_tokens"] == 321 and "temperature" not in request


def test_missing_key_or_model_is_a_setup_error_before_any_request(
    write_env: Callable[[str], None], openai_class: MagicMock
) -> None:
    write_env("OPENAI_MODEL=configured-model\n")

    exit_code = rag_eval.main(
        [
            "--database-url",
            "postgresql+psycopg://u:p@127.0.0.1:1/revintel_ml",
            "--provider",
            "openai",
        ]
    )

    assert exit_code == 2
    openai_class.assert_not_called()
