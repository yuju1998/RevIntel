"""OpenAI implementation of `LLMProvider` (the only module that uses the OpenAI SDK).

Uses the Responses API with structured output (`text_format=CopilotAnswer`), no tools,
no conversation state and `store=False`. Low variance: temperature 0 unless configured
otherwise. Every SDK failure, refusal or incomplete response becomes `LLMError`; logs carry
only the error type, HTTP status and request ID, never the API key or the prompt.
"""

from typing import Any

import openai
import structlog

from app.ai.llm import CopilotAnswer, CopilotPrompt, LLMError
from app.core.config import LLMSettings

logger = structlog.get_logger("app.assistant.openai")


class OpenAILLMProvider:
    name = "openai"

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        # Required: None omits the parameter (models that reject it); never implicit.
        temperature: float | None,
        max_output_tokens: int = 800,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    @classmethod
    def from_settings(cls, settings: LLMSettings) -> OpenAILLMProvider:
        """The provider configured exactly as the API configures it (LLM_* / OPENAI_* settings)."""
        if settings.openai_api_key is None or not settings.openai_model:
            raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required for LLM_PROVIDER=openai")
        client = openai.OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        return cls(
            client,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_output_tokens=settings.openai_max_output_tokens,
        )

    def generate(self, prompt: CopilotPrompt) -> CopilotAnswer:
        options: dict[str, Any] = {
            "model": self._model,
            "instructions": prompt.instructions,
            "input": prompt.input,
            "text_format": CopilotAnswer,
            "max_output_tokens": self._max_output_tokens,
            "store": False,
        }
        if self._temperature is not None:
            options["temperature"] = self._temperature
        try:
            response = self._client.responses.parse(**options)
        except openai.OpenAIError as exc:
            details = {
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "request_id": getattr(exc, "request_id", None),
            }
            logger.warning("openai_request_failed", model=self._model, **details)
            raise LLMError(f"OpenAI request failed: {details}") from exc

        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, CopilotAnswer):
            incomplete = getattr(getattr(response, "incomplete_details", None), "reason", None)
            logger.warning(
                "openai_no_structured_answer",
                model=self._model,
                status=getattr(response, "status", None),
                incomplete_reason=incomplete,
            )
            raise LLMError(
                f"OpenAI returned no structured answer (status={getattr(response, 'status', None)},"
                f" incomplete_reason={incomplete}); possibly a refusal"
            )
        return parsed
