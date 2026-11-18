"""LLM provider abstraction for the CRM Copilot.

`LLMProvider` turns a prompt (instructions, input text and the evidence records it was
built from) into a structured `CopilotAnswer`. `OpenAILLMProvider`
(`app.ai.openai_provider`) is the V1 production implementation; `MockLLMProvider` is the
deterministic, offline provider for local development and tests. Prompting, citation
validation and retrieval never depend on which provider is used.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.models.enums import DocumentSourceType


class LLMError(RuntimeError):
    """The provider could not produce a usable answer (details are for logs only)."""


@dataclass(frozen=True)
class EvidenceRecord:
    """One retrieved CRM record as shown to the model."""

    source_type: DocumentSourceType
    source_id: uuid.UUID
    occurred_at: datetime
    content: str


@dataclass(frozen=True)
class CopilotPrompt:
    instructions: str
    input: str
    records: tuple[EvidenceRecord, ...]


class CopilotAnswer(BaseModel):
    """Structured model output. Citations are untrusted until validated."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="Answer text with inline [SOURCE:<source_id>] citations.")
    cited_source_ids: list[str] = Field(description="source_id of every cited record.")
    insufficient_evidence: bool = Field(
        description="True when the records do not contain enough evidence to answer."
    )


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, prompt: CopilotPrompt) -> CopilotAnswer: ...


MOCK_CITATIONS = 3


class MockLLMProvider:
    """Deterministic and offline: an extractive answer citing the top retrieved records."""

    name = "mock"

    def generate(self, prompt: CopilotPrompt) -> CopilotAnswer:
        if not prompt.records:
            return CopilotAnswer(
                answer="The CRM records do not contain enough evidence to answer this question.",
                cited_source_ids=[],
                insufficient_evidence=True,
            )
        cited = prompt.records[:MOCK_CITATIONS]
        facts = [
            f"- {' '.join(record.content.split())} [SOURCE:{record.source_id}]" for record in cited
        ]
        return CopilotAnswer(
            answer="Facts (mock answer, quoted from the most relevant records):\n"
            + "\n".join(facts),
            cited_source_ids=[str(record.source_id) for record in cited],
            insufficient_evidence=False,
        )


def create_llm_provider(settings: Settings) -> LLMProvider:
    """The configured provider. Creating it makes no network call."""
    if settings.llm_provider == "openai":
        from app.ai.openai_provider import OpenAILLMProvider

        return OpenAILLMProvider.from_settings(settings)
    return MockLLMProvider()
