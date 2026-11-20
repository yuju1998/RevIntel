"""Deterministic RAG evaluation of the CRM Copilot (no LLM-as-judge).

    PYTHONPATH=.:backend python -m ml.evaluation.rag_eval \\
        --database-url postgresql+psycopg://...@localhost:5432/revintel_ml [--provider mock|openai]

Runs every case in data/rag_eval/questions.json through the real AssistantQueryService
(account-scoped retrieval, prompt, provider, citation validation) against an indexed copy
of the synthetic CRM dataset, read-only.

Per case i: E = expected source IDs, R = the top-k retrieved source IDs (deduplicated, in
rank order), C = the returned (validated) citations, M = the model's raw citations.

- Retrieval Recall@k  = |E ∩ R| / |E|            (cases with E non-empty; |E| <= k)
- Citation Precision  = |C ∩ E| / |C|            (cases with E and C non-empty)
- Citation Coverage   = |C ∩ E| / |E ∩ R|        (cases with E ∩ R non-empty)
- Cross-account leakage rate = leaked items / all items, over all cases, where items are
  R ∪ C ∪ the [SOURCE:<id>] markers in the returned answer, and an item is leaked when its
  owning account (looked up independently in document_chunks) is not the case's account
  or is unknown. Must be 0: the run fails otherwise.
- Abstention Accuracy = share of no-evidence cases (E empty) answered with
  insufficient_evidence = true. These cases are judged only by this metric, never by
  Citation Precision (with E empty no citation can be relevant).
- Evidence-case Abstention Rate = share of evidence-bearing cases (E non-empty) answered
  with insufficient_evidence = true (false abstentions; lower is better). Reported only.
- Invalid Citation Rate = |M \\ R| / |M|, micro over all cases.

Recall, Precision and Coverage are macro means over the cases where they are defined;
undefined cases are excluded and counted. Only leakage is a gate.
"""

import argparse
import json
import os
import re
import sys
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = ROOT / "data" / "rag_eval" / "questions.json"
_MARKER = re.compile(r"\[SOURCE:([^\]]*)\]")


# --- metrics (pure) ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    account_id: str
    expected: list[str]
    retrieved: list[str]  # top-k, deduplicated, rank order
    cited: list[str]  # validated, returned citations
    model_citations: list[str]  # raw, before validation
    answer_markers: list[str]  # [SOURCE:<id>] markers in the returned answer
    insufficient_evidence: bool
    leaked_items: int
    total_items: int


def recall_at_k(expected: Iterable[str], retrieved: Sequence[str], k: int) -> float | None:
    expected_set = set(expected)
    if not expected_set:
        return None
    return len(expected_set & set(retrieved[:k])) / len(expected_set)


def citation_precision(cited: Iterable[str], expected: Iterable[str]) -> float | None:
    # Undefined without expected sources: no-evidence cases are judged by abstention.
    cited_set, expected_set = set(cited), set(expected)
    if not cited_set or not expected_set:
        return None
    return len(cited_set & expected_set) / len(cited_set)


def citation_coverage(
    cited: Iterable[str], expected: Iterable[str], retrieved: Sequence[str], k: int
) -> float | None:
    available = set(expected) & set(retrieved[:k])
    if not available:
        return None
    return len(set(cited) & available) / len(available)


def leakage_counts(
    items: Iterable[str], owners: dict[str, set[str]], account_id: str
) -> tuple[int, int]:
    """(leaked, total) distinct items; unknown ownership counts as leaked (fail-safe)."""
    unique = set(items)
    leaked = sum(1 for item in unique if owners.get(item, set()) != {account_id})
    return leaked, len(unique)


def _mean(values: list[float | None]) -> tuple[float | None, int]:
    defined = [value for value in values if value is not None]
    return (round(sum(defined) / len(defined), 4) if defined else None), len(defined)


def aggregate(outcomes: Sequence[CaseOutcome], k: int) -> dict[str, Any]:
    recall, recall_n = _mean([recall_at_k(o.expected, o.retrieved, k) for o in outcomes])
    precision, precision_n = _mean([citation_precision(o.cited, o.expected) for o in outcomes])
    coverage, coverage_n = _mean(
        [citation_coverage(o.cited, o.expected, o.retrieved, k) for o in outcomes]
    )
    leaked = sum(o.leaked_items for o in outcomes)
    items = sum(o.total_items for o in outcomes)
    raw = sum(len(o.model_citations) for o in outcomes)
    invalid = sum(
        sum(1 for c in o.model_citations if _canonical(c) not in set(o.retrieved)) for o in outcomes
    )
    no_evidence = [o for o in outcomes if not o.expected]
    with_evidence = [o for o in outcomes if o.expected]
    return {
        "cases": len(outcomes),
        f"retrieval_recall_at_{k}": recall,
        f"retrieval_recall_at_{k}_cases": recall_n,
        "citation_precision": precision,
        "citation_precision_cases": precision_n,
        "citation_coverage": coverage,
        "citation_coverage_cases": coverage_n,
        "cross_account_leakage_rate": round(leaked / items, 4) if items else 0.0,
        "cross_account_leaked_items": leaked,
        "cross_account_items": items,
        "invalid_citation_rate": round(invalid / raw, 4) if raw else None,
        "abstention_accuracy": (
            round(sum(o.insufficient_evidence for o in no_evidence) / len(no_evidence), 4)
            if no_evidence
            else None
        ),
        "abstention_accuracy_cases": len(no_evidence),
        "evidence_case_abstention_rate": (
            round(sum(o.insufficient_evidence for o in with_evidence) / len(with_evidence), 4)
            if with_evidence
            else None
        ),
        "evidence_case_abstention_cases": len(with_evidence),
    }


def _canonical(raw: str) -> str | None:
    try:
        return str(uuid.UUID(str(raw).strip()))
    except ValueError:
        return None


# --- running the pipeline ---------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    account_id: uuid.UUID
    question: str
    expected: list[str]


def load_cases(path: Path) -> tuple[dict[str, Any], list[EvalCase]]:
    document = json.loads(path.read_text())
    cases = []
    for raw in document["cases"]:
        expected = [str(uuid.UUID(s["source_id"])) for s in raw["expected_sources"]]
        if len(expected) > document["retrieval_k"]:
            raise ValueError(f"{raw['case_id']}: more expected sources than retrieval_k")
        cases.append(
            EvalCase(raw["case_id"], uuid.UUID(raw["account_id"]), raw["question"], expected)
        )
    return document, cases


def source_owners(session: Session, source_ids: Iterable[str]) -> dict[str, set[str]]:
    """Owning account(s) of each source, read from document_chunks independently."""
    from app.models.document_chunk import DocumentChunk

    ids = [uuid.UUID(i) for i in {i for i in source_ids if _canonical(i)}]
    owners: dict[str, set[str]] = {}
    if ids:
        rows = session.execute(
            select(DocumentChunk.source_id, DocumentChunk.account_id)
            .where(DocumentChunk.source_id.in_(ids))
            .distinct()
        )
        for source_id, account_id in rows:
            owners.setdefault(str(source_id), set()).add(str(account_id))
    return owners


def run_case(session: Session, service: Any, case: EvalCase, k: int) -> CaseOutcome:
    result = service.query(case.account_id, case.question)
    retrieved = list(dict.fromkeys(str(chunk.source_id) for chunk in result.retrieved))[:k]
    cited = [str(source.source_id) for source in result.sources]
    markers = [m for m in (_canonical(x) for x in _MARKER.findall(result.answer)) if m]
    items = [*retrieved, *cited, *markers]
    leaked, total = leakage_counts(items, source_owners(session, items), str(case.account_id))
    return CaseOutcome(
        case_id=case.case_id,
        account_id=str(case.account_id),
        expected=case.expected,
        retrieved=retrieved,
        cited=cited,
        model_citations=list(result.model_citations),
        answer_markers=markers,
        insufficient_evidence=result.insufficient_evidence,
        leaked_items=leaked,
        total_items=total,
    )


def _llm_provider(name: str, model_override: str | None) -> Any:
    """The evaluated provider, configured from the same LLM settings as the API.

    `model_override` (--openai-model) replaces OPENAI_MODEL for this evaluation only.
    """
    from app.ai.llm import MockLLMProvider

    if name == "mock":
        return MockLLMProvider()
    from app.ai.openai_provider import OpenAILLMProvider
    from app.core.config import get_llm_settings

    settings = get_llm_settings()
    if model_override:
        settings = settings.model_copy(update={"openai_model": model_override})
    return OpenAILLMProvider.from_settings(settings)


def _check_dataset(session: Session, document: dict[str, Any], cases: list[EvalCase]) -> None:
    from app.models.document_chunk import DocumentChunk
    from ml.features.build_features import DEFAULT_MANIFEST, SourceManifest
    from ml.features.sources import source_row_counts

    manifest = SourceManifest.read(DEFAULT_MANIFEST)
    if manifest.dataset_sha256 != document["dataset_sha256"]:
        raise ValueError("questions.json was written for a different synthetic dataset")
    counts = source_row_counts(session.connection())
    if counts != manifest.row_counts:
        raise ValueError(f"database row counts {counts} do not match the manifest")
    for case in cases:
        for source_id in case.expected:
            owner = session.scalars(
                select(DocumentChunk.account_id).where(
                    DocumentChunk.source_id == uuid.UUID(source_id),
                    DocumentChunk.embedding_model == document["embedding_model"],
                )
            ).first()
            if owner != case.account_id:
                raise ValueError(
                    f"{case.case_id}: expected source {source_id} is not indexed for the "
                    "account (run python -m app.cli.index_crm against the dataset first)"
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic RAG evaluation.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--database-url", default=os.environ.get("SYNTHETIC_DATABASE_URL"))
    parser.add_argument("--provider", choices=("mock", "openai"), default="mock")
    parser.add_argument(
        "--openai-model",
        help="override OPENAI_MODEL for this evaluation (default: the LLM settings' model)",
    )
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)
    if not args.database_url:
        print("Error: pass --database-url or set SYNTHETIC_DATABASE_URL", file=sys.stderr)
        return 2

    from app.ai.embeddings import SentenceTransformerProvider
    from app.core.config import EMBEDDING_DIMENSION, DatabaseSettings
    from app.db.session import create_db_engine
    from app.repositories.account import AccountRepository
    from app.repositories.semantic_search import SemanticSearchRepository
    from app.services.assistant_query_service import AssistantQueryService
    from app.services.semantic_search_service import SemanticSearchService
    from ml.features.sources import require_source_database

    try:
        require_source_database(args.database_url)
        document, cases = load_cases(args.questions)
        llm = _llm_provider(args.provider, args.openai_model)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    k = document["retrieval_k"]
    engine = create_db_engine(
        DatabaseSettings(_env_file=None, database_url=args.database_url)
    ).execution_options(postgresql_readonly=True)
    embeddings = SentenceTransformerProvider(document["embedding_model"], EMBEDDING_DIMENSION)
    try:
        with Session(engine) as session:
            try:
                _check_dataset(session, document, cases)
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 2
            search = SemanticSearchService(
                AccountRepository(session), SemanticSearchRepository(session), embeddings
            )
            service = AssistantQueryService(search, llm)
            outcomes = [run_case(session, service, case, k) for case in cases]
    finally:
        engine.dispose()

    metrics = aggregate(outcomes, k)
    report = {"provider": llm.name, "metrics": metrics, "cases": [asdict(o) for o in outcomes]}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    for outcome in outcomes:
        hit = len(set(outcome.expected) & set(outcome.retrieved))
        print(
            f"  {outcome.case_id:<44} expected {len(outcome.expected)} retrieved-hit {hit} "
            f"cited {len(outcome.cited)} insufficient={outcome.insufficient_evidence}"
        )
    if metrics["cross_account_leaked_items"]:
        print("FAIL: cross-account leakage detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
