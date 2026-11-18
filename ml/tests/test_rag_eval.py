"""RAG evaluation metrics (hand-computed cases) and the committed question set."""

import json
import uuid
from pathlib import Path

import pytest

from ml.evaluation.rag_eval import (
    DEFAULT_QUESTIONS,
    CaseOutcome,
    aggregate,
    citation_coverage,
    citation_precision,
    leakage_counts,
    load_cases,
    recall_at_k,
)

A, B, C, D, E, F = (str(uuid.uuid4()) for _ in range(6))
ACCOUNT, OTHER = str(uuid.uuid4()), str(uuid.uuid4())


def test_recall_at_k() -> None:
    assert recall_at_k([A, B], [A, C, D, E, F], 5) == 0.5
    assert recall_at_k([A, B], [C, D, E, F, A, B], 5) == 0.5  # B is at rank 6
    assert recall_at_k([A], [A], 5) == 1.0
    assert recall_at_k([], [A], 5) is None  # undefined, excluded


def test_citation_precision() -> None:
    assert citation_precision([A, C, D], [A, B]) == pytest.approx(1 / 3)
    assert citation_precision([A, A, B], [A, B]) == 1.0  # duplicates count once
    assert citation_precision([], [A]) is None
    assert citation_precision([A, B], []) is None  # no-evidence case: judged by abstention


def test_citation_coverage_only_counts_expected_sources_the_model_could_see() -> None:
    # Expected A, B; only A was retrieved, and the model cited it: full coverage.
    assert citation_coverage([A], [A, B], [A, C, D], 5) == 1.0
    assert citation_coverage([C], [A, B], [A, B, C], 5) == 0.0
    assert citation_coverage([A], [A, B], [C, D], 5) is None  # nothing expected was retrieved


def test_leakage_counts_other_or_unknown_owners() -> None:
    owners = {A: {ACCOUNT}, B: {OTHER}, C: {ACCOUNT, OTHER}}

    assert leakage_counts([A, A], owners, ACCOUNT) == (0, 1)
    assert leakage_counts([A, B, C, D], owners, ACCOUNT) == (3, 4)  # D unknown: fail-safe


def _outcome(**fields: object) -> CaseOutcome:
    values: dict = {
        "case_id": "c",
        "account_id": ACCOUNT,
        "expected": [],
        "retrieved": [],
        "cited": [],
        "model_citations": [],
        "answer_markers": [],
        "insufficient_evidence": False,
        "leaked_items": 0,
        "total_items": 0,
    }
    return CaseOutcome(**{**values, **fields})


def test_aggregate_macro_means_and_gates() -> None:
    outcomes = [
        # recall 1/2, precision 1/2, coverage 1/1
        _outcome(
            expected=[A, B],
            retrieved=[A, C],
            cited=[A, C],
            model_citations=[A, C, E],
            total_items=2,
        ),
        # recall 1, precision 1, coverage 1; a false abstention despite evidence
        _outcome(
            expected=[D],
            retrieved=[D],
            cited=[D],
            model_citations=[D],
            insufficient_evidence=True,
            total_items=1,
        ),
        # no expected sources, abstained while citing what it checked: excluded from
        # precision (would have scored 0), counted by abstention accuracy
        _outcome(
            retrieved=[F], cited=[F], model_citations=[F], insufficient_evidence=True, total_items=1
        ),
        # no expected sources, answered anyway: an abstention miss
        _outcome(retrieved=[F], total_items=1),
    ]

    metrics = aggregate(outcomes, 5)

    assert metrics["retrieval_recall_at_5"] == 0.75 and metrics["retrieval_recall_at_5_cases"] == 2
    assert metrics["citation_precision"] == 0.75 and metrics["citation_precision_cases"] == 2
    assert metrics["citation_coverage"] == 1.0 and metrics["citation_coverage_cases"] == 2
    assert metrics["invalid_citation_rate"] == 0.2  # 1 of 5 raw citations (E) not retrieved
    assert metrics["abstention_accuracy"] == 0.5 and metrics["abstention_accuracy_cases"] == 2
    assert metrics["evidence_case_abstention_rate"] == 0.5  # 1 of 2 evidence-bearing cases
    assert metrics["evidence_case_abstention_cases"] == 2
    assert metrics["cross_account_leakage_rate"] == 0.0

    leaky = aggregate([_outcome(leaked_items=1, total_items=4)], 5)
    assert leaky["cross_account_leakage_rate"] == 0.25 and leaky["cross_account_leaked_items"] == 1


def test_committed_question_set_is_well_formed() -> None:
    document, cases = load_cases(DEFAULT_QUESTIONS)

    assert document["retrieval_k"] == 5
    assert document["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len({case.case_id for case in cases}) == len(cases) >= 10
    assert any(not case.expected for case in cases)  # insufficient-evidence cases
    raw = json.loads(DEFAULT_QUESTIONS.read_text())["cases"]
    assert all(entry["rationale"] for entry in raw)  # every case is justified


def test_cases_with_more_expected_sources_than_k_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    sources = [{"source_type": "ACTIVITY", "source_id": str(uuid.uuid4())} for _ in range(6)]
    path.write_text(
        json.dumps(
            {
                "retrieval_k": 5,
                "cases": [
                    {
                        "case_id": "x",
                        "account_id": ACCOUNT,
                        "question": "q",
                        "expected_sources": sources,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="retrieval_k"):
        load_cases(path)
