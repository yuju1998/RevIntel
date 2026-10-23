"""Summary statistics, sanity counters, dataset hash and manifest."""

import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

import faker
import numpy as np

from app.models.enums import ActivityDirection, ActivityType, LeadStatus
from ml.synthetic.config import GENERATOR_VERSION
from ml.synthetic.conventions import (
    OBSERVATION_END_CONVENTION,
    TERMINAL_TIME_CONVENTION,
    lead_terminal_time,
)
from ml.synthetic.generator import SyntheticDataset

EARLY_WINDOW = timedelta(days=14)
LABEL_WINDOW = timedelta(days=30)


def _json_default(value: object) -> object:
    if isinstance(value, UUID | Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"not serializable: {type(value).__name__}")


def dataset_hash(dataset: SyntheticDataset) -> str:
    """SHA-256 over every row of every table in a canonical JSON form."""
    digest = hashlib.sha256()
    for table, rows in dataset.tables():
        digest.update(f"#{table}\n".encode())
        for row in rows:
            digest.update(json.dumps(row, sort_keys=True, default=_json_default).encode())
            digest.update(b"\n")
    return digest.hexdigest()


def _round(value: float, sample: list) -> float | None:
    return round(float(value), 1) if sample else None


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _share(counter: Counter, total: int) -> dict[str, float]:
    return {str(getattr(k, "value", k)): round(v / total, 4) for k, v in sorted(counter.items())}


def _quartile_rates(values: list[float], outcomes: list[bool]) -> list[float | None]:
    if not values:
        return []
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    buckets = np.searchsorted(edges, values, side="right")
    return [_rate([o for o, b in zip(outcomes, buckets, strict=True) if b == q]) for q in range(4)]


def _auc(scores: list[float], labels: list[bool]) -> float | None:
    """Rank-based ROC AUC (ties averaged)."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    sorted_scores = np.asarray(scores)[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positive_ranks = ranks[np.asarray(labels, dtype=bool)].sum()
    auc = (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)
    return round(float(auc), 4)


def early_snapshot(dataset: SyntheticDataset) -> tuple[list[int], list[float], list[bool]]:
    """Point-in-time check at T = created_at + 14d, for leads still active at T whose label
    window (T, T + 30d] ends before as_of.

    Returns (activities before T, engagement score from activities before T, converted in
    the label window). Uses only data available at T as inputs.
    """
    as_of = dataset.config.as_of
    by_lead: dict[UUID, list[dict]] = defaultdict(list)
    for activity in dataset.activities:
        if activity["lead_id"] is not None:
            by_lead[activity["lead_id"]].append(activity)

    counts, scores, labels = [], [], []
    for lead in dataset.leads:
        t = lead["created_at"] + EARLY_WINDOW
        if t + LABEL_WINDOW > as_of:
            continue
        terminal = lead_terminal_time(lead["status"], lead["converted_at"], lead["updated_at"])
        if terminal is not None and terminal <= t:
            continue  # no longer an active lead at T
        history = [a for a in by_lead[lead["id"]] if a["occurred_at"] <= t]
        score = sum(
            1.0
            + (a["activity_type"] in (ActivityType.MEETING, ActivityType.DEMO))
            + (a["direction"] == ActivityDirection.INBOUND)
            for a in history
        )
        converted = lead["converted_at"]
        counts.append(len(history))
        scores.append(score)
        labels.append(converted is not None and t < converted <= t + LABEL_WINDOW)
    return counts, scores, labels


def sanity_counters(dataset: SyntheticDataset) -> dict[str, int]:
    """Violations of point-in-time and relational invariants. All must be 0."""
    as_of = dataset.config.as_of
    leads = {lead["id"]: lead for lead in dataset.leads}
    accounts = {account["id"]: account for account in dataset.accounts}
    contacts = {contact["id"]: contact for contact in dataset.contacts}
    counters: Counter[str] = Counter()

    for _table, rows in dataset.tables():
        for row in rows:
            for column in ("created_at", "updated_at", "occurred_at", "converted_at"):
                value = row.get(column)
                if value is not None and value > as_of:
                    counters["timestamps_after_as_of"] += 1

    last_activity: dict[UUID, datetime] = {}
    for activity in dataset.activities:
        occurred = activity["occurred_at"]
        if not any(activity[k] for k in ("account_id", "lead_id", "opportunity_id", "contact_id")):
            counters["activities_without_parent"] += 1
        if activity["lead_id"] is not None:
            lead = leads[activity["lead_id"]]
            if occurred < lead["created_at"]:
                counters["activities_before_parent_created"] += 1
            terminal = lead_terminal_time(lead["status"], lead["converted_at"], lead["updated_at"])
            if terminal is not None and occurred >= terminal:
                key = (
                    "lead_activities_after_converted_at"
                    if lead["status"] == LeadStatus.CONVERTED
                    else "lead_activities_after_disqualification"
                )
                counters[key] += 1
            previous = last_activity.get(lead["id"])
            last_activity[lead["id"]] = occurred if previous is None else max(previous, occurred)
        if activity["account_id"] is not None:
            account = accounts[activity["account_id"]]
            contact = contacts.get(activity["contact_id"])
            if contact is not None and contact["account_id"] != account["id"]:
                counters["account_activity_cross_account"] += 1
            parent_created = contact["created_at"] if contact else account["created_at"]
            if occurred < parent_created:
                counters["activities_before_parent_created"] += 1

    for lead in dataset.leads:
        converted = lead["converted_at"]
        if (lead["status"] == LeadStatus.CONVERTED) != (converted is not None):
            counters["converted_status_mismatch"] += 1
        if converted is not None and converted <= lead["created_at"]:
            counters["converted_at_not_after_created_at"] += 1
        if lead["updated_at"] < lead["created_at"]:
            counters["updated_at_before_created_at"] += 1
        if lead["status"] == LeadStatus.DISQUALIFIED:
            last = last_activity.get(lead["id"])
            if last is not None and lead["updated_at"] < last:
                counters["disqualified_updated_at_before_last_activity"] += 1

    for contact in dataset.contacts:
        if contact["created_at"] < accounts[contact["account_id"]]["created_at"]:
            counters["contacts_before_account_created"] += 1

    names = (
        "timestamps_after_as_of",
        "activities_without_parent",
        "activities_before_parent_created",
        "lead_activities_after_converted_at",
        "lead_activities_after_disqualification",
        "disqualified_updated_at_before_last_activity",
        "converted_status_mismatch",
        "converted_at_not_after_created_at",
        "updated_at_before_created_at",
        "account_activity_cross_account",
        "contacts_before_account_created",
    )
    return {name: counters[name] for name in names}


def summary_stats(dataset: SyntheticDataset) -> dict[str, Any]:
    as_of = dataset.config.as_of
    leads = dataset.leads
    n_leads = len(leads)
    converted = [lead["converted_at"] is not None for lead in leads]
    lead_activity_counts = Counter(a["lead_id"] for a in dataset.activities if a["lead_id"])
    per_lead = [lead_activity_counts.get(lead["id"], 0) for lead in leads]
    contacts_per_account = Counter(c["account_id"] for c in dataset.contacts)

    by_source: dict[str, list[bool]] = defaultdict(list)
    for lead, outcome in zip(leads, converted, strict=True):
        by_source[lead["lead_source"].value].append(outcome)
    days_to_convert = [
        (lead["converted_at"] - lead["created_at"]).total_seconds() / 86400
        for lead in leads
        if lead["converted_at"] is not None
    ]
    mature = [lead["created_at"] + LABEL_WINDOW <= as_of for lead in leads]
    within_30 = [
        lead["converted_at"] is not None
        and lead["converted_at"] <= lead["created_at"] + LABEL_WINDOW
        for lead in leads
    ]
    counts, scores, labels = early_snapshot(dataset)
    buckets = {"0": [], "1-2": [], "3-5": [], "6+": []}
    for count, label in zip(counts, labels, strict=True):
        key = "0" if count == 0 else "1-2" if count <= 2 else "3-5" if count <= 5 else "6+"
        buckets[key].append(label)

    all_times = [
        row[col]
        for _, rows in dataset.tables()
        for row in rows
        for col in ("created_at", "occurred_at")
        if row.get(col) is not None
    ]
    employee_counts = [a["employee_count"] for a in dataset.accounts]
    return {
        "row_counts": {table: len(rows) for table, rows in dataset.tables()},
        "time_range": {
            "min": min(all_times).isoformat() if all_times else None,
            "max": max(all_times).isoformat() if all_times else None,
            "as_of": as_of.isoformat(),
        },
        "accounts": {
            "employee_count_median": _round(np.median(employee_counts), employee_counts),
            "employee_count_p90": _round(np.percentile(employee_counts, 90), employee_counts),
            "contacts_per_account": {
                "min": min(contacts_per_account.values(), default=0),
                "mean": round(len(dataset.contacts) / max(len(dataset.accounts), 1), 2),
                "max": max(contacts_per_account.values(), default=0),
            },
        },
        "leads": {
            "source_mix": _share(Counter(lead["lead_source"] for lead in leads), max(n_leads, 1)),
            "status_mix": _share(Counter(lead["status"] for lead in leads), max(n_leads, 1)),
            "conversion_rate": _rate(converted),
            "conversion_rate_by_source": {k: _rate(v) for k, v in sorted(by_source.items())},
            "conversion_rate_by_company_size_quartile": _quartile_rates(
                [lead["company_size"] for lead in leads], converted
            ),
            "conversion_rate_by_estimated_value_quartile": _quartile_rates(
                [float(lead["estimated_value"]) for lead in leads], converted
            ),
            "median_days_to_conversion": round(float(np.median(days_to_convert)), 1)
            if days_to_convert
            else None,
            "converted_within_30d_of_creation": _rate(
                [w for w, m in zip(within_30, mature, strict=True) if m]
            ),
            "open_at_as_of": sum(
                lead["status"] not in (LeadStatus.CONVERTED, LeadStatus.DISQUALIFIED)
                for lead in leads
            ),
        },
        "activities": {
            "lead_activities": dataset.lead_activity_count,
            "account_activities": len(dataset.activities) - dataset.lead_activity_count,
            "per_lead": {
                "mean": round(float(np.mean(per_lead)), 2) if per_lead else None,
                "median": float(np.median(per_lead)) if per_lead else None,
                "p95": _round(np.percentile(per_lead, 95), per_lead),
                "zero_share": _rate([v == 0 for v in per_lead]),
            },
            "type_mix": _share(Counter(a["activity_type"] for a in dataset.activities),
                               max(len(dataset.activities), 1)),
            "direction_mix": _share(Counter(a["direction"] for a in dataset.activities),
                                    max(len(dataset.activities), 1)),
        },
        "point_in_time_check": {
            "snapshot": "T = created_at + 14d; active at T; label = converted in (T, T + 30d]",
            "snapshots": len(labels),
            "label_rate": _rate(labels),
            "label_rate_by_activities_before_t": {k: _rate(v) for k, v in buckets.items()},
            "engagement_score_auc": _auc(scores, labels),
        },
    }


def build_manifest(
    dataset: SyntheticDataset, stats: dict[str, Any], sanity: dict[str, int]
) -> dict[str, Any]:
    config = asdict(dataset.config)
    config["as_of"] = dataset.config.as_of.isoformat()
    return {
        "generator": "scripts/generate_crm_data.py",
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),  # informational; not part of the hash
        "config": config,
        "dataset_sha256": dataset_hash(dataset),
        "row_counts": stats["row_counts"],
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faker": faker.VERSION,
        },
        "conventions": {
            "terminal_time": TERMINAL_TIME_CONVENTION,
            "terminal_time_function": "ml.synthetic.conventions.lead_terminal_time",
            "observation_end": OBSERVATION_END_CONVENTION,
            "observation_end_function": "ml.synthetic.conventions.lead_observation_end",
            "not_generated": "notes, opportunities",
        },
        "sanity": sanity,
        "stats": stats,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n")


def format_summary(stats: dict[str, Any], sanity: dict[str, int]) -> str:
    leads, acts, pit = stats["leads"], stats["activities"], stats["point_in_time_check"]
    lines = [
        "Rows: " + ", ".join(f"{k}={v:,}" for k, v in stats["row_counts"].items()),
        f"Time range: {stats['time_range']['min']} .. {stats['time_range']['max']} "
        f"(as_of {stats['time_range']['as_of']})",
        f"Accounts: employee_count median {stats['accounts']['employee_count_median']}, "
        f"p90 {stats['accounts']['employee_count_p90']}; contacts/account "
        f"{stats['accounts']['contacts_per_account']}",
        f"Leads: conversion {leads['conversion_rate']}, within 30d of creation "
        f"{leads['converted_within_30d_of_creation']}, median days to conversion "
        f"{leads['median_days_to_conversion']}, open at as_of {leads['open_at_as_of']:,}",
        f"  status mix: {leads['status_mix']}",
        f"  source mix: {leads['source_mix']}",
        f"  conversion by source: {leads['conversion_rate_by_source']}",
        f"  conversion by company_size quartile: "
        f"{leads['conversion_rate_by_company_size_quartile']}",
        f"  conversion by estimated_value quartile: "
        f"{leads['conversion_rate_by_estimated_value_quartile']}",
        f"Activities: lead {acts['lead_activities']:,}, account {acts['account_activities']:,}; "
        f"per lead {acts['per_lead']}",
        f"  type mix: {acts['type_mix']}",
        f"  direction mix: {acts['direction_mix']}",
        f"Point-in-time check ({pit['snapshot']}): snapshots {pit['snapshots']:,}, label rate "
        f"{pit['label_rate']}, AUC of engagement score {pit['engagement_score_auc']}",
        f"  label rate by activities before T: {pit['label_rate_by_activities_before_t']}",
        "Sanity counters (all must be 0): "
        + ", ".join(f"{k}={v}" for k, v in sanity.items()),
    ]
    return "\n".join(lines)
