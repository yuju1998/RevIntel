"""Determinism, counts, relationships, constraints and timeline invariants (in memory)."""

import dataclasses
import uuid
from collections import Counter, defaultdict
from decimal import Decimal

import pytest
from pwdlib.hashers.argon2 import Argon2Hasher
from sqlalchemy import String

import app.models  # noqa: F401
from app.db.base import Base
from app.models.enums import ActivityDirection, ActivityType, LeadSource, LeadStatus, UserRole
from ml.synthetic.config import GenerationConfig
from ml.synthetic.conventions import lead_terminal_time
from ml.synthetic.generator import ActivityBudgetExceeded, SyntheticDataset, generate
from ml.synthetic.summary import dataset_hash, sanity_counters
from ml.tests.conftest import SMALL

# --- determinism ----------------------------------------------------------------------------


def test_same_seed_reproduces_identical_rows(small_dataset: SyntheticDataset) -> None:
    again = generate(SMALL)

    assert dataset_hash(again) == dataset_hash(small_dataset)
    assert again.leads == small_dataset.leads
    assert again.activities == small_dataset.activities


def test_different_seed_changes_the_data(small_dataset: SyntheticDataset) -> None:
    other = generate(dataclasses.replace(SMALL, seed=43))

    assert dataset_hash(other) != dataset_hash(small_dataset)


# --- counts and relationships ---------------------------------------------------------------


def test_exact_counts(small_dataset: SyntheticDataset) -> None:
    assert len(small_dataset.users) == SMALL.reps + SMALL.admins
    assert len(small_dataset.accounts) == SMALL.accounts
    assert len(small_dataset.contacts) == SMALL.contacts
    assert len(small_dataset.leads) == SMALL.leads
    assert len(small_dataset.activities) == SMALL.activities
    lead_activities = sum(a["lead_id"] is not None for a in small_dataset.activities)
    assert lead_activities == small_dataset.lead_activity_count


def test_ids_are_unique_version_4_uuids(small_dataset: SyntheticDataset) -> None:
    ids = [row["id"] for _, rows in small_dataset.tables() for row in rows]

    assert len(ids) == len(set(ids))
    assert all(isinstance(i, uuid.UUID) and i.version == 4 for i in ids)


def test_foreign_keys_resolve_and_accounts_are_consistent(small_dataset: SyntheticDataset) -> None:
    users = {u["id"] for u in small_dataset.users}
    accounts = {a["id"] for a in small_dataset.accounts}
    leads = {lead["id"] for lead in small_dataset.leads}
    contact_account = {c["id"]: c["account_id"] for c in small_dataset.contacts}

    assert all(a["owner_id"] in users for a in small_dataset.accounts)
    assert all(lead["owner_id"] in users | {None} for lead in small_dataset.leads)
    assert set(contact_account.values()) == accounts  # every account has a contact
    for activity in small_dataset.activities:
        assert activity["created_by_id"] in users
        assert activity["opportunity_id"] is None
        if activity["lead_id"] is not None:
            assert activity["lead_id"] in leads
            assert activity["account_id"] is None and activity["contact_id"] is None
        else:
            assert activity["account_id"] in accounts
            assert contact_account[activity["contact_id"]] == activity["account_id"]


# --- constraints ----------------------------------------------------------------------------


def test_values_fit_the_schema(small_dataset: SyntheticDataset) -> None:
    for table_name, rows in small_dataset.tables():
        columns = Base.metadata.tables[table_name].columns
        for row in rows:
            assert set(row) == {column.name for column in columns}
            for column in columns:
                value = row[column.name]
                if value is None:
                    assert column.nullable, (table_name, column.name)
                    continue
                if isinstance(column.type, String) and column.type.length:
                    assert len(value) <= column.type.length, (table_name, column.name)
                if column.name.endswith(("_at",)):
                    assert value.utcoffset() is not None


def test_enums_money_and_counts_are_valid(small_dataset: SyntheticDataset) -> None:
    assert {u["role"] for u in small_dataset.users} <= set(UserRole)
    for account in small_dataset.accounts:
        assert account["employee_count"] >= 0
        assert account["annual_revenue"] >= 0
        assert account["annual_revenue"].as_tuple().exponent == -2
        assert account["annual_revenue"] <= Decimal("99999999999999.99")
    for lead in small_dataset.leads:
        assert lead["lead_source"] in set(LeadSource)
        assert lead["status"] in set(LeadStatus)
        assert lead["company_size"] >= 0
        assert lead["estimated_value"] >= 0
        assert lead["estimated_value"].as_tuple().exponent == -2
        assert lead["estimated_value"] <= Decimal("999999999999.99")
        # ck_leads_converted_requires_converted_at (and the generator never sets it otherwise)
        assert (lead["status"] == LeadStatus.CONVERTED) == (lead["converted_at"] is not None)
    for activity in small_dataset.activities:
        assert activity["activity_type"] in set(ActivityType)
        assert activity["direction"] in set(ActivityDirection)


def test_synthetic_users_have_valid_unusable_argon2_hashes(small_dataset: SyntheticDataset) -> None:
    hashes = {u["password_hash"] for u in small_dataset.users}

    assert len(hashes) == 1  # one hash per run, shared
    password_hash = hashes.pop()
    assert password_hash.startswith("$argon2id$")
    assert Argon2Hasher().identify(password_hash)
    assert not Argon2Hasher().verify("password", password_hash)
    assert all(u["email"].endswith("@synthetic.example.com") for u in small_dataset.users)
    assert len({u["email"] for u in small_dataset.users}) == len(small_dataset.users)


def test_generated_emails_use_reserved_example_domains(small_dataset: SyntheticDataset) -> None:
    emails = [c["email"] for c in small_dataset.contacts if c["email"]]
    emails += [lead["email"] for lead in small_dataset.leads if lead["email"]]

    assert emails
    assert all(e.rsplit("@", 1)[1].split(".")[-2] == "example" for e in emails)


# --- timeline / point-in-time invariants ----------------------------------------------------


def test_sanity_counters_are_all_zero(small_dataset: SyntheticDataset) -> None:
    sanity = sanity_counters(small_dataset)

    assert sanity and all(value == 0 for value in sanity.values()), sanity


def test_lead_activities_stop_at_the_terminal_transition(small_dataset: SyntheticDataset) -> None:
    activities = defaultdict(list)
    for activity in small_dataset.activities:
        if activity["lead_id"]:
            activities[activity["lead_id"]].append(activity["occurred_at"])
    statuses = Counter(lead["status"] for lead in small_dataset.leads)
    assert statuses[LeadStatus.CONVERTED] and statuses[LeadStatus.DISQUALIFIED]

    for lead in small_dataset.leads:
        times = activities[lead["id"]]
        assert all(t >= lead["created_at"] for t in times)
        terminal = lead_terminal_time(lead["status"], lead["converted_at"], lead["updated_at"])
        if lead["status"] == LeadStatus.CONVERTED:
            assert lead["updated_at"] == lead["converted_at"] > lead["created_at"]
        if lead["status"] == LeadStatus.DISQUALIFIED:
            # Disqualification timestamp: after every lead activity, nothing after it.
            assert all(t < lead["updated_at"] for t in times)
        if terminal is not None:
            assert all(t < terminal for t in times)
        else:
            assert lead["updated_at"] >= lead["created_at"]
        if lead["status"] == LeadStatus.NEW:
            assert lead["updated_at"] == lead["created_at"] and not times


def test_nothing_happens_after_as_of(small_dataset: SyntheticDataset) -> None:
    as_of = small_dataset.config.as_of
    for _, rows in small_dataset.tables():
        for row in rows:
            for column in ("created_at", "updated_at", "occurred_at", "converted_at"):
                assert row.get(column) is None or row[column] <= as_of


def test_activity_text_never_mentions_the_outcome(small_dataset: SyntheticDataset) -> None:
    text = " ".join(
        f"{a['subject']} {a['body'] or ''}".lower() for a in small_dataset.activities
    )

    for word in ("convert", "won", "closed", "disqualif", "lost", "signed"):
        assert word not in text


# --- budget and configuration ---------------------------------------------------------------


def test_activity_budget_overflow_fails_instead_of_recalibrating() -> None:
    config = dataclasses.replace(SMALL, activities=10)

    with pytest.raises(ActivityBudgetExceeded, match="not recalibrated"):
        generate(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"accounts": -1},
        {"contacts": 10, "accounts": 20},
        {"accounts": 0, "contacts": 5},
        {"reps": 0, "admins": 0},
        {"max_lead_age_days": 0},
    ],
)
def test_invalid_configuration_is_rejected(overrides: dict) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(SMALL, **overrides).validate()


def test_naive_as_of_is_rejected() -> None:
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone"):
        dataclasses.replace(SMALL, as_of=datetime(2026, 1, 1)).validate()


def test_zero_counts_produce_an_empty_but_valid_dataset() -> None:
    dataset = generate(GenerationConfig(accounts=0, contacts=0, leads=0, activities=0))

    assert dataset.accounts == dataset.leads == dataset.activities == []
    assert all(value == 0 for value in sanity_counters(dataset).values())


def test_terminal_time_convention() -> None:
    from datetime import UTC, datetime

    converted_at = datetime(2026, 5, 1, tzinfo=UTC)
    updated_at = datetime(2026, 6, 1, tzinfo=UTC)

    assert lead_terminal_time(LeadStatus.CONVERTED, converted_at, updated_at) == converted_at
    assert lead_terminal_time(LeadStatus.DISQUALIFIED, None, updated_at) == updated_at
    for status in (LeadStatus.NEW, LeadStatus.WORKING, LeadStatus.QUALIFIED):
        assert lead_terminal_time(status, None, updated_at) is None
