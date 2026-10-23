"""Builds the complete synthetic dataset in memory (no database access)."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from ml.synthetic import config as c
from ml.synthetic.account_activities import draft_account_activities
from ml.synthetic.entities import (
    at,
    build_accounts,
    build_contacts,
    build_lead_drafts,
    build_users,
    finish_leads,
)
from ml.synthetic.ids import Streams, deterministic_uuids
from ml.synthetic.simulation import activity_direction, activity_type, simulate_leads
from ml.synthetic.text import subject_and_body

Progress = Callable[[str], None]


class ActivityBudgetExceeded(ValueError):
    """The lead simulation produced more activities than --activities allows."""


@dataclass
class SyntheticDataset:
    config: c.GenerationConfig
    users: list[dict]
    accounts: list[dict]
    contacts: list[dict]
    leads: list[dict]
    activities: list[dict]
    lead_activity_count: int
    timings: dict[str, float] = field(default_factory=dict)

    def tables(self) -> list[tuple[str, list[dict]]]:
        """Tables in foreign-key insert order."""
        return [
            ("users", self.users),
            ("accounts", self.accounts),
            ("contacts", self.contacts),
            ("leads", self.leads),
            ("activities", self.activities),
        ]


def _delay_seconds(rng: np.random.Generator, count: int) -> np.ndarray:
    return (rng.random(count) * c.MAX_LOGGING_DELAY_HOURS * 3600).astype(np.int64)


def generate(cfg: c.GenerationConfig, progress: Progress = lambda _: None) -> SyntheticDataset:
    cfg.validate()
    streams = Streams(cfg.seed)
    timings: dict[str, float] = {}

    def stage(name: str, started: float, rows: int) -> None:
        elapsed = time.perf_counter() - started
        timings[name] = elapsed
        rate = f" ({rows / elapsed:,.0f} rows/s)" if elapsed > 0 and rows else ""
        progress(f"  {name:<24} {rows:>8,} rows in {elapsed:5.1f}s{rate}")

    started = time.perf_counter()
    users = build_users(cfg, streams)
    stage("users", started, len(users.rows))

    started = time.perf_counter()
    accounts = build_accounts(cfg, streams, users.owner_ids)
    stage("accounts", started, len(accounts.rows))

    started = time.perf_counter()
    contacts = build_contacts(cfg, streams, accounts)
    stage("contacts", started, len(contacts.rows))

    started = time.perf_counter()
    draft = build_lead_drafts(cfg, streams, users.owner_ids)
    simulation = simulate_leads(
        streams.rng("simulation"),
        draft.latent_quality,
        draft.sources,
        draft.created_offsets,
        cfg.max_lead_age_days,
    )
    leads = finish_leads(
        cfg, draft, simulation.statuses, simulation.converted_offsets, simulation.updated_offsets
    )
    stage("leads (simulated)", started, len(leads))

    lead_activity_count = int(simulation.event_lead.size)
    if lead_activity_count > cfg.activities:
        raise ActivityBudgetExceeded(
            f"The lead simulation produced {lead_activity_count:,} activities, more than "
            f"--activities {cfg.activities:,}. Increase --activities or reduce --leads; the "
            "simulation is not recalibrated to fit a budget."
        )

    started = time.perf_counter()
    text_rng = streams.rng("text")
    fill_rng = streams.rng("account_activities")
    owner_ids = users.owner_ids
    activities: list[dict] = []

    # Lead activities: lead only (leads have no account), never after a terminal transition.
    delays = _delay_seconds(fill_rng, lead_activity_count)
    with_body = text_rng.random(lead_activity_count) < c.ACTIVITY_BODY_SHARE
    fallback_owner = fill_rng.integers(0, max(len(owner_ids), 1), size=lead_activity_count)
    for k in range(lead_activity_count):
        lead_index = int(simulation.event_lead[k])
        lead = leads[lead_index]
        occurred = lead["created_at"] + timedelta(seconds=int(simulation.event_offset[k]))
        kind = activity_type(int(simulation.event_type[k]))
        direction = activity_direction(int(simulation.event_direction[k]))
        subject, body = subject_and_body(text_rng, kind, direction, bool(with_body[k]))
        activities.append(
            {
                "account_id": None,
                "lead_id": lead["id"],
                "opportunity_id": None,
                "contact_id": None,
                "created_by_id": draft.owner_ids[lead_index] or owner_ids[fallback_owner[k]],
                "activity_type": kind,
                "direction": direction,
                "subject": subject,
                "body": body,
                "occurred_at": occurred,
                "created_at": min(
                    occurred + timedelta(seconds=int(delays[k])), cfg.as_of
                ),
            }
        )

    # Account/contact activities fill the rest of the budget exactly.
    fill = draft_account_activities(
        fill_rng, cfg.activities - lead_activity_count, accounts, contacts
    )
    delays = _delay_seconds(fill_rng, len(fill.types))
    with_body = text_rng.random(len(fill.types)) < c.ACTIVITY_BODY_SHARE
    for k in range(len(fill.types)):
        account = accounts.rows[int(fill.account_index[k])]
        occurred = at(cfg.as_of, int(fill.occurred_offsets[k]))
        subject, body = subject_and_body(
            text_rng, fill.types[k], fill.directions[k], bool(with_body[k])
        )
        activities.append(
            {
                "account_id": account["id"],
                "lead_id": None,
                "opportunity_id": None,
                "contact_id": contacts.rows[int(fill.contact_index[k])]["id"],
                "created_by_id": account["owner_id"],
                "activity_type": fill.types[k],
                "direction": fill.directions[k],
                "subject": subject,
                "body": body,
                "occurred_at": occurred,
                "created_at": min(occurred + timedelta(seconds=int(delays[k])), cfg.as_of),
            }
        )

    activity_ids = deterministic_uuids(streams.rng("ids"), len(activities))
    for activity, activity_id in zip(activities, activity_ids, strict=True):
        activity["id"] = activity_id
    stage("activities", started, len(activities))

    return SyntheticDataset(
        config=cfg,
        users=users.rows,
        accounts=accounts.rows,
        contacts=contacts.rows,
        leads=leads,
        activities=activities,
        lead_activity_count=lead_activity_count,
        timings=timings,
    )

