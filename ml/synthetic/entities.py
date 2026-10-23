"""Users, accounts, contacts and lead attributes.

Timestamps are handled as integer seconds relative to `as_of` (negative = in the past)
and converted to aware UTC datetimes only when rows are built.
"""

import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
from pwdlib.hashers.argon2 import Argon2Hasher

from app.models.enums import LeadSource, LeadStatus, UserRole
from ml.synthetic import config as c
from ml.synthetic.ids import Streams, deterministic_uuids

DAY = 86_400
CENTS = Decimal("0.01")
MAX_ACCOUNT_REVENUE = Decimal("99999999999999.99")  # NUMERIC(16,2)
MAX_ESTIMATED_VALUE = Decimal("999999999999.99")  # NUMERIC(14,2)


def at(as_of: datetime, offset_seconds: int) -> datetime:
    return as_of + timedelta(seconds=int(offset_seconds))


def money(value: float, cap: Decimal) -> Decimal:
    return min(Decimal(str(round(float(value), 2))).quantize(CENTS), cap)


def _choice(rng: np.random.Generator, options: tuple[tuple, ...], size: int) -> np.ndarray:
    values = np.array([value for value, _ in options], dtype=object)
    weights = np.array([weight for _, weight in options], dtype=float)
    return values[rng.choice(len(values), size=size, p=weights / weights.sum())]


def _slug(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:40] or "company"


def _email_part(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _slug(text)) or "contact"


# --- users ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Users:
    rows: list[dict]
    owner_ids: list[uuid.UUID]  # sales reps who own records


def synthetic_password_hash(rng: np.random.Generator) -> str:
    """Valid Argon2id hash of a random throwaway secret that is never stored.

    Deterministic for a seed (fixed salt), so it is not a security boundary: synthetic
    users are not meant to log in.
    """
    secret = rng.bytes(32).hex()
    return Argon2Hasher().hash(secret, salt=rng.bytes(16))


def build_users(cfg: c.GenerationConfig, streams: Streams) -> Users:
    rng = streams.rng("users")
    fake = streams.faker("users")
    count = cfg.reps + cfg.admins
    ids = deterministic_uuids(streams.rng("ids"), count)
    password_hash = synthetic_password_hash(rng)
    created_at = at(cfg.as_of, -(cfg.account_history_days + 30) * DAY)
    rows = []
    for index, user_id in enumerate(ids):
        is_rep = index < cfg.reps
        number = index + 1 if is_rep else index - cfg.reps + 1
        prefix = "rep" if is_rep else "admin"
        rows.append(
            {
                "id": user_id,
                "email": f"{prefix}-{number:02d}@synthetic.example.com",
                "password_hash": password_hash,
                "full_name": fake.name(),
                "role": UserRole.SALES_REP if is_rep else UserRole.ADMIN,
                "is_active": True,
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
    owner_ids = ids[: cfg.reps] if cfg.reps else ids
    return Users(rows=rows, owner_ids=owner_ids)


# --- accounts and contacts ------------------------------------------------------------------


@dataclass(frozen=True)
class Accounts:
    rows: list[dict]
    created_offsets: np.ndarray
    employee_counts: np.ndarray
    slugs: list[str]


def build_accounts(cfg: c.GenerationConfig, streams: Streams, owner_ids: list) -> Accounts:
    rng = streams.rng("accounts")
    fake = streams.faker("accounts")
    n = cfg.accounts
    ids = deterministic_uuids(streams.rng("ids"), n)
    industries = _choice(rng, c.INDUSTRIES, n)
    low, high = c.EMPLOYEE_COUNT_RANGE
    employees = np.clip(
        np.rint(rng.lognormal(np.log(c.EMPLOYEE_COUNT_MEDIAN), c.EMPLOYEE_COUNT_SIGMA, n)),
        low,
        high,
    ).astype(int)
    revenue_per_employee = rng.lognormal(
        np.log(c.REVENUE_PER_EMPLOYEE_MEDIAN), c.REVENUE_PER_EMPLOYEE_SIGMA, n
    )
    created_offsets = -rng.integers(DAY, cfg.account_history_days * DAY, size=n)
    owners = rng.integers(0, len(owner_ids), size=n) if n else np.array([], dtype=int)

    rows, slugs = [], []
    for i in range(n):
        name = fake.company()
        slug = f"{_slug(name)}-{i + 1}"
        slugs.append(slug)
        created_at = at(cfg.as_of, created_offsets[i])
        rows.append(
            {
                "id": ids[i],
                "name": name,
                "industry": str(industries[i]),
                "employee_count": int(employees[i]),
                "annual_revenue": money(
                    employees[i] * revenue_per_employee[i], MAX_ACCOUNT_REVENUE
                ),
                "website": f"https://{slug}.example.com",
                "owner_id": owner_ids[owners[i]],
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
    return Accounts(rows, created_offsets, employees, slugs)


@dataclass(frozen=True)
class Contacts:
    rows: list[dict]
    account_index: np.ndarray
    created_offsets: np.ndarray


def build_contacts(cfg: c.GenerationConfig, streams: Streams, accounts: Accounts) -> Contacts:
    rng = streams.rng("contacts")
    fake = streams.faker("contacts")
    n_accounts, n = cfg.accounts, cfg.contacts
    ids = deterministic_uuids(streams.rng("ids"), n)
    if n == 0:
        return Contacts([], np.array([], dtype=int), np.array([], dtype=int))

    # Every account gets one contact; larger companies get more of the rest.
    weights = np.log1p(accounts.employee_counts.astype(float))
    extra = rng.multinomial(n - n_accounts, weights / weights.sum())
    account_index = np.repeat(np.arange(n_accounts), 1 + extra)
    account_created = accounts.created_offsets[account_index]
    # Created after the account (and before as_of).
    created_offsets = account_created + np.floor(
        rng.random(n) * (-account_created)
    ).astype(np.int64)
    titles = _choice(rng, c.JOB_TITLES, n)
    has_email = rng.random(n) < c.CONTACT_EMAIL_SHARE
    has_phone = rng.random(n) < c.CONTACT_PHONE_SHARE

    rows = []
    for i in range(n):
        first, last = fake.first_name(), fake.last_name()
        phone = fake.phone_number() if has_phone[i] else None
        created_at = at(cfg.as_of, created_offsets[i])
        rows.append(
            {
                "id": ids[i],
                "account_id": accounts.rows[account_index[i]]["id"],
                "first_name": first,
                "last_name": last,
                "email": (
                    f"{_email_part(first)}.{_email_part(last)}"
                    f"@{accounts.slugs[account_index[i]]}.example.com"
                )
                if has_email[i]
                else None,
                "phone": phone[:50] if phone else None,
                "job_title": str(titles[i]),
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
    return Contacts(rows, account_index, created_offsets)


# --- leads ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeadDraft:
    """Lead attributes before simulation (outcome fields are filled in afterwards)."""

    ids: list[uuid.UUID]
    sources: np.ndarray  # LeadSource
    created_offsets: np.ndarray  # seconds relative to as_of (negative)
    owner_ids: list[uuid.UUID | None]
    latent_quality: np.ndarray  # z: never stored
    base_rows: list[dict]


def _created_offsets_with_growth(
    rng: np.random.Generator, n: int, history_days: int
) -> np.ndarray:
    # Density 1 + g*x on x in [0, 1] (x = 1 at as_of); inverse-CDF sampling.
    g = c.LEAD_VOLUME_GROWTH
    x = (-1 + np.sqrt(1 + g * (2 + g) * rng.random(n))) / g
    seconds_before = (1 - x) * history_days * DAY
    return -np.maximum(np.floor(seconds_before), 60).astype(np.int64)


def build_lead_drafts(cfg: c.GenerationConfig, streams: Streams, owner_ids: list) -> LeadDraft:
    rng = streams.rng("leads")
    fake = streams.faker("leads")
    n = cfg.leads
    ids = deterministic_uuids(streams.rng("ids"), n)
    sources = _choice(rng, c.LEAD_SOURCE_WEIGHTS, n)
    industries = _choice(rng, c.INDUSTRIES, n)

    size_score = rng.standard_normal(n)
    company_size = np.maximum(
        np.rint(np.exp(np.log(c.COMPANY_SIZE_MEDIAN) + c.COMPANY_SIZE_SIGMA * size_score)), 1
    ).astype(int)
    rho = c.SIZE_VALUE_CORRELATION
    value_score = rho * size_score + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    estimated_value = np.exp(
        np.log(c.ESTIMATED_VALUE_MEDIAN) + c.ESTIMATED_VALUE_SIGMA * value_score
    )

    created_offsets = _created_offsets_with_growth(rng, n, cfg.lead_history_days)
    has_owner = rng.random(n) < c.LEAD_OWNER_SHARE
    owner_pick = rng.integers(0, max(len(owner_ids), 1), size=n)
    has_email = rng.random(n) < c.LEAD_EMAIL_SHARE

    # Latent quality from observable attributes (standardized with the known
    # distribution parameters) plus unobservable noise.
    source_effect = np.array([c.SOURCE_EFFECT[LeadSource(s)] for s in sources])
    latent = (
        source_effect
        + c.SIZE_EFFECT * size_score
        + c.VALUE_EFFECT * value_score
        + rng.normal(0, c.LATENT_NOISE_SIGMA, n)
    )

    owners: list[uuid.UUID | None] = [
        owner_ids[owner_pick[i]] if has_owner[i] else None for i in range(n)
    ]
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": ids[i],
                "company_name": fake.company(),
                "contact_name": fake.name(),
                "email": fake.safe_email() if has_email[i] else None,
                "lead_source": LeadSource(sources[i]),
                "industry": str(industries[i]),
                "company_size": int(company_size[i]),
                "estimated_value": money(estimated_value[i], MAX_ESTIMATED_VALUE),
                "owner_id": owners[i],
                "created_at": at(cfg.as_of, created_offsets[i]),
            }
        )
    return LeadDraft(ids, sources, created_offsets, owners, latent, rows)


def finish_leads(
    cfg: c.GenerationConfig,
    draft: LeadDraft,
    statuses: list[LeadStatus],
    converted_offsets: np.ndarray,
    updated_offsets: np.ndarray,
) -> list[dict]:
    """Add simulated outcome fields. Offsets are seconds after each lead's created_at."""
    rows = []
    for i, base in enumerate(draft.base_rows):
        created = base["created_at"]
        rows.append(
            {
                **base,
                "status": statuses[i],
                "converted_at": created + timedelta(seconds=int(converted_offsets[i]))
                if statuses[i] == LeadStatus.CONVERTED
                else None,
                # Most recent simulated state change (see ml.synthetic.conventions).
                "updated_at": created + timedelta(seconds=int(updated_offsets[i])),
            }
        )
    return rows
