"""Generation settings and every distribution/coefficient used by the generator.

Kept in one place so the synthetic world is easy to read and explain. Changing any value
changes the dataset (and its manifest hash), so bump GENERATOR_VERSION when doing so.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from app.models.enums import ActivityDirection, ActivityType, LeadSource

GENERATOR_VERSION = "1.0.0"
DEFAULT_AS_OF = datetime(2026, 10, 1, tzinfo=UTC)


@dataclass(frozen=True)
class GenerationConfig:
    seed: int = 42
    accounts: int = 5000
    contacts: int = 20000
    leads: int = 30000
    activities: int = 100000
    reps: int = 25
    admins: int = 2
    as_of: datetime = DEFAULT_AS_OF
    account_history_days: int = 1095
    lead_history_days: int = 540
    max_lead_age_days: int = 180

    def validate(self) -> None:
        for name in ("accounts", "contacts", "leads", "activities", "reps", "admins"):
            if getattr(self, name) < 0:
                raise ValueError(f"--{name} must be >= 0")
        if self.accounts > 0 and self.contacts < self.accounts:
            raise ValueError("--contacts must be >= --accounts (every account gets a contact)")
        if self.accounts == 0 and self.contacts > 0:
            raise ValueError("--contacts requires at least one account")
        if (self.accounts > 0 or self.leads > 0) and self.reps + self.admins == 0:
            raise ValueError("at least one user is required to own records")
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("--as-of must be timezone-aware")
        for name in ("account_history_days", "lead_history_days", "max_lead_age_days"):
            if getattr(self, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be >= 1")


# --- accounts / contacts ------------------------------------------------------------------

INDUSTRIES: tuple[tuple[str, float], ...] = (
    ("Software", 0.18),
    ("Financial Services", 0.10),
    ("Healthcare", 0.10),
    ("Manufacturing", 0.10),
    ("Retail", 0.08),
    ("Education", 0.06),
    ("Logistics", 0.07),
    ("Media", 0.05),
    ("Telecommunications", 0.05),
    ("Energy", 0.05),
    ("Professional Services", 0.10),
    ("Real Estate", 0.06),
)
EMPLOYEE_COUNT_MEDIAN = 120
EMPLOYEE_COUNT_SIGMA = 1.4
EMPLOYEE_COUNT_RANGE = (5, 100_000)
REVENUE_PER_EMPLOYEE_MEDIAN = 180_000
REVENUE_PER_EMPLOYEE_SIGMA = 0.5
CONTACT_EMAIL_SHARE = 0.85
CONTACT_PHONE_SHARE = 0.60
JOB_TITLES: tuple[tuple[str, float], ...] = (
    ("VP of Sales", 0.08),
    ("Head of Operations", 0.08),
    ("IT Manager", 0.10),
    ("Chief Financial Officer", 0.05),
    ("Procurement Manager", 0.10),
    ("Director of Marketing", 0.08),
    ("Software Engineer", 0.12),
    ("Account Manager", 0.10),
    ("Chief Technology Officer", 0.05),
    ("Operations Analyst", 0.12),
    ("Office Manager", 0.12),
)

# --- leads ----------------------------------------------------------------------------------

LEAD_SOURCE_WEIGHTS: tuple[tuple[LeadSource, float], ...] = (
    (LeadSource.WEB, 0.35),
    (LeadSource.OUTBOUND, 0.25),
    (LeadSource.EVENT, 0.12),
    (LeadSource.REFERRAL, 0.10),
    (LeadSource.PARTNER, 0.08),
    (LeadSource.OTHER, 0.10),
)
COMPANY_SIZE_MEDIAN = 80
COMPANY_SIZE_SIGMA = 1.3
ESTIMATED_VALUE_MEDIAN = 25_000
ESTIMATED_VALUE_SIGMA = 0.9
# Correlation between log(company_size) and log(estimated_value).
SIZE_VALUE_CORRELATION = 0.6
LEAD_EMAIL_SHARE = 0.70
LEAD_OWNER_SHARE = 0.85
# Lead volume grows over the history window (density 1 at the start, 1 + g at as_of).
LEAD_VOLUME_GROWTH = 0.5

# --- latent lead quality z (never stored) ---------------------------------------------------

SOURCE_EFFECT: dict[LeadSource, float] = {
    LeadSource.REFERRAL: 0.8,
    LeadSource.PARTNER: 0.6,
    LeadSource.EVENT: 0.3,
    LeadSource.WEB: 0.0,
    LeadSource.OTHER: -0.2,
    LeadSource.OUTBOUND: -0.5,
}
SIZE_EFFECT = 0.35
VALUE_EFFECT = 0.25
# Unobservable noise: keeps conversion from being a deterministic function of the data.
LATENT_NOISE_SIGMA = 0.8

# --- lead engagement / conversion simulation (one step per day of lead age) ---------------


@dataclass(frozen=True)
class SimulationParams:
    # Fixed calibration (seed-independent): ~2.4 lead activities per lead, ~15% conversion
    # overall, ~40% disqualified. Not adjusted per run to fit --activities.
    # Expected activities per day = base_rate * exp(intensity_z * z) * 0.5 ** (age / half_life)
    base_rate: float = 0.05
    intensity_z: float = 0.45
    activity_half_life_days: float = 45.0
    # Engagement E(t): recency-decayed sum of event weights.
    engagement_half_life_days: float = 10.0
    # Daily conversion hazard: sigmoid(c0 + c1*z + c2*log1p(E)).
    conversion_intercept: float = -7.2
    conversion_z: float = 0.9
    conversion_engagement: float = 1.1
    # Daily disqualification hazard: sigmoid(d0 - d1*z - d2*log1p(E) + d3*age_months).
    disqualify_intercept: float = -6.0
    disqualify_z: float = 0.4
    disqualify_engagement: float = 0.5
    disqualify_age: float = 0.2
    # Open leads move NEW -> WORKING on their first activity, WORKING -> QUALIFIED at E >= this.
    qualified_engagement: float = 3.0
    # Activities happen in the first part of a day; status transitions at its end, so a
    # transition is always later than that day's activities.
    activity_day_fraction: float = 0.8


SIMULATION = SimulationParams()

ACTIVITY_TYPES: tuple[ActivityType, ...] = (
    ActivityType.CALL,
    ActivityType.EMAIL,
    ActivityType.MEETING,
    ActivityType.DEMO,
    ActivityType.FOLLOW_UP,
)
# Type logits = base + slope * z: higher-quality leads get more meetings and demos.
ACTIVITY_TYPE_BASE_LOGITS = (0.6, 1.0, -1.0, -2.0, 0.0)
ACTIVITY_TYPE_Z_SLOPES = (0.0, 0.0, 0.6, 0.8, 0.1)
ENGAGEMENT_WEIGHTS: dict[ActivityType, float] = {
    ActivityType.CALL: 0.3,
    ActivityType.EMAIL: 0.1,
    ActivityType.MEETING: 2.0,
    ActivityType.DEMO: 3.0,
    ActivityType.FOLLOW_UP: 0.2,
}
INBOUND_ENGAGEMENT_BONUS = 1.0
INTERNAL_ENGAGEMENT_WEIGHT = 0.05
INTERNAL_SHARE = 0.05
# P(inbound) = sigmoid(base + slope * z + source bonus)
INBOUND_BASE_LOGIT = -1.2
INBOUND_Z_SLOPE = 0.8
INBOUND_SOURCE_BONUS: dict[LeadSource, float] = {
    LeadSource.WEB: 0.6,
    LeadSource.REFERRAL: 0.4,
    LeadSource.PARTNER: 0.3,
    LeadSource.EVENT: 0.2,
    LeadSource.OTHER: 0.0,
    LeadSource.OUTBOUND: -0.8,
}

# --- account / contact activities -----------------------------------------------------------

ACCOUNT_ACTIVITY_TYPE_WEIGHTS: tuple[tuple[ActivityType, float], ...] = (
    (ActivityType.CALL, 0.25),
    (ActivityType.EMAIL, 0.35),
    (ActivityType.MEETING, 0.20),
    (ActivityType.DEMO, 0.05),
    (ActivityType.FOLLOW_UP, 0.15),
)
ACCOUNT_ACTIVITY_DIRECTION_WEIGHTS: tuple[tuple[ActivityDirection, float], ...] = (
    (ActivityDirection.OUTBOUND, 0.55),
    (ActivityDirection.INBOUND, 0.35),
    (ActivityDirection.INTERNAL, 0.10),
)
# Activities are recorded up to this long after they happen (activities.created_at).
MAX_LOGGING_DELAY_HOURS = 48
ACTIVITY_BODY_SHARE = 0.5
