"""Forward-in-time lead simulation: engagement, conversion and disqualification.

One step per day of lead age, vectorized over all leads that are still open. On each day:

1. open leads produce Poisson activity counts (rate rises with latent quality z and
   decays with lead age);
2. engagement E(t) - a recency-decayed sum of event weights - is updated;
3. open leads may convert or be disqualified, with hazards that depend on z and on the
   engagement observed SO FAR.

Conversion therefore depends only on the past, and a terminal lead never produces another
lead activity: data available at T never encodes what happens after T. Activities occur
in the first `activity_day_fraction` of a day and status transitions at its end, so a
transition is strictly later than that day's activities.
"""

from dataclasses import dataclass

import numpy as np

from app.models.enums import ActivityDirection, ActivityType, LeadSource, LeadStatus
from ml.synthetic import config as c

DAY = 86_400

# Simulation states (index order of LeadStatus values used below).
NEW, WORKING, QUALIFIED, CONVERTED, DISQUALIFIED = range(5)
STATE_TO_STATUS = (
    LeadStatus.NEW,
    LeadStatus.WORKING,
    LeadStatus.QUALIFIED,
    LeadStatus.CONVERTED,
    LeadStatus.DISQUALIFIED,
)
DIRECTIONS = (ActivityDirection.INBOUND, ActivityDirection.OUTBOUND, ActivityDirection.INTERNAL)
INBOUND, OUTBOUND, INTERNAL = range(3)


@dataclass(frozen=True)
class LeadSimulation:
    statuses: list[LeadStatus]
    # Seconds after each lead's created_at.
    converted_offsets: np.ndarray  # -1 when not converted
    updated_offsets: np.ndarray  # most recent state change (0 = created_at)
    # One entry per lead activity.
    event_lead: np.ndarray  # lead index
    event_offset: np.ndarray  # seconds after the lead's created_at
    event_type: np.ndarray  # index into config.ACTIVITY_TYPES
    event_direction: np.ndarray  # index into DIRECTIONS


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_leads(
    rng: np.random.Generator,
    latent_quality: np.ndarray,
    sources: np.ndarray,
    created_offsets: np.ndarray,
    max_lead_age_days: int,
    params: c.SimulationParams = c.SIMULATION,
) -> LeadSimulation:
    n = latent_quality.size
    z = latent_quality
    # Whole days each lead can be simulated before as_of (right-censoring).
    age_limit = np.minimum(max_lead_age_days, (-created_offsets) // DAY)

    state = np.full(n, NEW, dtype=np.int8)
    engagement = np.zeros(n)
    updated = np.zeros(n, dtype=np.int64)
    converted = np.full(n, -1, dtype=np.int64)

    engagement_decay = 0.5 ** (1.0 / params.engagement_half_life_days)
    intensity = params.base_rate * np.exp(params.intensity_z * z)
    type_logits = np.asarray(c.ACTIVITY_TYPE_BASE_LOGITS) + np.outer(
        z, c.ACTIVITY_TYPE_Z_SLOPES
    )
    type_probs = np.exp(type_logits)
    type_cdf = np.cumsum(type_probs / type_probs.sum(axis=1, keepdims=True), axis=1)
    source_bonus = np.array([c.INBOUND_SOURCE_BONUS[LeadSource(s)] for s in sources])
    p_inbound = _sigmoid(c.INBOUND_BASE_LOGIT + c.INBOUND_Z_SLOPE * z + source_bonus)
    type_weights = np.array([c.ENGAGEMENT_WEIGHTS[t] for t in c.ACTIVITY_TYPES])

    activity_window = int(params.activity_day_fraction * DAY)
    events: list[tuple[np.ndarray, ...]] = []

    for day in range(max_lead_age_days):
        open_ = np.flatnonzero((state < CONVERTED) & (day < age_limit))
        if open_.size == 0:
            break
        engagement[open_] *= engagement_decay

        # 1. Activities today.
        rate = intensity[open_] * 0.5 ** (day / params.activity_half_life_days)
        counts = rng.poisson(rate)
        lead = np.repeat(open_, counts)
        if lead.size:
            kind = (rng.random(lead.size)[:, None] > type_cdf[lead]).sum(axis=1)
            internal = rng.random(lead.size) < c.INTERNAL_SHARE
            inbound = ~internal & (rng.random(lead.size) < p_inbound[lead])
            direction = np.where(internal, INTERNAL, np.where(inbound, INBOUND, OUTBOUND))
            offset = day * DAY + (rng.random(lead.size) * activity_window).astype(np.int64)
            events.append((lead, offset, kind, direction))

            # 2. Engagement from today's activities.
            weight = np.where(
                internal,
                c.INTERNAL_ENGAGEMENT_WEIGHT,
                type_weights[kind] + inbound * c.INBOUND_ENGAGEMENT_BONUS,
            )
            np.add.at(engagement, lead, weight)

        # End-of-day transition times (strictly after today's activities).
        transition = day * DAY + activity_window + (
            rng.random(open_.size) * (DAY - activity_window - 1)
        ).astype(np.int64)

        # Open-pipeline progress: NEW -> WORKING on first activity, -> QUALIFIED when engaged.
        had_activity = np.isin(open_, lead) if lead.size else np.zeros(open_.size, bool)
        start_working = (state[open_] == NEW) & had_activity
        qualify = (state[open_] <= WORKING) & had_activity & (
            engagement[open_] >= params.qualified_engagement
        )
        state[open_[start_working]] = WORKING
        state[open_[qualify]] = QUALIFIED
        progressed = start_working | qualify
        updated[open_[progressed]] = transition[progressed]

        # 3. Terminal outcomes from the history observed so far.
        log_engagement = np.log1p(engagement[open_])
        p_convert = _sigmoid(
            params.conversion_intercept
            + params.conversion_z * z[open_]
            + params.conversion_engagement * log_engagement
        )
        p_disqualify = _sigmoid(
            params.disqualify_intercept
            - params.disqualify_z * z[open_]
            - params.disqualify_engagement * log_engagement
            + params.disqualify_age * day / 30
        )
        draw = rng.random(open_.size)
        convert = draw < p_convert
        disqualify = ~convert & (draw < p_convert + p_disqualify)

        state[open_[convert]] = CONVERTED
        converted[open_[convert]] = transition[convert]
        state[open_[disqualify]] = DISQUALIFIED
        terminal = convert | disqualify
        updated[open_[terminal]] = transition[terminal]

    if events:
        event_lead, event_offset, event_type, event_direction = (
            np.concatenate(parts) for parts in zip(*events, strict=True)
        )
    else:
        empty = np.array([], dtype=np.int64)
        event_lead = event_offset = event_type = event_direction = empty

    return LeadSimulation(
        statuses=[STATE_TO_STATUS[s] for s in state],
        converted_offsets=converted,
        updated_offsets=updated,
        event_lead=event_lead,
        event_offset=event_offset,
        event_type=event_type,
        event_direction=event_direction,
    )


def activity_type(index: int) -> ActivityType:
    return c.ACTIVITY_TYPES[index]


def activity_direction(index: int) -> ActivityDirection:
    return DIRECTIONS[index]
