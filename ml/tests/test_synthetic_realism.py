"""Broad realism and leakage checks.

These are sanity bands, not calibration targets: the generator is not tuned to hit them.
"""

from collections import defaultdict

from app.models.enums import ActivityDirection, ActivityType, LeadSource
from ml.synthetic.generator import SyntheticDataset
from ml.synthetic.summary import early_snapshot, summary_stats


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def test_overall_conversion_rate_is_plausible(realism_dataset: SyntheticDataset) -> None:
    rate = summary_stats(realism_dataset)["leads"]["conversion_rate"]

    assert 0.08 <= rate <= 0.25


def test_referral_and_partner_convert_more_than_outbound(realism_dataset: SyntheticDataset) -> None:
    by_source: dict[LeadSource, list[bool]] = defaultdict(list)
    for lead in realism_dataset.leads:
        by_source[lead["lead_source"]].append(lead["converted_at"] is not None)

    outbound = _rate(by_source[LeadSource.OUTBOUND])
    assert _rate(by_source[LeadSource.REFERRAL]) > outbound
    assert _rate(by_source[LeadSource.PARTNER]) > outbound


def test_larger_and_more_valuable_leads_convert_more(realism_dataset: SyntheticDataset) -> None:
    leads = summary_stats(realism_dataset)["leads"]

    size = leads["conversion_rate_by_company_size_quartile"]
    value = leads["conversion_rate_by_estimated_value_quartile"]
    assert size[3] > size[0]
    assert value[3] > value[0]


def test_engagement_before_t_predicts_conversion_after_t(realism_dataset: SyntheticDataset) -> None:
    counts, _, labels = early_snapshot(realism_dataset)
    none = [label for count, label in zip(counts, labels, strict=True) if count == 0]
    some = [label for count, label in zip(counts, labels, strict=True) if count >= 3]

    assert none and some
    assert _rate(some) > _rate(none)


def test_meetings_and_inbound_replies_before_t_raise_conversion(
    realism_dataset: SyntheticDataset,
) -> None:
    from datetime import timedelta

    from ml.synthetic.conventions import lead_terminal_time

    as_of = realism_dataset.config.as_of
    strong: dict = defaultdict(bool)
    for activity in realism_dataset.activities:
        if activity["lead_id"] and (
            activity["activity_type"] in (ActivityType.MEETING, ActivityType.DEMO)
            or activity["direction"] == ActivityDirection.INBOUND
        ):
            strong[(activity["lead_id"], activity["occurred_at"])] = True
    with_signal, without_signal = [], []
    for lead in realism_dataset.leads:
        t = lead["created_at"] + timedelta(days=14)
        terminal = lead_terminal_time(lead["status"], lead["converted_at"], lead["updated_at"])
        if t + timedelta(days=30) > as_of or (terminal is not None and terminal <= t):
            continue
        signal = any(
            lead_id == lead["id"] and occurred <= t for lead_id, occurred in strong
        )
        converted = lead["converted_at"]
        label = converted is not None and t < converted <= t + timedelta(days=30)
        (with_signal if signal else without_signal).append(label)

    assert _rate(with_signal) > _rate(without_signal)


def test_label_is_predictable_but_not_deterministic(realism_dataset: SyntheticDataset) -> None:
    # Broad leakage canary: a simple engagement score using only data <= T must carry
    # signal, but must not (near-)perfectly separate the future label.
    auc = summary_stats(realism_dataset)["point_in_time_check"]["engagement_score_auc"]

    assert 0.6 < auc < 0.92
