"""Activity subjects and bodies.

Text depends only on the activity's type and direction, never on the lead's outcome, so
it cannot leak conversion (no "won", "converted", "closed", ...).
"""

import numpy as np

from app.models.enums import ActivityDirection, ActivityType

SUBJECTS: dict[ActivityType, tuple[str, ...]] = {
    ActivityType.CALL: ("Intro call", "Discovery call", "Check-in call", "Pricing questions"),
    ActivityType.EMAIL: ("Introduction", "Follow-up email", "Product information", "Resources"),
    ActivityType.MEETING: ("Discovery meeting", "Requirements workshop", "Stakeholder meeting"),
    ActivityType.DEMO: ("Product demo", "Technical walkthrough"),
    ActivityType.FOLLOW_UP: ("Follow-up", "Next steps", "Checking in"),
}
TOPICS = (
    "pricing",
    "integrations",
    "the security review",
    "onboarding",
    "the implementation timeline",
    "reporting features",
)
BODIES: dict[ActivityDirection, tuple[str, ...]] = {
    ActivityDirection.INBOUND: (
        "Customer reached out with questions about {topic}.",
        "Received a reply asking for more detail on {topic}.",
    ),
    ActivityDirection.OUTBOUND: (
        "Reached out to discuss {topic}.",
        "Shared information about {topic}.",
    ),
    ActivityDirection.INTERNAL: (
        "Internal note: prepare material on {topic}.",
        "Team sync about {topic}.",
    ),
}


def subject_and_body(
    rng: np.random.Generator, kind: ActivityType, direction: ActivityDirection, with_body: bool
) -> tuple[str, str | None]:
    subjects = SUBJECTS[kind]
    subject = subjects[int(rng.integers(len(subjects)))]
    if not with_body:
        return subject, None
    templates = BODIES[direction]
    template = templates[int(rng.integers(len(templates)))]
    return subject, template.format(topic=TOPICS[int(rng.integers(len(TOPICS)))])
