"""Account timeline schemas: activities and notes in one newest-first stream."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field

from app.models.activity import Activity
from app.models.note import Note
from app.schemas.activity import ActivityRead
from app.schemas.common import Page
from app.schemas.note import NoteRead

EVENT_TIMESTAMP_DESCRIPTION = (
    "Ordering timestamp (API-only): occurred_at for activities, created_at for notes."
)


class ActivityTimelineItem(ActivityRead):
    record_type: Literal["activity"] = "activity"
    event_timestamp: datetime = Field(description=EVENT_TIMESTAMP_DESCRIPTION)


class NoteTimelineItem(NoteRead):
    record_type: Literal["note"] = "note"
    event_timestamp: datetime = Field(description=EVENT_TIMESTAMP_DESCRIPTION)


TimelineItem = Annotated[
    ActivityTimelineItem | NoteTimelineItem, Field(discriminator="record_type")
]


class TimelinePage(Page[TimelineItem]):
    """Named page type so the OpenAPI schema is readable."""


def timeline_item(record: Activity | Note) -> ActivityTimelineItem | NoteTimelineItem:
    """Build the API item for a timeline record; event_timestamp is derived, not stored."""
    if isinstance(record, Activity):
        activity = ActivityRead.model_validate(record).model_dump()
        return ActivityTimelineItem(**activity, event_timestamp=record.occurred_at)
    note = NoteRead.model_validate(record).model_dump()
    return NoteTimelineItem(**note, event_timestamp=record.created_at)
