"""Contact API schemas."""

import uuid
from datetime import datetime

from app.schemas.common import CrmEmail, Phone, RequestModel, ResponseModel, Text255


class ContactCreate(RequestModel):
    account_id: uuid.UUID
    first_name: Text255
    last_name: Text255
    email: CrmEmail | None = None
    phone: Phone | None = None
    job_title: Text255 | None = None


class ContactRead(ResponseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    first_name: str
    last_name: str
    email: str | None
    phone: str | None
    job_title: str | None
    created_at: datetime
    updated_at: datetime
