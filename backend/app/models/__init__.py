"""ORM models.

Import every model module here: Alembic's env.py imports this package so that all
models are registered on `Base.metadata` before autogenerate runs.
"""

from app.models.account import Account
from app.models.activity import Activity
from app.models.contact import Contact
from app.models.lead import Lead
from app.models.lead_prediction import LeadPrediction
from app.models.note import Note
from app.models.opportunity import Opportunity
from app.models.user import User

__all__ = [
    "Account",
    "Activity",
    "Contact",
    "Lead",
    "LeadPrediction",
    "Note",
    "Opportunity",
    "User",
]
