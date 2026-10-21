"""Domain enums shared by ORM models and API schemas.

Member values equal their names; the database stores these uppercase values in
VARCHAR(32) columns guarded by CHECK constraints (see docs/DATABASE.md).
"""

from enum import StrEnum


class UserRole(StrEnum):
    ADMIN = "ADMIN"
    SALES_REP = "SALES_REP"


class LeadSource(StrEnum):
    WEB = "WEB"
    REFERRAL = "REFERRAL"
    EVENT = "EVENT"
    OUTBOUND = "OUTBOUND"
    PARTNER = "PARTNER"
    OTHER = "OTHER"


class LeadStatus(StrEnum):
    NEW = "NEW"
    WORKING = "WORKING"
    QUALIFIED = "QUALIFIED"
    CONVERTED = "CONVERTED"
    DISQUALIFIED = "DISQUALIFIED"


class OpportunityStage(StrEnum):
    PROSPECTING = "PROSPECTING"
    QUALIFICATION = "QUALIFICATION"
    PROPOSAL = "PROPOSAL"
    NEGOTIATION = "NEGOTIATION"
    CLOSED_WON = "CLOSED_WON"
    CLOSED_LOST = "CLOSED_LOST"


class ActivityType(StrEnum):
    CALL = "CALL"
    EMAIL = "EMAIL"
    MEETING = "MEETING"
    DEMO = "DEMO"
    FOLLOW_UP = "FOLLOW_UP"


class ActivityDirection(StrEnum):
    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"
    INTERNAL = "INTERNAL"
