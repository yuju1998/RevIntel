"""Account endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.auth import UNAUTHORIZED_RESPONSE, RequireUser
from app.api.deps import AccountServiceDep, TimelineServiceDep
from app.api.pagination import PageParamsDep
from app.models.account import Account
from app.schemas.account import AccountCreate, AccountRead
from app.schemas.common import Page
from app.schemas.timeline import TimelinePage, timeline_item
from app.services.pagination import PageResult

router = APIRouter(prefix="/accounts", tags=["accounts"])


@router.post(
    "",
    response_model=AccountRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[RequireUser],
    responses={
        **UNAUTHORIZED_RESPONSE,
        422: {"description": "Validation error, or owner_id is unknown or inactive"},
    },
)
def create_account(payload: AccountCreate, service: AccountServiceDep) -> Account:
    return service.create_account(payload)


@router.get("", response_model=Page[AccountRead])
def list_accounts(
    service: AccountServiceDep,
    page: PageParamsDep,
    industry: Annotated[str | None, Query(max_length=255, description="Exact match")] = None,
) -> PageResult[Account]:
    return service.list_accounts(industry=industry, limit=page.limit, offset=page.offset)


@router.get(
    "/{account_id}",
    response_model=AccountRead,
    responses={404: {"description": "Account not found"}},
)
def get_account(account_id: uuid.UUID, service: AccountServiceDep) -> AccountRead:
    """Served from the Redis cache when possible (see docs/API.md)."""
    return service.get_account(account_id)


@router.get(
    "/{account_id}/timeline",
    response_model=TimelinePage,
    responses={404: {"description": "Account not found"}},
)
def get_account_timeline(
    account_id: uuid.UUID, service: TimelineServiceDep, page: PageParamsDep
) -> TimelinePage:
    """Activities and notes for the account, newest first by event_timestamp.

    Ties are ordered by record_type, then id. Lead-only records never appear here.
    """
    result = service.get_account_timeline(account_id, limit=page.limit, offset=page.offset)
    return TimelinePage(
        items=[timeline_item(record) for record in result.items],
        total=result.total,
        limit=result.limit,
        offset=result.offset,
    )
