"""Account endpoints."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import AccountServiceDep
from app.api.pagination import PageParamsDep
from app.models.account import Account
from app.schemas.account import AccountCreate, AccountRead
from app.schemas.common import Page
from app.services.pagination import PageResult

router = APIRouter(prefix="/accounts", tags=["accounts"])


@router.post(
    "",
    response_model=AccountRead,
    status_code=status.HTTP_201_CREATED,
    responses={422: {"description": "Validation error, or owner_id is unknown or inactive"}},
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
def get_account(account_id: uuid.UUID, service: AccountServiceDep) -> Account:
    return service.get_account(account_id)
