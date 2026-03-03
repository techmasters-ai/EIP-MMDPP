"""Shared Pydantic schema utilities."""

import uuid
from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class APIModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class CursorPage(APIModel, Generic[T]):
    """Cursor-based pagination wrapper."""

    items: list[T]
    next_cursor: Optional[str] = None
    total_hint: Optional[int] = None
