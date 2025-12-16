from typing import Literal
from pydantic import BaseModel


class APIStatusResponse(BaseModel):
    status: Literal["active", "not active"]

