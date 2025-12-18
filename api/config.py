from enum import Enum


class Config(Enum):
    STATUS = "active"
    ALLOWED_TYPES = ["audio/wav", "audio/ogg"]


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
