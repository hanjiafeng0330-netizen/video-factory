"""异步任务的状态与记录（设计文档 5.3）。

本文件只定义状态集合与记录结构。**合法跃迁的状态机、持久化、重试退避、
人工队列属于 T2-1**，这里不实现，以免 M0 阶段写出一个之后要推翻的半成品。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.capability import CapabilityRequest, CapabilityResult


class JobStatus(StrEnum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    WAITING_REVIEW = "WAITING_REVIEW"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    capability: str
    status: JobStatus
    request: CapabilityRequest
    attempt: int = Field(default=1, ge=1)
    result: CapabilityResult | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
