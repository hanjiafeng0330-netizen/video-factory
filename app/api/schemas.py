"""API 出入参模型。

与 `app.domain` 的类型分开：领域类型服务于业务规则，API 类型服务于对外契约，
两者的演进节奏不同，混用会让改一个字段就动到另一层。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.capability import CapabilitySpec
from app.domain.jobs import JobStatus
from app.domain.prompts import ResolvedPrompt
from app.domain.refs import ArtifactRef


class ExecuteRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_refs: tuple[ArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = Field(default_factory=dict)
    resolved_prompts: tuple[ResolvedPrompt, ...] = ()
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]


class ResultBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_refs: tuple[ArtifactRef, ...]
    metrics: Mapping[str, float]
    notes: tuple[str, ...]


class JobBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    capability: str
    status: JobStatus
    attempt: int
    request_fingerprint: str
    result: ResultBody | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class CapabilityListBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capabilities: tuple[CapabilitySpec, ...]


class ErrorBody(BaseModel):
    """统一错误响应（设计文档 8.3）。"""

    model_config = ConfigDict(extra="forbid")

    code: str
    category: str
    message: str
    retryable: bool
    suggested_action: str
