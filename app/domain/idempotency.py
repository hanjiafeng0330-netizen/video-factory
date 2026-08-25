"""幂等与审计（设计文档 8.3、13.1）。

设计文档 8.3 要求「所有写操作必须支持幂等键」，13.1 要求敏感操作留审计。

幂等这件事容易做成「见过这个 key 就跳过」，那是不够的：调用方复用幂等键但换了
输入的情况必须被**发现**，而不是被静默当成重复请求跳过——后者会让调用方以为
新参数生效了，实际返回的是旧结果，而这种错在成片出来之前不会有任何症状。

所以这里用「幂等键 + 请求指纹」两段式：
- key 相同、指纹相同  → 重复请求，返回既有结果；
- key 相同、指纹不同  → 冲突，报 idempotency_key_reused；
- key 不同            → 新请求。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.domain.refs import ArtifactRef


class IdempotencyOutcome(StrEnum):
    FRESH = "fresh"
    """首次见到该幂等键，调用方应当继续执行。"""

    REPLAY = "replay"
    """同键同指纹，已有结果，调用方应当直接返回既有结果。"""


class IdempotencyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=8, max_length=128)
    fingerprint: str = Field(min_length=8, max_length=128)
    outputs: tuple[ArtifactRef, ...] = ()
    created_at: datetime
    completed_at: datetime | None = None


class IdempotencyReservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: IdempotencyOutcome
    record: IdempotencyRecord


class IdempotencyStore(Protocol):
    def reserve(self, key: str, fingerprint: str) -> IdempotencyReservation:
        """登记一次写操作。

        实现必须在同键不同指纹时抛 `idempotency_key_reused`，而不是当成重复请求。
        """
        ...

    def complete(self, key: str, outputs: tuple[ArtifactRef, ...]) -> IdempotencyRecord:
        """记录结果，使后续同键同指纹的请求可以直接回放。"""
        ...

    def get(self, key: str) -> IdempotencyRecord | None: ...


class AuditAction(StrEnum):
    """需要审计的敏感操作（设计文档 13.1）。

    设计文档点名了四类：产品声明修改、脚本批准、供应商配置修改、成片导出与删除。
    这里额外加入提示词变更——它直接决定生成内容，改动必须可追责（计划 1.3 节）。
    """

    PRODUCT_CLAIM_MODIFIED = "product_claim_modified"
    SCRIPT_APPROVED = "script_approved"
    SCRIPT_REJECTED = "script_rejected"
    PROVIDER_CONFIG_CHANGED = "provider_config_changed"
    PROMPT_CHANGED = "prompt_changed"
    VIDEO_EXPORTED = "video_exported"
    ASSET_DELETED = "asset_deleted"


class AuditEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    action: AuditAction
    actor: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=128)
    """被操作对象的标识。产物用 `type:id@vN`，其他用各自的业务 id。"""

    summary: str = Field(min_length=1, max_length=500)
    occurred_at: datetime


class AuditLog(Protocol):
    def record(
        self,
        action: AuditAction,
        *,
        actor: str,
        subject: str,
        summary: str,
    ) -> AuditEntry: ...

    def entries_for(self, subject: str) -> tuple[AuditEntry, ...]: ...

    def entries_by(self, actor: str) -> tuple[AuditEntry, ...]: ...
