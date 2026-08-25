"""幂等与审计的内存实现。"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import UTC, datetime

from app.domain.errors import CapabilityError, ErrorCode
from app.domain.idempotency import (
    AuditAction,
    AuditEntry,
    IdempotencyOutcome,
    IdempotencyRecord,
    IdempotencyReservation,
)
from app.domain.refs import ArtifactRef


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}

    def reserve(self, key: str, fingerprint: str) -> IdempotencyReservation:
        existing = self._records.get(key)
        if existing is None:
            record = IdempotencyRecord(
                key=key, fingerprint=fingerprint, created_at=datetime.now(UTC)
            )
            self._records[key] = record
            return IdempotencyReservation(outcome=IdempotencyOutcome.FRESH, record=record)

        if existing.fingerprint != fingerprint:
            # 静默当成重复请求跳过会让调用方以为新参数生效了，实际返回旧结果，
            # 而这种错在成片出来之前不会有任何症状。
            raise CapabilityError(
                ErrorCode.IDEMPOTENCY_KEY_REUSED,
                f"幂等键 {key} 已用于不同的请求内容，拒绝执行",
            )

        return IdempotencyReservation(outcome=IdempotencyOutcome.REPLAY, record=existing)

    def complete(self, key: str, outputs: tuple[ArtifactRef, ...]) -> IdempotencyRecord:
        record = self._records.get(key)
        if record is None:
            raise CapabilityError(
                ErrorCode.INTERNAL_ERROR, f"幂等键 {key} 未登记就被标记完成，这是缺陷"
            )
        record.outputs = outputs
        record.completed_at = datetime.now(UTC)
        return record

    def get(self, key: str) -> IdempotencyRecord | None:
        return self._records.get(key)


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []
        self._by_subject: dict[str, list[AuditEntry]] = defaultdict(list)
        self._by_actor: dict[str, list[AuditEntry]] = defaultdict(list)

    def record(
        self,
        action: AuditAction,
        *,
        actor: str,
        subject: str,
        summary: str,
    ) -> AuditEntry:
        entry = AuditEntry(
            id=f"audit_{uuid.uuid4().hex[:16]}",
            action=action,
            actor=actor,
            subject=subject,
            summary=summary,
            occurred_at=datetime.now(UTC),
        )
        # 审计只追加。提供删除或修改接口等于让审计失去意义。
        self._entries.append(entry)
        self._by_subject[subject].append(entry)
        self._by_actor[actor].append(entry)
        return entry

    def entries_for(self, subject: str) -> tuple[AuditEntry, ...]:
        return tuple(self._by_subject.get(subject, ()))

    def entries_by(self, actor: str) -> tuple[AuditEntry, ...]:
        return tuple(self._by_actor.get(actor, ()))
