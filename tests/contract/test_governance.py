"""T1-4 验收：幂等与审计（设计文档 8.3、13.1）。"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from app.domain.capability import CapabilityRequest
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.idempotency import (
    AuditAction,
    AuditLog,
    IdempotencyOutcome,
    IdempotencyStore,
)
from app.domain.refs import ArtifactRef, ArtifactType
from app.storage.memory_governance import InMemoryAuditLog, InMemoryIdempotencyStore

IDEMPOTENCY_FACTORIES: dict[str, Callable[[], IdempotencyStore]] = {
    "memory": InMemoryIdempotencyStore
}
AUDIT_FACTORIES: dict[str, Callable[[], AuditLog]] = {"memory": InMemoryAuditLog}

OUTPUT = ArtifactRef(type=ArtifactType.VIDEO_ANALYSIS, id="va_001", version=1)


@pytest.fixture(params=sorted(IDEMPOTENCY_FACTORIES), ids=sorted(IDEMPOTENCY_FACTORIES))
def store(request: pytest.FixtureRequest) -> Iterator[IdempotencyStore]:
    yield IDEMPOTENCY_FACTORIES[str(request.param)]()


@pytest.fixture(params=sorted(AUDIT_FACTORIES), ids=sorted(AUDIT_FACTORIES))
def audit(request: pytest.FixtureRequest) -> Iterator[AuditLog]:
    yield AUDIT_FACTORIES[str(request.param)]()


# --------------------------------------------------------------- 幂等


def test_first_reservation_is_fresh(store: IdempotencyStore) -> None:
    assert store.reserve("job-key-0001", "sha256:aaa").outcome is IdempotencyOutcome.FRESH


def test_same_key_same_fingerprint_is_replay(store: IdempotencyStore) -> None:
    """设计文档 8.3：事件至少投递一次，消费方必须幂等。重复请求不应重复执行。"""
    store.reserve("job-key-0001", "sha256:aaa")
    store.complete("job-key-0001", (OUTPUT,))

    replay = store.reserve("job-key-0001", "sha256:aaa")
    assert replay.outcome is IdempotencyOutcome.REPLAY
    assert replay.record.outputs == (OUTPUT,)


def test_same_key_different_fingerprint_conflicts(store: IdempotencyStore) -> None:
    """核心断言：复用幂等键必须被发现，不能静默当成重复请求。

    静默跳过会让调用方以为新参数生效了，实际返回旧结果，而这种错在成片出来之前
    不会有任何症状。
    """
    store.reserve("job-key-0001", "sha256:aaa")
    with pytest.raises(CapabilityError) as excinfo:
        store.reserve("job-key-0001", "sha256:bbb")
    assert excinfo.value.code is ErrorCode.IDEMPOTENCY_KEY_REUSED


def test_different_keys_are_independent(store: IdempotencyStore) -> None:
    assert store.reserve("job-key-0001", "sha256:aaa").outcome is IdempotencyOutcome.FRESH
    assert store.reserve("job-key-0002", "sha256:aaa").outcome is IdempotencyOutcome.FRESH


def test_replay_before_completion_returns_no_outputs(store: IdempotencyStore) -> None:
    """并发的第二个请求会看到「已登记但未完成」，不应误以为拿到了结果。"""
    store.reserve("job-key-0001", "sha256:aaa")
    replay = store.reserve("job-key-0001", "sha256:aaa")
    assert replay.outcome is IdempotencyOutcome.REPLAY
    assert replay.record.outputs == ()
    assert replay.record.completed_at is None


def test_completing_unreserved_key_is_a_defect(store: IdempotencyStore) -> None:
    with pytest.raises(CapabilityError) as excinfo:
        store.complete("job-key-9999", (OUTPUT,))
    assert excinfo.value.code is ErrorCode.INTERNAL_ERROR


def test_request_fingerprint_feeds_the_store(store: IdempotencyStore) -> None:
    """指纹来自 CapabilityRequest.fingerprint()，两者必须能直接对接。"""
    same_key = "echo-va001-pp001"
    first = CapabilityRequest(parameters={"candidate_count": 3}, idempotency_key=same_key)
    changed = CapabilityRequest(parameters={"candidate_count": 5}, idempotency_key=same_key)

    store.reserve(first.idempotency_key, first.fingerprint())
    with pytest.raises(CapabilityError) as excinfo:
        store.reserve(changed.idempotency_key, changed.fingerprint())
    assert excinfo.value.code is ErrorCode.IDEMPOTENCY_KEY_REUSED


# --------------------------------------------------------------- 审计


def test_audit_records_who_did_what_to_what(audit: AuditLog) -> None:
    entry = audit.record(
        AuditAction.SCRIPT_APPROVED,
        actor="reviewer_001",
        subject="marketing_script:ms_001@v3",
        summary="批准，卖点与证据一致",
    )
    assert entry.actor == "reviewer_001"
    assert entry.occurred_at.tzinfo is not None


def test_audit_is_queryable_by_subject_and_actor(audit: AuditLog) -> None:
    audit.record(
        AuditAction.SCRIPT_APPROVED,
        actor="reviewer_001",
        subject="marketing_script:ms_001@v3",
        summary="批准",
    )
    audit.record(
        AuditAction.VIDEO_EXPORTED,
        actor="ops_002",
        subject="video_output:vo_001@v1",
        summary="导出交付",
    )

    assert len(audit.entries_for("marketing_script:ms_001@v3")) == 1
    assert len(audit.entries_by("reviewer_001")) == 1
    assert len(audit.entries_by("ops_002")) == 1


def test_audit_is_append_only(audit: AuditLog) -> None:
    """提供删除或修改接口等于让审计失去意义。"""
    forbidden = {"delete", "remove", "update", "purge", "clear"}
    assert forbidden.isdisjoint(dir(audit))


def test_repeated_actions_all_leave_traces(audit: AuditLog) -> None:
    """同一对象被反复操作时每次都要留痕，不能只保留最后一次。"""
    for i in range(3):
        audit.record(
            AuditAction.PROMPT_CHANGED,
            actor="admin_001",
            subject="prompt:script_gen.system",
            summary=f"第 {i + 1} 次修改",
        )
    assert len(audit.entries_for("prompt:script_gen.system")) == 3


def test_design_doc_sensitive_actions_are_all_covered() -> None:
    """设计文档 13.1 点名的四类敏感操作必须都有对应动作。"""
    actions = set(AuditAction)
    assert AuditAction.PRODUCT_CLAIM_MODIFIED in actions
    assert AuditAction.SCRIPT_APPROVED in actions
    assert AuditAction.PROVIDER_CONFIG_CHANGED in actions
    assert AuditAction.VIDEO_EXPORTED in actions
    assert AuditAction.ASSET_DELETED in actions
    # 提示词改动直接决定生成内容，必须可追责（计划 1.3 节）
    assert AuditAction.PROMPT_CHANGED in actions
