"""T0-3 验收：事件信封字段与设计文档 8.2 完全一致。

设计文档 8.2 给出的样例（逐字保留，作为契约基准）：

    {
      "event_id": "evt_001",
      "event_type": "marketing_script.generated",
      "event_version": "1.0",
      "occurred_at": "2026-08-24T12:00:00+08:00",
      "workflow_run_id": "wf_run_001",
      "task_run_id": "task_004",
      "correlation_id": "corr_001",
      "producer": "marketing-script-service",
      "subject": {"type": "marketing_script", "id": "ms_001", "version": 3},
      "data": {"status": "generated"}
    }
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.domain.events import (
    CURRENT_EVENT_VERSION,
    EventEnvelope,
    EventType,
    EventVersion,
    SubjectType,
)

DESIGN_DOC_SAMPLE = {
    "event_id": "evt_001",
    "event_type": "marketing_script.generated",
    "event_version": "1.0",
    "occurred_at": "2026-08-24T12:00:00+08:00",
    "workflow_run_id": "wf_run_001",
    "task_run_id": "task_004",
    "correlation_id": "corr_001",
    "producer": "marketing-script-service",
    "subject": {"type": "marketing_script", "id": "ms_001", "version": 3},
    "data": {"status": "generated"},
}


# --------------------------------------------------------------- 字段一致性


def test_design_doc_sample_round_trips() -> None:
    envelope = EventEnvelope.model_validate(DESIGN_DOC_SAMPLE)
    dumped = envelope.model_dump(mode="json")

    assert dumped["event_id"] == "evt_001"
    assert dumped["event_type"] == "marketing_script.generated"
    assert dumped["event_version"] == "1.0"
    assert dumped["workflow_run_id"] == "wf_run_001"
    assert dumped["task_run_id"] == "task_004"
    assert dumped["correlation_id"] == "corr_001"
    assert dumped["producer"] == "marketing-script-service"
    assert dumped["subject"] == {"type": "marketing_script", "id": "ms_001", "version": 3}
    assert dumped["data"] == {"status": "generated"}


def test_envelope_top_level_fields_match_design_doc_exactly() -> None:
    """顶层字段集合不允许多也不允许少。

    多一个字段意味着有人绕过 `data` 往信封上挂业务数据，那会让所有消费方被迫
    跟着升级；少一个字段意味着回溯链断了。
    """
    assert set(EventEnvelope.model_fields) == set(DESIGN_DOC_SAMPLE)


def test_business_payload_must_go_into_data() -> None:
    resp = dict(DESIGN_DOC_SAMPLE) | {"script_id": "ms_001"}
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate(resp)


def test_current_version_is_wellformed() -> None:
    assert str(EventVersion.parse(CURRENT_EVENT_VERSION)) == CURRENT_EVENT_VERSION


# --------------------------------------------------------------- 版本兼容


@pytest.mark.parametrize(
    ("producer", "consumer", "compatible"),
    [
        ("1.0", "1.0", True),
        ("1.3", "1.0", True),  # 生产方新增可选字段，消费方忽略即可
        ("1.0", "1.3", True),  # 消费方认识更多字段，旧事件仍可处理
        ("2.0", "1.9", False),  # 破坏性变更必须拒绝，不允许尽力解析
        ("1.9", "2.0", False),
    ],
)
def test_major_version_decides_compatibility(
    producer: str, consumer: str, compatible: bool
) -> None:
    envelope = EventEnvelope.model_validate(DESIGN_DOC_SAMPLE | {"event_version": producer})
    assert envelope.is_compatible_with(consumer) is compatible


@pytest.mark.parametrize("raw", ["1", "1.0.0", "v1.0", "", "0.1.x"])
def test_malformed_version_is_rejected(raw: str) -> None:
    with pytest.raises(ValidationError):
        EventEnvelope.model_validate(DESIGN_DOC_SAMPLE | {"event_version": raw})


# --------------------------------------------------------------- 主体规则


def test_artifact_subject_requires_version() -> None:
    """没有版本号的产物事件会让「成片由哪个分析版本产生」无从回答。"""
    with pytest.raises(ValidationError, match="必须带 version"):
        EventEnvelope.model_validate(
            DESIGN_DOC_SAMPLE | {"subject": {"type": "marketing_script", "id": "ms_001"}}
        )


def test_lifecycle_subject_must_not_carry_version() -> None:
    envelope = EventEnvelope.model_validate(
        DESIGN_DOC_SAMPLE
        | {
            "event_type": "task_run.failed",
            "subject": {"type": "task_run", "id": "task_004"},
        }
    )
    assert envelope.subject.type is SubjectType.TASK_RUN
    assert envelope.subject.version is None


def test_event_type_and_subject_type_must_agree() -> None:
    """类型错配在语法上合法，但会静默产出错误的血缘。"""
    with pytest.raises(ValidationError, match="主体类型应为"):
        EventEnvelope.model_validate(
            DESIGN_DOC_SAMPLE | {"subject": {"type": "storyboard", "id": "sb_001", "version": 1}}
        )


def test_every_event_type_has_a_resolvable_subject_type() -> None:
    """命名规则不是文档约定，是可执行的：每个事件类型的前缀都必须是合法主体。"""
    for event_type in EventType:
        assert isinstance(event_type.subject_type, SubjectType)


def test_every_event_type_uses_past_tense_suffix() -> None:
    """事件描述已发生的事实，不是待办的命令。"""
    for event_type in EventType:
        action = event_type.split(".", 1)[1]
        assert action.endswith(("ed", "ted")), event_type


# --------------------------------------------------------------- 时间


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError, match="必须带时区"):
        EventEnvelope.model_validate(
            DESIGN_DOC_SAMPLE | {"occurred_at": datetime(2026, 8, 24, 12, 0, 0)}  # noqa: DTZ001
        )


def test_aware_datetime_is_accepted() -> None:
    envelope = EventEnvelope.model_validate(
        DESIGN_DOC_SAMPLE | {"occurred_at": datetime(2026, 8, 24, 12, 0, tzinfo=UTC)}
    )
    assert envelope.occurred_at.tzinfo is not None


# --------------------------------------------------------------- 独立入口


def test_workflow_ids_are_optional_for_standalone_invocations() -> None:
    """设计文档 5.2 允许直接调用单个能力，此时没有工作流上下文。"""
    payload = dict(DESIGN_DOC_SAMPLE)
    del payload["workflow_run_id"]
    del payload["task_run_id"]
    envelope = EventEnvelope.model_validate(payload)
    assert envelope.workflow_run_id is None
    assert envelope.correlation_id == "corr_001"
