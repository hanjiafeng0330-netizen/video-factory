"""统一事件信封（设计文档 8.2）。

事件是模块之间唯一的异步耦合点，一旦信封字段随手扩散，后面就没法在不停机的
情况下换消费方。这里把信封锁成固定形状：**顶层字段封闭（extra="forbid"），
业务数据一律放进 `data`。**

版本兼容规则（设计文档 8.3「事件至少投递一次，消费方必须幂等」的前置条件）：

- `event_version` 形如 `MAJOR.MINOR`；
- **同 MAJOR 即兼容**。MINOR 递增只允许在 `data` 里新增可选字段，消费方忽略
  不认识的字段即可，因此生产方 MINOR 高于或低于消费方都能处理；
- **MAJOR 不同即不兼容**，消费方必须拒绝处理而不是尽力解析——半懂的事件会
  产出半对的产物，那比直接失败更难排查；
- 破坏性变更（删字段、改语义、改类型）必须升 MAJOR，且新旧事件并行一段时间。

本文件只定义信封与规则。事件总线（投递、去重、事务边界）属于 T2-3。
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.domain.refs import ArtifactType

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)$")


class EventVersion(BaseModel):
    """事件信封版本。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    major: Annotated[int, Field(ge=1)]
    minor: Annotated[int, Field(ge=0)]

    @classmethod
    def parse(cls, raw: str) -> EventVersion:
        match = _VERSION_PATTERN.match(raw)
        if match is None:
            raise ValueError(f"事件版本格式非法：{raw!r}，应为 MAJOR.MINOR")
        return cls(major=int(match.group(1)), minor=int(match.group(2)))

    def is_compatible_with(self, consumer: EventVersion) -> bool:
        """本事件能否被声明支持 `consumer` 版本的消费方处理。"""
        return self.major == consumer.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


CURRENT_EVENT_VERSION = "1.0"


class SubjectType(StrEnum):
    """事件主体类型。

    产物类主体沿用 `ArtifactType`；此外还有三类非产物主体，用于流程与审核的
    生命周期事件。它们不带版本号，因为它们不是可回溯的产物。
    """

    HOT_VIDEO = ArtifactType.HOT_VIDEO
    MEDIA_ASSET = ArtifactType.MEDIA_ASSET
    PREPROCESS_RESULT = ArtifactType.PREPROCESS_RESULT
    VIDEO_ANALYSIS = ArtifactType.VIDEO_ANALYSIS
    SCRIPT_PATTERN = ArtifactType.SCRIPT_PATTERN
    PRODUCT_PROFILE = ArtifactType.PRODUCT_PROFILE
    SELLING_POINT_SET = ArtifactType.SELLING_POINT_SET
    MARKETING_SCRIPT = ArtifactType.MARKETING_SCRIPT
    STORYBOARD = ArtifactType.STORYBOARD
    VIDEO_JOB = ArtifactType.VIDEO_JOB
    VIDEO_OUTPUT = ArtifactType.VIDEO_OUTPUT
    QUALITY_REPORT = ArtifactType.QUALITY_REPORT

    WORKFLOW_RUN = "workflow_run"
    TASK_RUN = "task_run"
    REVIEW = "review"


_NON_ARTIFACT_SUBJECTS = frozenset(
    {SubjectType.WORKFLOW_RUN, SubjectType.TASK_RUN, SubjectType.REVIEW}
)


class EventType(StrEnum):
    """已登记的事件类型。

    命名规则：`{主体}.{已发生的动作过去式}`。用过去式是刻意的——事件描述已经
    发生的事实，而不是待办的命令；`script.generate` 那样的命名会诱使消费方把
    事件当指令用，从而把编排逻辑散进消费方。
    """

    HOT_VIDEO_REGISTERED = "hot_video.registered"
    MEDIA_ASSET_STORED = "media_asset.stored"
    PREPROCESS_RESULT_COMPLETED = "preprocess_result.completed"
    VIDEO_ANALYSIS_COMPLETED = "video_analysis.completed"
    SCRIPT_PATTERN_EXTRACTED = "script_pattern.extracted"
    SCRIPT_PATTERN_APPROVED = "script_pattern.approved"
    PRODUCT_PROFILE_UPDATED = "product_profile.updated"
    SELLING_POINT_SET_EXTRACTED = "selling_point_set.extracted"
    MARKETING_SCRIPT_GENERATED = "marketing_script.generated"
    MARKETING_SCRIPT_SCORED = "marketing_script.scored"
    MARKETING_SCRIPT_APPROVED = "marketing_script.approved"
    STORYBOARD_GENERATED = "storyboard.generated"
    VIDEO_JOB_SUBMITTED = "video_job.submitted"
    VIDEO_JOB_SUCCEEDED = "video_job.succeeded"
    VIDEO_JOB_FAILED = "video_job.failed"
    VIDEO_OUTPUT_ASSEMBLED = "video_output.assembled"
    QUALITY_REPORT_COMPLETED = "quality_report.completed"

    WORKFLOW_RUN_STARTED = "workflow_run.started"
    WORKFLOW_RUN_FINISHED = "workflow_run.finished"
    TASK_RUN_FAILED = "task_run.failed"
    REVIEW_REQUESTED = "review.requested"
    REVIEW_DECIDED = "review.decided"

    @property
    def subject_type(self) -> SubjectType:
        return SubjectType(self.split(".", 1)[0])


class EventSubject(BaseModel):
    """事件主体（设计文档 8.2 的 `subject`）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: SubjectType
    id: str = Field(min_length=1, max_length=64)
    version: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _artifact_subjects_must_be_versioned(self) -> Self:
        """产物类主体必须带版本号。

        没有版本号的产物事件会让「某条成片由哪个分析版本产生」无从回答，
        而那正是设计文档 3.5 要求系统必须能回答的问题。
        """
        is_artifact = self.type not in _NON_ARTIFACT_SUBJECTS
        if is_artifact and self.version is None:
            raise ValueError(f"产物类主体 {self.type} 必须带 version")
        if not is_artifact and self.version is not None:
            raise ValueError(f"非产物主体 {self.type} 不应带 version")
        return self


class EventEnvelope(BaseModel):
    """统一事件信封。字段与设计文档 8.2 严格一致，顶层封闭。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1, max_length=64)
    event_type: EventType
    event_version: str = Field(default=CURRENT_EVENT_VERSION)
    occurred_at: datetime
    workflow_run_id: str | None = None
    task_run_id: str | None = None
    correlation_id: str = Field(min_length=1, max_length=64)
    producer: str = Field(min_length=1, max_length=64)
    subject: EventSubject
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_version")
    @classmethod
    def _version_is_wellformed(cls, value: str) -> str:
        EventVersion.parse(value)
        return value

    @field_validator("occurred_at")
    @classmethod
    def _occurred_at_is_aware(cls, value: datetime) -> datetime:
        """拒绝无时区时间。

        跨时区排查「这条成片是什么时候生成的」时，naive datetime 会让时间线
        对不上，而且这种错一旦入库就无法事后修复。
        """
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("occurred_at 必须带时区")
        return value

    @model_validator(mode="after")
    def _subject_matches_event_type(self) -> Self:
        """事件类型与主体类型必须自洽。

        `marketing_script.generated` 带一个 `storyboard` 主体在语法上完全合法，
        但消费方会按事件类型去解读主体，错配会静默产出错误的血缘。
        """
        expected = self.event_type.subject_type
        if self.subject.type is not expected:
            raise ValueError(
                f"事件 {self.event_type} 的主体类型应为 {expected}，实际为 {self.subject.type}"
            )
        return self

    def version(self) -> EventVersion:
        return EventVersion.parse(self.event_version)

    def is_compatible_with(self, consumer_version: str) -> bool:
        return self.version().is_compatible_with(EventVersion.parse(consumer_version))
