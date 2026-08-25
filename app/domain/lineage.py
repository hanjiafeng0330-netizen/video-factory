"""血缘（设计文档 11.2、3.5）。

设计文档 3.5 要求系统必须能回答：

> 某条成片由哪个热点视频、哪个分析版本、哪个脚本模板、哪个产品版本、哪个提示词、
> 哪个模型及哪些生成参数产生？

关键设计决定：**血缘由框架在产物落地时自动写入，业务代码没有「记得写血缘」这个
义务。** 理由是血缘的价值全在完整性——只要有一处忘写，那条链就断了，而断点只有
在几个月后回溯某条成片时才会暴露，此时已无法补回。

因此 `LineageRecorder` 挂在仓储上，`create_version()` 落地即记录；能力模块甚至
看不到这个接口。契约测试直接验证「没有任何业务代码调用记录接口，血缘依然完整」。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.domain.refs import ArtifactRef


class RelationType(StrEnum):
    """血缘关系类型。对应设计文档 7.1 实体关系图里的动词。"""

    PRODUCES = "produces"
    """热点视频 → 视频分析。"""

    ABSTRACTS = "abstracts"
    """视频分析 → 脚本模板。"""

    GUIDES = "guides"
    """脚本模板 → 营销脚本。"""

    USED_BY = "used_by"
    """产品档案 → 营销脚本。"""

    CONVERTED_TO = "converted_to"
    """营销脚本 → 分镜。"""

    GENERATES = "generates"
    """分镜 → 视频任务。"""

    CHECKED_BY = "checked_by"
    """成片 → 质检报告。"""

    DERIVED_FROM = "derived_from"
    """通用派生关系。类型组合未被上面枚举覆盖时的兜底。"""


# 按 (上游类型, 下游类型) 推断关系类型。推断而不是让调用方传，是因为调用方传错
# 不会有任何症状——血缘看起来是完整的，只是关系名是错的，而这种错无法事后校验。
_RELATION_BY_PAIR: dict[tuple[str, str], RelationType] = {
    ("hot_video", "preprocess_result"): RelationType.PRODUCES,
    ("preprocess_result", "video_analysis"): RelationType.PRODUCES,
    ("hot_video", "video_analysis"): RelationType.PRODUCES,
    ("video_analysis", "script_pattern"): RelationType.ABSTRACTS,
    ("script_pattern", "marketing_script"): RelationType.GUIDES,
    ("product_profile", "marketing_script"): RelationType.USED_BY,
    ("selling_point_set", "marketing_script"): RelationType.USED_BY,
    ("marketing_script", "storyboard"): RelationType.CONVERTED_TO,
    ("storyboard", "video_job"): RelationType.GENERATES,
    ("video_job", "video_output"): RelationType.PRODUCES,
    ("video_output", "quality_report"): RelationType.CHECKED_BY,
}


def infer_relation(source: ArtifactRef, target: ArtifactRef) -> RelationType:
    return _RELATION_BY_PAIR.get((source.type.value, target.type.value), RelationType.DERIVED_FROM)


class ArtifactRelation(BaseModel):
    """一条血缘记录。字段对齐设计文档 11.2 的 `artifact_relation` 表。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: ArtifactRef
    target: ArtifactRef
    relation_type: RelationType
    workflow_run_id: str | None = None
    created_at: datetime


class TraceNode(BaseModel):
    """回溯树的一个节点。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref: ArtifactRef
    relation_from_child: RelationType | None = None
    """本节点与它的下游（树中的父节点）之间的关系。根节点为空。"""

    depth: int = Field(ge=0)


class LineageRecorder(Protocol):
    """血缘写入。由仓储在产物落地时调用，业务代码不接触。"""

    def record(self, relation: ArtifactRelation) -> None: ...


class LineageRepository(Protocol):
    def relations_into(self, target: ArtifactRef) -> tuple[ArtifactRelation, ...]:
        """指向该产物的全部上游关系。"""
        ...

    def relations_out_of(self, source: ArtifactRef) -> tuple[ArtifactRelation, ...]: ...

    def trace(self, ref: ArtifactRef) -> tuple[TraceNode, ...]:
        """回溯该产物的完整祖先树，广度优先、按深度升序。

        这是设计文档 3.5 那个问题的直接答案，也是 T6-4「一键导出回溯报告」的
        数据来源。
        """
        ...


class LineageStore(LineageRecorder, LineageRepository, Protocol):
    """同时具备写入与查询的血缘实现。

    写入与查询共享同一份索引，拆成两个对象会立刻带来「两份索引如何保持一致」的
    问题，而这里没有任何需要拆的理由。协议仍然分开声明，是为了让只读的调用点
    在类型上就拿不到写入能力。
    """
