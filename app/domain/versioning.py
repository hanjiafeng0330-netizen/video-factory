"""产物的「逻辑实体 + 不可变版本」模型（设计文档 11.2）。

设计文档要求「所有 AI 产物采用逻辑实体 + 不可变版本的方式保存。新修改产生新
版本，不覆盖旧版本」，并且（9.2）「上游版本发生变化时，系统标记下游产物为
可能过期，不直接覆盖」。

这里把这两条做成仓储层的不变量，而不是各业务模块的自觉：

- `create_version()` 只会追加，没有任何接口能改已有版本的 `body`；
- 新版本落地时自动把引用旧版本的下游**递归**标记为可能过期；
- 过期产物默认不可用于生产，需要重新审核而不是被静默沿用。

「不覆盖」这条如果只靠约定，第一次赶进度做「就地修一下脚本」时就会破掉，
而破掉之后所有历史成片的回溯链同时失效，且无法事后补回。
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.domain.errors import CapabilityError, ErrorCode
from app.domain.refs import ArtifactRef, ArtifactType


class ArtifactStatus(StrEnum):
    """产物版本的生命周期状态。

    设计文档里各实体的状态字段写法不一（`ready` / `reviewed` / `approved` /
    `active`），但语义可归为同一条链：草稿 → 可用 → 人工批准 / 驳回。统一成
    一套状态是为了让审核中心（T2-4）不必为每种产物写一遍状态判断。
    """

    DRAFT = "draft"
    """生成中或未定稿，不可用于生产。"""

    READY = "ready"
    """自动流程已完成，可作为下游输入，但尚未经人工审核。"""

    APPROVED = "approved"
    """人工审核通过。"""

    REJECTED = "rejected"
    """人工驳回。终态。"""


_ALLOWED_TRANSITIONS: Mapping[ArtifactStatus, frozenset[ArtifactStatus]] = {
    ArtifactStatus.DRAFT: frozenset({ArtifactStatus.READY, ArtifactStatus.REJECTED}),
    ArtifactStatus.READY: frozenset({ArtifactStatus.APPROVED, ArtifactStatus.REJECTED}),
    # 已批准的产物事后发现问题仍可驳回——合规问题往往是发布后才暴露的。
    ArtifactStatus.APPROVED: frozenset({ArtifactStatus.REJECTED}),
    # 驳回是终态：翻案必须产生新版本，不能原地改状态。否则「这一版曾被驳回」
    # 这个事实会消失，而它恰恰是审核记录里最需要留存的部分。
    ArtifactStatus.REJECTED: frozenset(),
}

_USABLE_STATUSES = frozenset({ArtifactStatus.READY, ArtifactStatus.APPROVED})


class StaleMark(BaseModel):
    """「可能过期」标记。

    刻意叫「可能」而不是「已失效」：上游改了不代表下游一定错了，需要人来判断。
    因此这里只记录事实与触发源，不做自动作废（设计文档 9.2）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    marked_at: datetime
    reason: str = Field(min_length=1, max_length=500)
    triggered_by: ArtifactRef
    """导致过期的那个上游新版本。用于回答「为什么这条脚本要重做」。"""


class ArtifactVersion(BaseModel):
    """某个逻辑产物的某一个版本。

    `body` 用 JSON 而非具体类型：一个仓储要存所有产物类型，泛型化会让仓储接口
    被迫按类型拆开。类型化访问由 `body_as()` 在读取点完成。
    """

    model_config = ConfigDict(extra="forbid")

    type: ArtifactType
    id: str = Field(min_length=1, max_length=64)
    version: Annotated[int, Field(ge=1)]
    status: ArtifactStatus
    body: Mapping[str, Any]
    sources: tuple[ArtifactRef, ...] = ()
    """直接上游。逐条引用确切版本，是回溯与过期传播的依据。"""

    stale: StaleMark | None = None
    created_at: datetime
    created_by: str = Field(min_length=1, max_length=64)

    @property
    def ref(self) -> ArtifactRef:
        return ArtifactRef(type=self.type, id=self.id, version=self.version)

    @property
    def is_usable(self) -> bool:
        """能否作为下游生产的输入。"""
        return self.status in _USABLE_STATUSES and self.stale is None

    def body_as[T: BaseModel](self, model: builtins.type[T]) -> T:
        return TypeAdapter(model).validate_python(self.body)

    def require_usable(self) -> None:
        """在把该版本用于生产前调用。

        过期或未定稿的产物被静默沿用，会产出「看起来正常但依据已经变了」的成片，
        这类问题在质检阶段发现不了。
        """
        if self.stale is not None:
            raise CapabilityError(
                ErrorCode.ARTIFACT_VERSION_STALE,
                f"{self.ref} 已被标记为可能过期（{self.stale.reason}），需重新确认后才能用于生产",
            )
        if self.status not in _USABLE_STATUSES:
            raise CapabilityError(
                ErrorCode.ARTIFACT_STATUS_TRANSITION_INVALID,
                f"{self.ref} 当前状态为 {self.status}，不可用于生产",
            )


def validate_transition(current: ArtifactStatus, target: ArtifactStatus) -> None:
    if target not in _ALLOWED_TRANSITIONS[current]:
        targets = sorted(_ALLOWED_TRANSITIONS[current])
        allowed = ", ".join(targets) if targets else "（终态，无可用跃迁）"
        raise CapabilityError(
            ErrorCode.ARTIFACT_STATUS_TRANSITION_INVALID,
            f"状态不能从 {current} 变为 {target}，允许的目标为 {allowed}",
        )


class ArtifactRepository(Protocol):
    """产物仓储协议。

    协议放在 domain、实现放在 `app.storage`，是为了让能力模块只依赖协议而不
    依赖具体存储——这样它既满足「能力模块不得 import app.storage」的边界契约，
    也让同一套契约测试能同时约束内存实现与 PostgreSQL 实现。
    """

    def create_version(
        self,
        artifact_type: ArtifactType,
        artifact_id: str,
        body: Mapping[str, Any],
        *,
        created_by: str,
        sources: tuple[ArtifactRef, ...] = (),
        status: ArtifactStatus = ArtifactStatus.DRAFT,
    ) -> ArtifactVersion:
        """追加一个新版本。版本号由仓储分配，调用方无法指定。

        实现必须保证：`body` 不覆盖任何已有版本；引用的 `sources` 必须已存在；
        落地后自动传播过期标记。
        """
        ...

    def get(self, ref: ArtifactRef) -> ArtifactVersion: ...

    def latest(self, artifact_type: ArtifactType, artifact_id: str) -> ArtifactVersion: ...

    def history(self, artifact_type: ArtifactType, artifact_id: str) -> tuple[ArtifactVersion, ...]:
        """按版本号升序返回全部版本。旧版本永远可查（设计文档 11.2）。"""
        ...

    def transition(self, ref: ArtifactRef, target: ArtifactStatus) -> ArtifactVersion: ...

    def mark_stale(self, ref: ArtifactRef, mark: StaleMark) -> ArtifactVersion: ...

    def dependents(self, ref: ArtifactRef) -> tuple[ArtifactRef, ...]:
        """直接引用了该版本的下游产物。"""
        ...

    def list_by_type(self, artifact_type: ArtifactType) -> tuple[ArtifactVersion, ...]:
        """列出某类所有版本，供热点库等只读索引使用。"""
        ...


def propagate_staleness(
    repo: ArtifactRepository, superseding: ArtifactVersion, *, now: datetime
) -> tuple[ArtifactRef, ...]:
    """新版本落地后，把引用该产物旧版本的下游**递归**标记为可能过期。

    递归是必需的：热点分析出了新版本，不只是直接引用它的脚本模板要重新确认，
    模板下游的营销脚本、分镜、成片同样都建立在旧依据上。只标记一层等于把问题
    藏在第二层。

    写成独立的领域函数而不是各实现内部的私有逻辑，是为了让内存实现与
    PostgreSQL 实现共用同一份传播规则；契约测试会验证每个实现都调用了它。
    """
    older = tuple(
        version.ref
        for version in repo.history(superseding.type, superseding.id)
        if version.version < superseding.version
    )
    if not older:
        return ()

    reason = f"上游 {superseding.type}:{superseding.id} 产生了新版本 v{superseding.version}"
    marked: list[ArtifactRef] = []
    visited: set[ArtifactRef] = set()
    frontier: list[ArtifactRef] = list(older)

    while frontier:
        current = frontier.pop()
        for dependent in repo.dependents(current):
            if dependent in visited:
                continue
            visited.add(dependent)
            repo.mark_stale(
                dependent,
                StaleMark(marked_at=now, reason=reason, triggered_by=superseding.ref),
            )
            marked.append(dependent)
            frontier.append(dependent)

    return tuple(marked)
