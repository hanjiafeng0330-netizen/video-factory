"""产物仓储的内存实现。

它不是「测试替身」，而是与 PostgreSQL 实现同等地位的一个实现：两者由
`tests/contract/test_artifact_repository.py` 里同一套契约测试约束。这样做的
收益是快测试不需要数据库，而代价被契约测试兜住——任何一方少实现一条不变量，
测试就会红。

用途：本地开发、dev 环境预览、以及编排器的干跑（dry run）。生产使用
PostgreSQL 实现。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from app.domain.errors import CapabilityError, ErrorCode
from app.domain.lineage import ArtifactRelation, LineageRecorder, infer_relation
from app.domain.refs import ArtifactRef, ArtifactType
from app.domain.versioning import (
    ArtifactStatus,
    ArtifactVersion,
    StaleMark,
    propagate_staleness,
    validate_transition,
)


class InMemoryArtifactRepository:
    def __init__(self, lineage: LineageRecorder | None = None) -> None:
        self._versions: dict[tuple[ArtifactType, str], list[ArtifactVersion]] = defaultdict(list)
        self._dependents: dict[ArtifactRef, list[ArtifactRef]] = defaultdict(list)
        self._lineage = lineage
        self._workflow_run_id: str | None = None

    def bind_workflow_run(self, workflow_run_id: str | None) -> None:
        """把后续写入归属到某次流程运行。

        编排器在启动一次运行时设置它，血缘记录随之带上 `workflow_run_id`。
        做成仓储状态而不是 `create_version()` 的参数，是为了让能力模块无需感知
        自己是否运行在流程里——独立调用与流程内调用的能力代码完全一致
        （设计文档 5.2）。
        """
        self._workflow_run_id = workflow_run_id

    # ------------------------------------------------------------------ 写

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
        for source in sources:
            # 引用不存在的上游会让血缘出现断点，而断点要等到回溯时才发现。
            self.get(source)

        chain = self._versions[(artifact_type, artifact_id)]
        version = ArtifactVersion(
            type=artifact_type,
            id=artifact_id,
            version=len(chain) + 1,
            status=status,
            # 深拷贝：调用方之后修改传入的 dict 不应该改到已落地的版本，
            # 那等于绕过「内容不可变」。
            body=deepcopy(dict(body)),
            sources=sources,
            created_at=datetime.now(UTC),
            created_by=created_by,
        )
        chain.append(version)

        for source in sources:
            self._dependents[source].append(version.ref)

        # 血缘在这里写，而不是让每个能力模块自己记得写。血缘的价值全在完整性：
        # 只要有一处忘写那条链就断了，而断点只有在几个月后回溯某条成片时才会
        # 暴露，此时已无法补回。
        if self._lineage is not None:
            now = datetime.now(UTC)
            for source in sources:
                self._lineage.record(
                    ArtifactRelation(
                        source=source,
                        target=version.ref,
                        relation_type=infer_relation(source, version.ref),
                        workflow_run_id=self._workflow_run_id,
                        created_at=now,
                    )
                )

        propagate_staleness(self, version, now=datetime.now(UTC))
        return version

    def transition(self, ref: ArtifactRef, target: ArtifactStatus) -> ArtifactVersion:
        version = self.get(ref)
        validate_transition(version.status, target)
        if target is ArtifactStatus.APPROVED and version.stale is not None:
            raise CapabilityError(
                ErrorCode.ARTIFACT_VERSION_STALE,
                f"{ref} 已被标记为可能过期，必须重新确认依据后才能批准",
            )
        version.status = target
        return version

    def mark_stale(self, ref: ArtifactRef, mark: StaleMark) -> ArtifactVersion:
        version = self.get(ref)
        # 保留最早的标记：原始触发源是排查起点，被后续标记覆盖后就查不到
        # 「最先是哪次上游改动导致这条链需要重做」。
        if version.stale is None:
            version.stale = mark
        return version

    # ------------------------------------------------------------------ 读

    def get(self, ref: ArtifactRef) -> ArtifactVersion:
        chain = self._versions.get((ref.type, ref.id), [])
        if not 1 <= ref.version <= len(chain):
            raise CapabilityError(ErrorCode.ARTIFACT_NOT_FOUND, f"产物版本不存在：{ref}")
        return chain[ref.version - 1]

    def latest(self, artifact_type: ArtifactType, artifact_id: str) -> ArtifactVersion:
        chain = self._versions.get((artifact_type, artifact_id), [])
        if not chain:
            raise CapabilityError(
                ErrorCode.ARTIFACT_NOT_FOUND,
                f"产物不存在：{artifact_type}:{artifact_id}",
            )
        return chain[-1]

    def history(self, artifact_type: ArtifactType, artifact_id: str) -> tuple[ArtifactVersion, ...]:
        return tuple(self._versions.get((artifact_type, artifact_id), []))

    def dependents(self, ref: ArtifactRef) -> tuple[ArtifactRef, ...]:
        return tuple(self._dependents.get(ref, ()))

    def list_by_type(self, artifact_type: ArtifactType) -> tuple[ArtifactVersion, ...]:
        versions: list[ArtifactVersion] = []
        for (stored_type, _), chain in self._versions.items():
            if stored_type is artifact_type:
                versions.extend(chain)
        return tuple(sorted(versions, key=lambda item: item.created_at, reverse=True))
