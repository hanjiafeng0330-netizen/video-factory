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
from app.domain.refs import ArtifactRef, ArtifactType
from app.domain.versioning import (
    ArtifactStatus,
    ArtifactVersion,
    StaleMark,
    propagate_staleness,
    validate_transition,
)


class InMemoryArtifactRepository:
    def __init__(self) -> None:
        self._versions: dict[tuple[ArtifactType, str], list[ArtifactVersion]] = defaultdict(list)
        self._dependents: dict[ArtifactRef, list[ArtifactRef]] = defaultdict(list)

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
