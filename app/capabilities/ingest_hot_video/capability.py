"""热点视频入库（设计文档 6.1）。

把一个本地视频文件连同来源、授权口径和入库理由，登记成一条 `HotVideo` 产物。

校验顺序是刻意的：**先查授权口径，再探测文件，最后才建资产。** 反过来会在拒绝
一条视频之前就把它复制进了原始桶，留下一个没有任何业务记录指向的孤儿资产，
而孤儿资产在回收策略里也找不到删除依据。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.domain.assets import AssetKind, AssetOrigin, AssetStore, RightsStatus
from app.domain.capability import Capability, CapabilityRequest, CapabilityResult
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.hot_video import (
    HotVideoBody,
    MetricsSnapshot,
    SourcePlatform,
    logical_id_for,
)
from app.domain.media import MediaProbe
from app.domain.refs import ArtifactType
from app.domain.versioning import ArtifactRepository, ArtifactStatus


class IngestHotVideoParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1, max_length=255)
    source_platform: SourcePlatform
    rights_status: RightsStatus
    registered_by: str = Field(min_length=1, max_length=64)
    selection_reason: str = Field(min_length=4, max_length=500)
    source_url: str | None = None
    author: str | None = None
    tags: tuple[str, ...] = ()
    metrics_snapshot: MetricsSnapshot | None = None
    mime_type: str = "video/mp4"


class IngestHotVideoCapability(Capability[IngestHotVideoParameters]):
    name: ClassVar[str] = "hot_video_ingest"
    stage: ClassVar[str] = "analysis"
    summary: ClassVar[str] = "登记热点视频，记录来源、授权口径与运营的入库判断"
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.HOT_VIDEO,)
    parameters_model = IngestHotVideoParameters
    required_prompts: ClassVar[tuple[()]] = ()

    def __init__(
        self,
        *,
        artifacts: ArtifactRepository,
        assets: AssetStore,
        probe: MediaProbe,
    ) -> None:
        self._artifacts = artifacts
        self._assets = assets
        self._probe = probe

    async def run(
        self, request: CapabilityRequest, params: IngestHotVideoParameters
    ) -> CapabilityResult:
        # ffprobe 与文件拷贝都是阻塞调用，一条视频可能耗时数秒。留在事件循环里
        # 会让整个 API 进程在这段时间内无法响应任何其他请求。
        return await asyncio.to_thread(self._ingest, params)

    def _ingest(self, params: IngestHotVideoParameters) -> CapabilityResult:
        if params.rights_status is RightsStatus.UNKNOWN:
            raise CapabilityError(
                ErrorCode.ARTIFACT_RIGHTS_UNCLEARED,
                "入库必须标注授权口径，未标注的素材无法进入生产（设计文档 13.3）",
            )

        path = Path(params.file_path)
        if not path.is_file():
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, f"文件不存在：{path}")

        metadata = self._probe.probe(path)
        if metadata.duration_ms <= 0:
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "视频时长为 0，拒绝入库")

        asset = self._assets.put(
            path,
            kind=AssetKind.VIDEO,
            origin=AssetOrigin.EXTERNAL_REFERENCE,
            mime_type=params.mime_type,
            created_by=params.registered_by,
            rights_status=params.rights_status,
            source_url=params.source_url,
        )

        body = HotVideoBody(
            source_platform=params.source_platform,
            source_url=params.source_url,
            author=params.author,
            asset_id=asset.id,
            original_filename=params.original_filename,
            metrics_snapshot=params.metrics_snapshot,
            tags=params.tags,
            rights_status=params.rights_status,
            registered_by=params.registered_by,
            selection_reason=params.selection_reason,
        )

        # 同内容视频落到同一逻辑 id，重复录入自然成为新版本——
        # 「谁又录了一次、理由是什么」保留在版本历史里。
        version = self._artifacts.create_version(
            ArtifactType.HOT_VIDEO,
            logical_id_for(asset.sha256),
            body.model_dump(mode="json"),
            created_by=params.registered_by,
            status=ArtifactStatus.READY,
        )

        notes = [
            f"{metadata.width}x{metadata.height} {metadata.aspect_ratio} "
            f"{metadata.duration_ms}ms，{'竖屏' if metadata.is_vertical else '横屏'}"
        ]
        if version.version > 1:
            notes.append(f"该内容此前已入库，本次为第 {version.version} 次录入")

        return CapabilityResult(
            output_refs=(version.ref,),
            metrics={
                "duration_ms": float(metadata.duration_ms),
                "ingest_version": float(version.version),
            },
            notes=tuple(notes),
        )
