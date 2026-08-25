"""视频预处理能力（设计文档 6.2）。

输入一条已入库的热点视频，输出：媒体元数据、音轨资产、镜头列表、按镜头抽取的
关键帧资产。这是「拆分视频」的落点。

不做的事，以及为什么：

- **不做语音转写与 OCR。** 它们依赖外部服务（ASR / OCR 供应商），而本能力的
  价值是纯本地、零费用、可无限重跑。把它们塞进来会让「重跑一次预处理」变成
  一件要花钱的事，而设计文档 3.1 明确要求独立重跑。转写与 OCR 作为独立能力接在
  后面，各自消费本能力产出的音轨与关键帧资产。
- **不做语义标注。** 「这一段是钩子还是证明」属于视频理解（6.3）。混在一起会让
  「切分错了」和「理解错了」两类问题无法分辨。

产出落成一个 `PREPROCESS_RESULT` 产物版本，血缘由仓储自动记录到热点视频上。
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.domain.assets import AssetKind, AssetOrigin, AssetStore
from app.domain.capability import Capability, CapabilityRequest, CapabilityResult
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.media import (
    AudioExtractor,
    FrameExtractor,
    MediaMetadata,
    MediaProbe,
    ShotDetector,
    ShotList,
)
from app.domain.refs import ArtifactType
from app.domain.versioning import ArtifactRepository, ArtifactStatus


class PreprocessParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shot_sensitivity: float = Field(default=0.5, ge=0.05, le=0.95)
    """镜头切分灵敏度，越大切得越碎。0.5 对应检测库的默认值，是已验证可用的点。"""

    frames_per_shot: int = Field(default=3, ge=1, le=10)
    """每个镜头抽几帧用于视觉理解。3 帧（首/中/尾）是默认，能覆盖动作起止。
    详细策略见 docs/镜头切分策略.md 第 6 节。"""

    extract_audio: bool = True
    extract_keyframes: bool = True
    max_shots_to_sample: int = Field(default=40, ge=1, le=200)
    """最多抽多少镜头的关键帧。超过则截断并置 truncated_keyframes=True。"""


class PreprocessResult(BaseModel):
    """预处理产物的 body 结构。"""

    model_config = ConfigDict(extra="forbid")

    metadata: MediaMetadata
    shots: ShotList
    audio_asset_id: str | None
    keyframes: tuple[tuple[str, ...], ...]
    """每镜头的关键帧资产 id。`keyframes[i]` 是第 i 个镜头的 N 帧（按时间排序）。"""
    keyframe_timestamps: tuple[tuple[int, ...], ...]
    """与 keyframes 平行：`keyframe_timestamps[i][j]` 是第 i 镜头第 j 帧的时间。"""
    frames_per_shot: int
    """实际每镜头抽的帧数。`frames_per_shot` 字段让下游能校验「我拿到的帧数就是抽帧
    参数要求的那个数」，而不是靠猜。"""
    truncated_shots: bool
    """镜头数超过 max_shots_to_sample 时为真。"""


class PreprocessVideoCapability(Capability[PreprocessParameters]):
    name: ClassVar[str] = "video_preprocess"
    stage: ClassVar[str] = "analysis"
    summary: ClassVar[str] = "抽取媒体元数据、音轨、镜头边界与关键帧"
    accepts: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.HOT_VIDEO,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.PREPROCESS_RESULT,)
    parameters_model = PreprocessParameters
    # 纯本地处理，不调用任何大模型，因此不需要提示词。
    required_prompts: ClassVar[tuple[()]] = ()

    def __init__(
        self,
        *,
        artifacts: ArtifactRepository,
        assets: AssetStore,
        probe: MediaProbe,
        audio: AudioExtractor,
        shots: ShotDetector,
        frames: FrameExtractor,
    ) -> None:
        # 依赖全部注入：能力模块只认 domain 协议，不 import app.storage 或
        # app.adapters 的具体实现。这样它既满足边界契约，也能在测试里用假实现驱动。
        self._artifacts = artifacts
        self._assets = assets
        self._probe = probe
        self._audio = audio
        self._shots = shots
        self._frames = frames

    async def run(
        self, request: CapabilityRequest, params: PreprocessParameters
    ) -> CapabilityResult:
        # 切分、抽帧、抽音轨全是阻塞的重活，一条 60 秒视频可达数十秒。
        # 必须离开事件循环，否则单条视频预处理期间整个进程无法响应。
        return await asyncio.to_thread(self._preprocess, request, params)

    def _preprocess(
        self, request: CapabilityRequest, params: PreprocessParameters
    ) -> CapabilityResult:
        hot_ref = next(
            (ref for ref in request.input_refs if ref.type is ArtifactType.HOT_VIDEO), None
        )
        if hot_ref is None:
            raise CapabilityError(ErrorCode.INVALID_INPUT_REFS, "预处理需要一个 hot_video 输入")

        hot = self._artifacts.get(hot_ref)
        hot.require_usable()
        asset_id = str(hot.body.get("asset_id") or "")
        if not asset_id:
            raise CapabilityError(ErrorCode.INVALID_INPUT_REFS, f"{hot_ref} 未关联视频资产")

        source_asset = self._assets.get(asset_id)
        # 版权口径未标注的素材连分析都不允许（设计文档 13.3）。
        source_asset.require_usable()
        source = self._assets.open_path(asset_id)

        metadata = self._probe.probe(source)
        shot_list = self._shots.detect_shots(
            source, metadata=metadata, sensitivity=params.shot_sensitivity
        )

        with tempfile.TemporaryDirectory(prefix="vf_preprocess_") as workdir:
            work = Path(workdir)
            audio_asset_id = self._maybe_extract_audio(
                source, work, asset_id, metadata, enabled=params.extract_audio
            )
            keyframes, keyframe_times, frames_per_shot, truncated = self._maybe_extract_keyframes(
                source, work, asset_id, shot_list, params
            )

        result = PreprocessResult(
            metadata=metadata,
            shots=shot_list,
            audio_asset_id=audio_asset_id,
            keyframes=keyframes,
            keyframe_timestamps=keyframe_times,
            frames_per_shot=frames_per_shot,
            truncated_shots=truncated,
        )

        version = self._artifacts.create_version(
            ArtifactType.PREPROCESS_RESULT,
            f"pp_{hot_ref.id}",
            result.model_dump(mode="json"),
            created_by=self.name,
            sources=(hot_ref,),
            # 预处理是确定性的机器产物，无需人工审核即可作为下游输入。
            status=ArtifactStatus.READY,
        )

        total_keyframes = sum(len(shot_frames) for shot_frames in keyframes)
        return CapabilityResult(
            output_refs=(version.ref,),
            metrics={
                "shot_count": float(shot_list.count),
                "duration_ms": float(metadata.duration_ms),
                "keyframe_count": float(total_keyframes),
                "frames_per_shot": float(frames_per_shot),
            },
            notes=(
                f"{metadata.width}x{metadata.height} {metadata.aspect_ratio} "
                f"{metadata.duration_ms}ms，{shot_list.count} 个镜头 × {frames_per_shot} 帧/镜头",
            ),
        )

    def _maybe_extract_audio(
        self,
        source: Path,
        work: Path,
        source_asset_id: str,
        metadata: MediaMetadata,
        *,
        enabled: bool,
    ) -> str | None:
        if not enabled or not metadata.has_audio:
            return None
        audio_path = self._audio.extract_audio(source, work / "audio.wav")
        return self._assets.put(
            audio_path,
            kind=AssetKind.AUDIO,
            origin=AssetOrigin.DERIVED,
            mime_type="audio/wav",
            created_by=self.name,
            derived_from=source_asset_id,
        ).id

    def _maybe_extract_keyframes(
        self,
        source: Path,
        work: Path,
        source_asset_id: str,
        shot_list: ShotList,
        params: PreprocessParameters,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...], int, bool]:
        """每个镜头均匀抽 frames_per_shot 帧。

        返回 (每镜头帧资产 id, 每镜头时间戳, 实际帧数, 是否截断镜头采样)。
        """
        if not params.extract_keyframes:
            return (), (), params.frames_per_shot, False

        selected = shot_list.shots[: params.max_shots_to_sample]
        per_shot_ids: list[tuple[str, ...]] = []
        per_shot_times: list[tuple[int, ...]] = []

        for shot in selected:
            # 把镜头等分成 N 段，取每段中点。比「首/中/尾」好，因为首帧和尾帧
            # 常落在转场上，抽出来是黑帧或叠化的糊图。
            sample_times = tuple(
                shot.start_ms + (2 * i + 1) * shot.duration_ms // (2 * params.frames_per_shot)
                for i in range(params.frames_per_shot)
            )
            shot_asset_ids: list[str] = []
            for j, at_ms in enumerate(sample_times):
                frame_path = self._frames.extract_frame(
                    source,
                    work / f"frame_{shot.index:04d}_{j}.jpg",
                    at_ms=at_ms,
                )
                shot_asset_ids.append(
                    self._assets.put(
                        frame_path,
                        kind=AssetKind.IMAGE,
                        origin=AssetOrigin.DERIVED,
                        mime_type="image/jpeg",
                        created_by=self.name,
                        derived_from=source_asset_id,
                    ).id
                )
            per_shot_ids.append(tuple(shot_asset_ids))
            per_shot_times.append(sample_times)

        return (
            tuple(per_shot_ids),
            tuple(per_shot_times),
            params.frames_per_shot,
            shot_list.count > params.max_shots_to_sample,
        )
