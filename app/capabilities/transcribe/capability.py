"""语音转写能力（设计文档 6.2）。

独立于预处理，消费预处理产出的音轨资产。

**为什么不合进预处理。** 预处理是纯本地、零费用、可无限重跑的；转写则可能走云
服务、按时长计费。合在一起会让「换灵敏度重切一次镜头」也顺带重跑一次转写，
而重跑要花钱的话就没人敢重跑——设计文档 3.1 要求独立重跑，那条要求只有在拆开
之后才真的成立。

拆开的另一个收益：转写换模型时不需要重跑抽帧与切分，反之亦然。
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.domain.assets import AssetStore
from app.domain.capability import Capability, CapabilityRequest, CapabilityResult
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.refs import ArtifactType
from app.domain.transcript import SpeechRecognizer, Transcript
from app.domain.versioning import ArtifactRepository, ArtifactStatus


class TranscribeParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(default=None, max_length=16)
    """留空则自动识别。明确指定能提升准确率并避免中英混说时误判语种。"""


class TranscribeCapability(Capability[TranscribeParameters]):
    name: ClassVar[str] = "audio_transcribe"
    stage: ClassVar[str] = "analysis"
    summary: ClassVar[str] = "把音轨转写为带时间戳的句子序列"
    accepts: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.PREPROCESS_RESULT,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.TRANSCRIPT,)
    parameters_model = TranscribeParameters
    # ASR 不经过大模型对话接口，因此不需要提示词。
    required_prompts: ClassVar[tuple[()]] = ()

    def __init__(
        self,
        *,
        artifacts: ArtifactRepository,
        assets: AssetStore,
        recognizer: SpeechRecognizer,
    ) -> None:
        self._artifacts = artifacts
        self._assets = assets
        self._recognizer = recognizer

    async def run(
        self, request: CapabilityRequest, params: TranscribeParameters
    ) -> CapabilityResult:
        # 转写是重 CPU 任务，一条 60 秒音频在本地模型上可达数十秒。
        # 留在事件循环里会让整个进程在此期间无法响应。
        return await asyncio.to_thread(self._transcribe, request, params)

    def _transcribe(
        self, request: CapabilityRequest, params: TranscribeParameters
    ) -> CapabilityResult:
        source_ref = next(
            (r for r in request.input_refs if r.type is ArtifactType.PREPROCESS_RESULT), None
        )
        if source_ref is None:
            raise CapabilityError(
                ErrorCode.INVALID_INPUT_REFS, "转写需要一个 preprocess_result 输入"
            )

        preprocess = self._artifacts.get(source_ref)
        preprocess.require_usable()

        audio_asset_id = preprocess.body.get("audio_asset_id")
        if not audio_asset_id:
            # 「该视频无音轨」和「转写失败」必须区分开：前者是素材事实，后者是故障。
            # 把无音轨当成空转写会让视频理解把技术情况误读成创作手法。
            raise CapabilityError(
                ErrorCode.INVALID_INPUT_REFS,
                f"{source_ref} 没有音轨资产，该视频可能无声轨或预处理时关闭了抽取",
            )

        asset = self._assets.get(str(audio_asset_id))
        asset.require_usable()

        lines, language, model_name = self._recognizer.transcribe(
            self._assets.open_path(str(audio_asset_id)), language=params.language
        )

        transcript = Transcript(
            language=language,
            lines=lines,
            model_name=model_name,
            audio_asset_id=str(audio_asset_id),
        )

        version = self._artifacts.create_version(
            ArtifactType.TRANSCRIPT,
            f"tr_{source_ref.id}",
            transcript.model_dump(mode="json"),
            created_by=self.name,
            sources=(source_ref,),
            # 转写是机器产物，可直接作为下游输入；人工修订会产生新版本。
            status=ArtifactStatus.READY,
        )

        return CapabilityResult(
            output_refs=(version.ref,),
            metrics={
                "line_count": float(len(lines)),
                "speech_ms": float(transcript.speech_ms),
                "character_count": float(len(transcript.full_text)),
            },
            notes=(
                f"{model_name} 识别语言 {language}，"
                f"{len(lines)} 句 / {len(transcript.full_text)} 字",
            ),
        )
