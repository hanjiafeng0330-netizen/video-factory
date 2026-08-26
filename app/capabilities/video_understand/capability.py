"""视频理解能力（设计文档 6.3）。

把镜头脚本（多帧 + 台词）喂给视觉模型，输出每个镜头的画面描述。
这是 T3-3 的实现。

职责：
- 对每个镜头调用 vision 模型（多帧 + 台词 → 画面描述）
- 把视觉描述回填到镜头脚本的 visual_description 字段
- 输出完整的、带画面描述的镜头脚本作为产物
"""

from __future__ import annotations

import asyncio
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.adapters.llm import LLMClient
from app.domain.assets import AssetStore
from app.domain.capability import Capability, CapabilityRequest, CapabilityResult
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.media import ShotList
from app.domain.prompt_registry import PromptRegistry
from app.domain.prompts import PromptRequirement
from app.domain.refs import ArtifactType
from app.domain.shot_script import ShotScript, ShotScriptEntry, build_shot_script
from app.domain.versioning import ArtifactRepository, ArtifactStatus


class VideoUnderstandParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int = Field(default=800, ge=100, le=4000)
    """单个镜头画面描述的最大输出 token。"""


class VideoUnderstandCapability(Capability[VideoUnderstandParameters]):
    name: ClassVar[str] = "video_understand"
    stage: ClassVar[str] = "视频理解"
    summary: ClassVar[str] = "基于多帧关键帧描述每个镜头的画面内容与动作过程"
    accepts: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.PREPROCESS_RESULT,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.SHOT_SCRIPT,)
    parameters_model = VideoUnderstandParameters
    required_prompts: ClassVar[tuple[PromptRequirement, ...]] = (
        PromptRequirement(
            key="video_understand.shot_visual",
            purpose="基于多帧关键帧描述单个镜头的画面内容与动作过程",
        ),
    )

    def __init__(
        self,
        *,
        artifacts: ArtifactRepository,
        assets: AssetStore,
        llm: LLMClient,
        prompts: PromptRegistry,
        vision_model: str,
    ) -> None:
        self._artifacts = artifacts
        self._assets = assets
        self._llm = llm
        self._prompts = prompts
        self._vision_model = vision_model

    async def run(
        self, request: CapabilityRequest, params: VideoUnderstandParameters
    ) -> CapabilityResult:
        return await asyncio.to_thread(self._run_sync, request, params)

    def _run_sync(
        self, request: CapabilityRequest, params: VideoUnderstandParameters
    ) -> CapabilityResult:
        # 找到预处理产物
        preprocess_ref = next(
            (ref for ref in request.input_refs if ref.type == ArtifactType.PREPROCESS_RESULT),
            None,
        )
        if not preprocess_ref:
            raise CapabilityError(
                ErrorCode.INVALID_INPUT_REFS,
                "video_understand 需要 preprocess_result 作为输入",
            )

        preprocess = self._artifacts.get(preprocess_ref)
        preprocess_body = preprocess.body

        # 构建镜头脚本（无转写版本，后续可扩展）
        shot_script = build_shot_script(
            shots=ShotList.model_validate(preprocess_body["shots"]),
            keyframes=tuple(preprocess_body["keyframes"]),
            keyframe_timestamps=tuple(preprocess_body["keyframe_timestamps"]),
        )

        # 逐镜头调用视觉模型
        entries_with_descriptions: list[ShotScriptEntry] = []
        for entry in shot_script.entries:
            if not entry.keyframe_asset_ids:
                # 没有关键帧的镜头，跳过视觉描述
                entries_with_descriptions.append(entry)
                continue

            # 获取关键帧文件路径
            image_paths = [
                self._assets.open_path(asset_id)
                for asset_id in entry.keyframe_asset_ids
            ]

            # 构建用户提示词变量
            variables = {
                "shot_index": str(entry.index),
                "start_ms": str(entry.start_ms),
                "end_ms": str(entry.end_ms),
                "duration_ms": str(entry.duration_ms),
                "spoken_text": entry.text if entry.lines else "",
                "platform": "douyin",  # TODO: 从预处理产物或请求参数获取
            }

            # 从请求中获取提示词并渲染
            resolved_prompt = request.prompt("video_understand.shot_visual")
            # 重新渲染提示词，填入当前镜头的变量
            shot_visual_prompt = self._prompts.resolve(resolved_prompt.key, variables)

            # 调用视觉模型
            response = self._llm.vision(
                model=self._vision_model,
                system=shot_visual_prompt.text,
                user_text=f"请分析镜头 {entry.index}（{entry.start_ms}ms - {entry.end_ms}ms）",
                image_paths=tuple(image_paths),
                max_tokens=params.max_output_tokens,
            )

            # 解析响应（期望 JSON）
            visual_description = self._parse_visual_description(response.text)

            # 更新 entry
            updated_entry = entry.model_copy(update={"visual_description": visual_description})
            entries_with_descriptions.append(updated_entry)

        # 构建新的镜头脚本
        result_script = ShotScript(
            entries=tuple(entries_with_descriptions),
            has_transcript=False,  # video_understand 不包含转写
        )

        # 保存产物
        result = self._artifacts.create_version(
            artifact_type=ArtifactType.SHOT_SCRIPT,
            artifact_id=f"ss_{preprocess_ref.id}",
            body=result_script.model_dump(),
            created_by=f"capability:{self.name}",
            sources=(preprocess_ref,),
            status=ArtifactStatus.READY,
        )

        return CapabilityResult(
            output_refs=(result.ref,),
            metrics={
                "shot_count": len(shot_script.entries),
                "shots_with_visual_description": sum(
                    1 for e in entries_with_descriptions if e.visual_description
                ),
            },
        )

    def _build_user_text(self, entry: ShotScriptEntry) -> str:
        """构建用户提示词，填入镜头元数据。"""
        text = f"""请分析以下镜头：

镜头序号：{entry.index}
时间范围：{entry.start_ms}ms - {entry.end_ms}ms

请根据提供的 {len(entry.keyframe_asset_ids)} 张关键帧图片，描述这个镜头的画面内容和动作过程。
输出 JSON 格式。"""
        return text

    def _parse_visual_description(self, response_text: str) -> str:
        """从模型响应中提取视觉描述。

        期望模型返回 JSON，包含 visual_summary 和 motion 字段。
        """
        try:
            # 尝试从 markdown 代码块中提取 JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            data = json.loads(json_str)

            # 提取关键信息
            visual_summary: str = data.get("visual_summary", "")
            motion: str = data.get("motion", "")

            if visual_summary and motion:
                return f"{visual_summary} | 动作：{motion}"
            elif visual_summary:
                return visual_summary
            else:
                return json.dumps(data, ensure_ascii=False)

        except (json.JSONDecodeError, KeyError, IndexError):
            # JSON 解析失败，返回原始文本
            return response_text
