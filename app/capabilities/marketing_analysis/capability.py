"""营销口径分析能力（T3-4）。

基于镜头脚本（画面描述 + 台词 + 视觉信息），提取营销维度：
- 钩子：开头的情绪/好奇调动方式
- 痛点：用户痛点，出现在哪些分镜
- 卖点结构：营销卖点的叙事顺序
- 视觉风格：镜头语言、表达方式、场景
- 备注：其他营销相关信息

输入：ShotScript（带画面描述和台词）
输出：MarketingAnalysis（结构化营销分析报告）
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.domain.assets import AssetStore
from app.domain.capability import Capability, CapabilityRequest, CapabilityResult
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.llm import LLMClient
from app.domain.prompt_registry import PromptRegistry
from app.domain.prompts import PromptRequirement
from app.domain.refs import ArtifactType
from app.domain.shot_script import ShotScript
from app.domain.versioning import ArtifactRepository, ArtifactStatus


class MarketingAnalysis(BaseModel):
    """营销口径分析结果。"""

    model_config = ConfigDict(extra="forbid")

    hook: str = ""
    """钩子：开头的情绪/好奇调动方式"""

    pain_points: str = ""
    """痛点：用户痛点，出现在哪些分镜"""

    selling_point_structure: str = ""
    """卖点结构：营销卖点的叙事顺序"""

    visual_style: str = ""
    """视觉风格：镜头语言、表达方式、场景"""

    notes: str = ""
    """备注：其他营销相关信息"""

    raw_output: str = ""
    """模型原始输出，用于调试"""

    created_at: datetime = Field(default_factory=datetime.now)


class MarketingAnalysisParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_output_tokens: int = Field(default=2000, ge=500, le=8000)
    text_model: str | None = None
    """本次任务覆盖默认文本模型。由 API 按模型目录校验后传入。"""


class MarketingAnalysisCapability(Capability[MarketingAnalysisParameters]):
    name: ClassVar[str] = "marketing_analysis"
    stage: ClassVar[str] = "营销分析"
    summary: ClassVar[str] = "基于镜头脚本提取营销口径（钩子/痛点/卖点/视觉风格）"
    accepts: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.SHOT_SCRIPT,)
    produces: ClassVar[tuple[ArtifactType, ...]] = (ArtifactType.MARKETING_ANALYSIS,)
    parameters_model = MarketingAnalysisParameters
    required_prompts: ClassVar[tuple[PromptRequirement, ...]] = (
        PromptRequirement(
            key="marketing_analysis.prompt",
            purpose="基于镜头脚本提取营销口径维度",
        ),
    )

    def __init__(
        self,
        *,
        artifacts: ArtifactRepository,
        assets: AssetStore,
        llm: LLMClient,
        prompts: PromptRegistry,
        text_model: str,
    ) -> None:
        self._artifacts = artifacts
        self._assets = assets
        self._llm = llm
        self._prompts = prompts
        self._text_model = text_model

    async def run(
        self, request: CapabilityRequest, params: MarketingAnalysisParameters
    ) -> CapabilityResult:
        import asyncio

        return await asyncio.to_thread(self._run_sync, request, params)

    def _run_sync(
        self, request: CapabilityRequest, params: MarketingAnalysisParameters
    ) -> CapabilityResult:
        # 找到输入镜头脚本
        shot_script_ref = next(
            (ref for ref in request.input_refs if ref.type == ArtifactType.SHOT_SCRIPT),
            None,
        )
        if not shot_script_ref:
            raise CapabilityError(
                ErrorCode.INVALID_INPUT_REFS,
                "marketing_analysis 需要 shot_script 作为输入",
            )

        shot_script = self._artifacts.get(shot_script_ref).body_as(ShotScript)

        # 渲染提示词
        resolved_prompt = request.prompt("marketing_analysis.prompt")
        variables: dict[str, object] = {
            "shot_script": self._format_shot_script_for_analysis(shot_script),
        }
        prompt = self._prompts.resolve(resolved_prompt.key, variables)

        # 调用 LLM
        response = self._llm.text(
            model=params.text_model or self._text_model,
            system=prompt.text,
            user_text="请分析以上镜头脚本的营销口径。",
            max_tokens=params.max_output_tokens,
        )

        # 解析输出
        analysis = self._parse_analysis(response.text)

        # 保存产物
        result = self._artifacts.create_version(
            artifact_type=ArtifactType.MARKETING_ANALYSIS,
            artifact_id=f"ma_{shot_script_ref.id}",
            body=analysis.model_dump(),
            created_by=f"capability:{self.name}",
            sources=(shot_script_ref,),
            status=ArtifactStatus.READY,
        )

        return CapabilityResult(
            output_refs=(result.ref,),
            metrics={"output_tokens": float(response.output_tokens)},
        )

    def _format_shot_script_for_analysis(self, script: ShotScript) -> str:
        """格式化镜头脚本为分析输入。"""
        lines = []
        for entry in script.entries:
            lines.append(f"【分镜{entry.index}】{entry.start_ms}ms-{entry.end_ms}ms")
            if entry.visual_description:
                lines.append(f"  画面：{entry.visual_description}")
            if entry.lines:
                for line in entry.lines:
                    lines.append(f"  台词：{line.text}")
        return "\n".join(lines)

    def _parse_analysis(self, raw_text: str) -> MarketingAnalysis:
        """解析 LLM 输出为 MarketingAnalysis。

        期望输出格式：
        钩子：xxx
        痛点：xxx
        卖点结构：xxx
        视觉风格：xxx
        备注：xxx
        """
        analysis = MarketingAnalysis(raw_output=raw_text)

        # 简单解析
        sections = {
            "钩子": "hook",
            "痛点": "pain_points",
            "卖点结构": "selling_point_structure",
            "视觉风格": "visual_style",
            "备注": "notes",
        }

        current_section = None
        current_content = []

        for line in raw_text.split("\n"):
            matched = False
            for zh_key, attr_name in sections.items():
                if line.startswith(zh_key):
                    if current_section:
                        setattr(analysis, current_section, "\n".join(current_content).strip())
                    current_section = attr_name
                    current_content = [line.split("：", 1)[-1].strip() if "：" in line else ""]
                    matched = True
                    break
            if not matched and current_section:
                current_content.append(line)

        if current_section:
            setattr(analysis, current_section, "\n".join(current_content).strip())

        return analysis
