"""`echo` 参考能力。

它不是业务能力，而是统一能力协议的**可执行规范**：新增任何真实能力时照它抄。
它刻意声明了一个必需提示词，用来证明「提示词由外部注入、缺失即报错」这条通路
在框架层就成立，而不是等到第一个真实 LLM 能力才发现。
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.domain.capability import Capability, CapabilityRequest, CapabilityResult
from app.domain.prompts import PromptRequirement


class EchoParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=500)
    repeat: int = Field(default=1, ge=1, le=10)


class EchoCapability(Capability[EchoParameters]):
    name: ClassVar[str] = "echo"
    stage: ClassVar[str] = "diagnostics"
    summary: ClassVar[str] = "回显参考能力，用于验证能力协议、任务通路与提示词注入"
    parameters_model = EchoParameters
    required_prompts: ClassVar[tuple[PromptRequirement, ...]] = (
        PromptRequirement(
            key="echo.system",
            purpose="回显能力的前缀模板，仅用于协议自检，不参与任何生产内容",
            variables=(),
        ),
    )

    async def run(self, request: CapabilityRequest, params: EchoParameters) -> CapabilityResult:
        prefix = request.prompt("echo.system").text
        lines = tuple(f"{prefix} :: {params.message}" for _ in range(params.repeat))
        return CapabilityResult(
            metrics={"lines": float(len(lines)), "characters": float(sum(map(len, lines)))},
            notes=lines,
        )
