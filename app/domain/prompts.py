"""提示词的「注入侧」类型。

提示词注册表本身（存储、版本、渲染、后台读写）属于 T1-5 的 `app.prompts`。
这里只放能力模块需要感知的两个值对象，原因是依赖方向：

    app.prompts  ──解析/渲染──▶  ResolvedPrompt  ──注入──▶  app.capabilities

能力模块**不允许** import `app.prompts`，也不允许内联提示词字面量。它只声明
「我需要哪些提示词键位」（`PromptRequirement`），由调用方解析出确切版本后注入
（`ResolvedPrompt`）。这样提示词和 input_refs 一样是带版本的输入，可回溯、可
在后台修改，且换一版提示词不需要改任何能力模块代码。
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

PromptKey = Annotated[str, Field(pattern=r"^[a-z0-9_]+(\.[a-z0-9_]+)+$", max_length=128)]


class PromptRequirement(BaseModel):
    """能力模块对提示词的声明。会出现在 `GET /capabilities/{cap}/schema` 里，
    因此它同时是「配置管理台该给这个环节展示哪些提示词」的数据来源。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: PromptKey
    purpose: str = Field(min_length=1, max_length=200)
    variables: tuple[str, ...] = ()
    required: bool = True


class ResolvedPrompt(BaseModel):
    """已解析到确切版本并渲染完成的提示词。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: PromptKey
    version: Annotated[int, Field(ge=1)]
    text: str = Field(min_length=1)

    def __str__(self) -> str:
        return f"prompt:{self.key}@v{self.version}"
