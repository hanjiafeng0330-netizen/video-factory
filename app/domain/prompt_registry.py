"""提示词注册表（开发计划 T1-5、1.3 节）。

治理规则做成结构约束，而不是文档约定：

1. **不硬编码**：能力模块拿不到注册表（边界契约禁止 `capabilities` import
   `app.prompts`），只能声明需要哪些键位，由调用方注入解析结果。
2. **先评审后落库**：新版本一律以 `DRAFT` 状态落地，`DRAFT` **无法被解析用于
   生产**。激活是一个独立、需留痕的动作。
3. **可查可改**：按环节分组浏览，后台可编辑。
4. **改动即新版本**：正文不可变，改动追加新版本，旧版本永远可渲染。
5. **变更留痕**：激活与回滚写审计日志。

第 2 条是最重要的一条。它把「提示词写完先给业务看」从一句口头约定变成了一个技术
事实：开发把草案写进注册表这个动作本身是安全的，因为草案跑不起来；而激活需要
显式操作并留下谁在什么时候激活了哪一版。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.prompts import PromptKey, ResolvedPrompt


class PromptStatus(StrEnum):
    DRAFT = "draft"
    """草案。已入库、可预览、**不可用于生产**，等待业务评审。"""

    ACTIVE = "active"
    """已评审通过并激活。同一 key 同时最多一个 active 版本。"""

    RETIRED = "retired"
    """曾激活过、现已被新版本取代。仍可渲染（历史产物要能重算）。"""


class PromptTemplate(BaseModel):
    """逻辑提示词：一个键位及其契约。正文在各版本里。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: PromptKey
    stage: str = Field(min_length=1, max_length=64)
    """所属生产环节。配置管理台（T2-6）按它分组展示。"""

    purpose: str = Field(min_length=1, max_length=300)
    variables: tuple[str, ...] = ()
    """模板声明的变量。渲染时缺一个就报错，而不是渲染出一个带空洞的提示词。"""


class PromptVersion(BaseModel):
    """不可变的提示词版本。"""

    model_config = ConfigDict(extra="forbid")

    key: PromptKey
    version: Annotated[int, Field(ge=1)]
    body: str = Field(min_length=1)
    status: PromptStatus
    change_note: str = Field(min_length=1, max_length=500)
    """为什么改。必填——「上一版为什么不够用」是后续调优唯一可靠的线索。"""

    author: str = Field(min_length=1, max_length=64)
    created_at: datetime
    activated_at: datetime | None = None
    activated_by: str | None = None

    @property
    def is_usable(self) -> bool:
        """能否用于生产。草案不行。"""
        return self.status in (PromptStatus.ACTIVE, PromptStatus.RETIRED)


class PromptRenderError(Exception):
    """渲染失败。缺变量、多变量、模板语法错误都归此类。"""


class PromptRegistry(Protocol):
    """实现见 `app.prompts`。"""

    def register(self, template: PromptTemplate) -> PromptTemplate:
        """登记一个键位。重复登记同 key 且契约不同应当报错——契约变了意味着
        调用方的注入代码也要改，静默覆盖会让两边错位。"""
        ...

    def templates(self) -> tuple[PromptTemplate, ...]: ...

    def templates_by_stage(self) -> dict[str, tuple[PromptTemplate, ...]]: ...

    def add_version(self, key: str, body: str, *, change_note: str, author: str) -> PromptVersion:
        """追加一个版本。**一律以 DRAFT 落地**，不能直接激活。"""
        ...

    def activate(self, key: str, version: int, *, actor: str) -> PromptVersion:
        """激活某版本。业务评审通过后的显式动作，会写审计。"""
        ...

    def history(self, key: str) -> tuple[PromptVersion, ...]: ...

    def get_version(self, key: str, version: int) -> PromptVersion: ...

    def active_version(self, key: str) -> PromptVersion | None: ...

    def resolve(self, key: str, variables: dict[str, object]) -> ResolvedPrompt:
        """解析当前激活版本并渲染。没有激活版本或只有草案时报错。"""
        ...

    def resolve_version(
        self, key: str, version: int, variables: dict[str, object]
    ) -> ResolvedPrompt:
        """解析指定版本。用于重算历史产物——它们必须能用当初那一版重跑。"""
        ...


class PromptRequirementReport(BaseModel):
    """某个能力所需提示词的就绪情况。

    配置管理台和启动自检都用它回答「这个环节现在能不能跑」。做成显式结构而不是
    让调用方自己去比对，是因为「提示词还是草案」这种情况需要一个明确的、可以展示
    给运营的答案，而不是一个 KeyError。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: PromptKey
    registered: bool
    active_version: int | None
    draft_versions: tuple[int, ...]

    @property
    def ready(self) -> bool:
        return self.registered and self.active_version is not None

    @model_validator(mode="after")
    def _active_implies_registered(self) -> Self:
        if self.active_version is not None and not self.registered:
            raise ValueError("未登记的键位不可能有激活版本")
        return self
