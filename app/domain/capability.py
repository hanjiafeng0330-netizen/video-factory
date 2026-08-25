"""统一能力接口协议（设计文档 8.1 / 8.3）。

每个能力模块都是一个 `Capability` 子类，声明自己的输入契约、参数模型和所需提示词，
并实现纯粹的 `run()`。围绕它的路由、任务、幂等、血缘、事件全部由框架处理，能力
模块本身不感知这些。
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.errors import (
    InvalidInputRefsError,
    InvalidParametersError,
    MissingPromptError,
    UnknownPromptError,
)
from app.domain.prompts import PromptRequirement, ResolvedPrompt
from app.domain.refs import ArtifactRef, ArtifactType


class CapabilityRequest(BaseModel):
    """能力执行请求（设计文档 8.1 的请求体）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_refs: tuple[ArtifactRef, ...] = ()
    parameters: Mapping[str, Any] = Field(default_factory=dict)
    resolved_prompts: tuple[ResolvedPrompt, ...] = ()
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]

    def prompt(self, key: str) -> ResolvedPrompt:
        """取一个已注入的提示词。缺失即抛错，不做静默降级。"""
        for item in self.resolved_prompts:
            if item.key == key:
                return item
        raise KeyError(key)

    def fingerprint(self) -> str:
        """请求指纹。幂等键相同但指纹不同意味着调用方复用了幂等键，
        这是必须报错的情况——T1-4 会用到它。
        """
        payload = {
            "input_refs": [str(ref) for ref in self.input_refs],
            "parameters": self.parameters,
            "resolved_prompts": [str(p) for p in self.resolved_prompts],
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class CapabilityResult(BaseModel):
    """能力执行结果。

    产物本身已经持久化，这里只回引用——大文件与长文本一律不进消息体
    （设计文档 8.3）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_refs: tuple[ArtifactRef, ...] = ()
    metrics: Mapping[str, float] = Field(default_factory=dict)
    notes: tuple[str, ...] = ()


class CapabilitySpec(BaseModel):
    """能力的自描述，由 `GET /capabilities/{name}/schema` 返回。

    配置管理台（T2-6）依赖它来渲染「这个环节用了哪些提示词」。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    stage: str
    summary: str
    accepts: tuple[ArtifactType, ...]
    produces: tuple[ArtifactType, ...]
    required_prompts: tuple[PromptRequirement, ...]
    parameters_schema: Mapping[str, Any]
    supports_async: bool


class Capability[ParamsT: BaseModel](ABC):
    """能力模块基类。

    子类只需声明元数据、参数模型和 `run()`。输入、参数、提示词三者的契约校验全部
    由基类的 `validate_request()` 统一完成，避免每个能力各写一遍、各漏一点——
    漏掉参数校验会让一个坏请求以 500 而不是 422 结束，那是框架的问题，不该让
    每个能力自己记得处理。
    """

    name: ClassVar[str]
    stage: ClassVar[str]
    summary: ClassVar[str]
    accepts: ClassVar[tuple[ArtifactType, ...]] = ()
    produces: ClassVar[tuple[ArtifactType, ...]] = ()
    required_prompts: ClassVar[tuple[PromptRequirement, ...]] = ()
    parameters_model: type[ParamsT]
    supports_async: ClassVar[bool] = True

    def spec(self) -> CapabilitySpec:
        return CapabilitySpec(
            name=self.name,
            stage=self.stage,
            summary=self.summary,
            accepts=self.accepts,
            produces=self.produces,
            required_prompts=self.required_prompts,
            parameters_schema=self.parameters_model.model_json_schema(),
            supports_async=self.supports_async,
        )

    def validate_request(self, request: CapabilityRequest) -> ParamsT:
        """校验请求是否满足本能力声明的契约，并返回解析好的参数。

        四件事：输入类型必须在 `accepts` 白名单内；参数必须通过 `parameters_model`；
        必需提示词一个都不能少；未声明的提示词一个都不能多。最后一条同样报错是
        刻意的——多传提示词通常意味着调用方或配置管理台的键位写错了，静默忽略会让
        排查变成猜谜。
        """
        allowed = set(self.accepts)
        unexpected = sorted({ref.type for ref in request.input_refs} - allowed)
        if unexpected:
            raise InvalidInputRefsError(
                self.name,
                f"不接受的产物类型 {unexpected}，本能力只接受 {sorted(allowed)}",
            )

        try:
            params = self.parameters_model.model_validate(request.parameters)
        except ValidationError as exc:
            raise InvalidParametersError(self.name, str(exc)) from exc

        supplied = {p.key for p in request.resolved_prompts}
        declared = {p.key for p in self.required_prompts}

        mandatory = {p.key for p in self.required_prompts if p.required}
        missing = tuple(sorted(mandatory - supplied))
        if missing:
            raise MissingPromptError(self.name, missing)

        unknown = tuple(sorted(supplied - declared))
        if unknown:
            raise UnknownPromptError(self.name, unknown)

        return params

    @abstractmethod
    async def run(self, request: CapabilityRequest, params: ParamsT) -> CapabilityResult:
        """执行能力。参数已解析，输入与提示词契约已校验通过。"""

    async def execute(self, request: CapabilityRequest) -> CapabilityResult:
        return await self.run(request, self.validate_request(request))


CapabilityRegistry = Mapping[str, Capability[Any]]
CapabilityList = Sequence[Capability[Any]]
