"""API 层对外部依赖的声明。

API 层不认识任何具体实现——不认识内存仓储、本地资产存储，也不认识 ffmpeg。它只
声明「我需要一个具备这些能力的容器」，由组合根（`app.bootstrap`）提供。

这样做不是为了好看：`app.api` 一旦能 import `app.adapters` 或 `app.storage`，
就没有任何机制阻止某个路由为了赶进度直接调用适配器，而那会绕过能力模块的参数
校验、提示词契约、幂等与血缘。边界契约里 `api-not-adapters` 那一条守的就是这个，
它在这个文件出现之前拦住过一次真实违规。
"""

from __future__ import annotations

from typing import Protocol

from fastapi import Request

from app.capabilities.registry import CapabilityRegistry
from app.domain.assets import AssetStore
from app.domain.lineage import LineageRepository
from app.domain.prompt_registry import PromptRegistry
from app.domain.versioning import ArtifactRepository
from app.platform.config import Settings


class Container(Protocol):
    """API 层需要的全部依赖。用结构化协议而非具体类型，实现方无需继承。"""

    @property
    def capabilities(self) -> CapabilityRegistry: ...

    @property
    def artifacts(self) -> ArtifactRepository: ...

    @property
    def assets(self) -> AssetStore: ...

    @property
    def lineage(self) -> LineageRepository: ...

    @property
    def prompts(self) -> PromptRegistry: ...

    @property
    def settings(self) -> Settings: ...


def container_of(request: Request) -> Container:
    container: Container = request.app.state.container
    return container
