"""能力注册表。

新增能力时在 `_CAPABILITIES` 里加一行即可，路由、schema、任务通路自动生效。
注册表位于 `app.capabilities` 包内但不属于任何单个能力，因此不受「能力模块
互不依赖」契约约束——它只做汇总，不在能力之间建立依赖。
"""

from __future__ import annotations

from typing import Any

from app.capabilities.echo.capability import EchoCapability
from app.domain.capability import Capability
from app.domain.errors import CapabilityError, ErrorCode

_CAPABILITIES: tuple[Capability[Any], ...] = (EchoCapability(),)

_BY_NAME: dict[str, Capability[Any]] = {c.name: c for c in _CAPABILITIES}

if len(_BY_NAME) != len(_CAPABILITIES):
    raise RuntimeError("能力名称重复，注册表构建失败")


def list_capabilities() -> tuple[Capability[Any], ...]:
    return _CAPABILITIES


def get_capability(name: str) -> Capability[Any]:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise CapabilityError(
            ErrorCode.CAPABILITY_NOT_FOUND,
            f"未注册的能力：{name}",
        ) from None
