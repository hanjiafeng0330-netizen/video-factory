"""大模型调用协议。

能力模块只依赖这个协议，不依赖任何供应商 SDK 或适配器实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class LLMClient(Protocol):
    def vision(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        image_paths: tuple[Path, ...],
        max_tokens: int,
    ) -> Any: ...

    def text(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        max_tokens: int,
    ) -> Any: ...
