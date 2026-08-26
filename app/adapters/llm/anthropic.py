"""LLM 适配器（Anthropic vision）。

包装 anthropic SDK，提供 vision（多图 + 文本）和 text-only 两种调用。
返回原始结构化输出，不含任何业务逻辑。

错误映射遵循设计文档 10.1：
- 4xx（参数错误、速率限制等） → 不可重试
- 5xx 和超时 → 可重试
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import anthropic

from app.domain.errors import CapabilityError, ErrorCode


class LLMResponse:
    """LLM 调用的结构化输出。"""

    def __init__(self, text: str, raw_response: Any, usage: dict[str, int]) -> None:
        self.text = text
        self.raw = raw_response
        self.usage = usage

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)


class LLMClient(Protocol):
    """LLM 客户端协议。

    能力模块只认这个协议，不直接 import anthropic SDK。
    """

    def vision(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        image_paths: tuple[Path, ...],
        max_tokens: int,
    ) -> LLMResponse: ...

    def text(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        max_tokens: int,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Anthropic Claude 客户端实现。"""

    def __init__(self, base_url: str, api_key: str, timeout: float = 300) -> None:
        self._client = anthropic.Anthropic(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

    def vision(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        image_paths: tuple[Path, ...],
        max_tokens: int,
    ) -> LLMResponse:
        """多图 + 文本的 vision 调用。

        image_paths 里的每张图片都会被编码为 base64 放进消息里。
        """
        if not image_paths:
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "vision 调用至少需要一张图片")

        content: list[dict[str, Any]] = []
        for path in image_paths:
            if not path.is_file():
                raise CapabilityError(ErrorCode.STORAGE_FAILURE, f"图片文件不存在：{path}")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self._detect_media_type(path),
                        "data": self._encode_base64(path),
                    },
                }
            )
        content.append({"type": "text", "text": user_text})

        try:
            response = self._client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": content}],  # type: ignore[typeddict-item]
                max_tokens=max_tokens,
            )
        except anthropic.RateLimitError as exc:
            raise CapabilityError(
                ErrorCode.PROVIDER_RATE_LIMITED, f"LLM 速率限制：{exc.message}"
            ) from exc
        except anthropic.APIStatusError as exc:
            # 4xx 不可重试，5xx 可重试
            if exc.status_code < 500:
                raise CapabilityError(
                    ErrorCode.PROVIDER_REJECTED_REQUEST,
                    f"LLM 拒绝请求 ({exc.status_code})：{exc.message}",
                ) from exc
            raise CapabilityError(
                ErrorCode.PROVIDER_UNAVAILABLE, f"LLM 服务不可用：{exc.message}"
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise CapabilityError(ErrorCode.PROVIDER_TIMEOUT, f"LLM 调用超时：{exc}") from exc
        except Exception as exc:
            raise CapabilityError(ErrorCode.INTERNAL_ERROR, f"LLM 调用失败：{exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            raw_response=response,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def text(
        self,
        *,
        model: str,
        system: str,
        user_text: str,
        max_tokens: int,
    ) -> LLMResponse:
        """纯文本调用。"""
        try:
            response = self._client.messages.create(
                model=model,
                system=system,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=max_tokens,
            )
        except anthropic.RateLimitError as exc:
            raise CapabilityError(
                ErrorCode.PROVIDER_RATE_LIMITED, f"LLM 速率限制：{exc.message}"
            ) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:
                raise CapabilityError(
                    ErrorCode.PROVIDER_REJECTED_REQUEST,
                    f"LLM 拒绝请求 ({exc.status_code})：{exc.message}",
                ) from exc
            raise CapabilityError(
                ErrorCode.PROVIDER_UNAVAILABLE, f"LLM 服务不可可用：{exc.message}"
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise CapabilityError(ErrorCode.PROVIDER_TIMEOUT, f"LLM 调用超时：{exc}") from exc
        except Exception as exc:
            raise CapabilityError(ErrorCode.INTERNAL_ERROR, f"LLM 调用失败：{exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            raw_response=response,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    @staticmethod
    def _detect_media_type(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/jpeg")

    @staticmethod
    def _encode_base64(path: Path) -> str:
        import base64

        return base64.b64encode(path.read_bytes()).decode()
