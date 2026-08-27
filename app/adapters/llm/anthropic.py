"""LLM 适配器（Anthropic vision）。

包装 Anthropic SDK，提供 vision（多图 + 文本）和 text-only 两种调用。
返回原始结构化输出，不含任何业务逻辑。

错误映射遵循设计文档 10.1：
- 速率限制（rate limit） → 可重试
- 月度/账户额度耗尽（quota exceeded） → 不可重试，转预算/额度处理
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
    """LLM 客户端协议。"""

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


def _classify_rate_limit(exc: anthropic.RateLimitError) -> CapabilityError:
    """区分瞬时速率限制与账户额度耗尽。

    两者都以 HTTP 429 返回，但处理动作完全相反：
    - `rate_limit`：稍后重试有效；
    - `quota_exceeded`：当前账户/模型的月度额度已耗尽，重试只会重复失败，必须切换
      模型、补充额度或等待额度周期刷新。

    把 quota_exceeded 标成 provider_rate_limited 会误导工作流无限重试，浪费队列资源，
    也让运营以为「等一会就好」。设计文档 18 章要求成本上限和人工处理，这里必须
    返回 BUDGET_EXCEEDED（不可重试，联系管理员）。
    """
    message = str(exc.message)
    if "quota_exceeded" in message:
        return CapabilityError(
            ErrorCode.BUDGET_EXCEEDED,
            "LLM 账户或模型额度已耗尽（quota_exceeded）。"
            "请在模型配置中切换到有额度的模型，补充额度，或等待额度周期刷新。",
        )
    return CapabilityError(ErrorCode.PROVIDER_RATE_LIMITED, "LLM 速率限制，请稍后重试。")


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
        """多图 + 文本的 vision 调用。"""
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
            raise _classify_rate_limit(exc) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:
                raise CapabilityError(
                    ErrorCode.PROVIDER_REJECTED_REQUEST,
                    f"LLM 拒绝请求（HTTP {exc.status_code}）。",
                ) from exc
            raise CapabilityError(
                ErrorCode.PROVIDER_UNAVAILABLE, "LLM 服务暂不可用，请稍后重试。"
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise CapabilityError(ErrorCode.PROVIDER_TIMEOUT, "LLM 调用超时，请稍后重试。") from exc
        except Exception as exc:
            raise CapabilityError(ErrorCode.INTERNAL_ERROR, "LLM 调用失败，请稍后重试。") from exc

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
            raise _classify_rate_limit(exc) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code < 500:
                raise CapabilityError(
                    ErrorCode.PROVIDER_REJECTED_REQUEST,
                    f"LLM 拒绝请求（HTTP {exc.status_code}）。",
                ) from exc
            raise CapabilityError(
                ErrorCode.PROVIDER_UNAVAILABLE, "LLM 服务暂不可用，请稍后重试。"
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise CapabilityError(ErrorCode.PROVIDER_TIMEOUT, "LLM 调用超时，请稍后重试。") from exc
        except Exception as exc:
            raise CapabilityError(ErrorCode.INTERNAL_ERROR, "LLM 调用失败，请稍后重试。") from exc

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
