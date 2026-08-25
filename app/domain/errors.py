"""能力执行错误。

设计文档 8.3 要求错误返回统一错误码、是否可重试及建议动作。这里只定义骨架与
T0-2 自身需要的几个错误码；完整错误码表属于 T0-3。
"""

from __future__ import annotations

from enum import StrEnum


class SuggestedAction(StrEnum):
    """告诉调用方下一步该做什么，避免每个调用点自己猜。"""

    RETRY = "retry"
    FIX_INPUT = "fix_input"
    MANUAL_REVIEW = "manual_review"
    CONTACT_ADMIN = "contact_admin"


class ErrorCode(StrEnum):
    CAPABILITY_NOT_FOUND = "capability_not_found"
    INVALID_INPUT_REFS = "invalid_input_refs"
    INVALID_PARAMETERS = "invalid_parameters"
    MISSING_REQUIRED_PROMPT = "missing_required_prompt"
    UNKNOWN_PROMPT_SUPPLIED = "unknown_prompt_supplied"
    JOB_NOT_FOUND = "job_not_found"
    JOB_NOT_CANCELLABLE = "job_not_cancellable"
    JOB_NOT_RETRYABLE = "job_not_retryable"


class CapabilityError(Exception):
    """能力层的受控失败。未被包装成本异常的错误一律视为缺陷。"""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retryable: bool,
        suggested_action: SuggestedAction,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.suggested_action = suggested_action


class MissingPromptError(CapabilityError):
    def __init__(self, capability: str, keys: tuple[str, ...]) -> None:
        super().__init__(
            ErrorCode.MISSING_REQUIRED_PROMPT,
            f"能力 {capability} 缺少必需的提示词：{', '.join(keys)}",
            retryable=False,
            suggested_action=SuggestedAction.FIX_INPUT,
        )
        self.keys = keys


class UnknownPromptError(CapabilityError):
    def __init__(self, capability: str, keys: tuple[str, ...]) -> None:
        super().__init__(
            ErrorCode.UNKNOWN_PROMPT_SUPPLIED,
            f"能力 {capability} 未声明这些提示词，拒绝接收：{', '.join(keys)}",
            retryable=False,
            suggested_action=SuggestedAction.FIX_INPUT,
        )
        self.keys = keys


class InvalidInputRefsError(CapabilityError):
    def __init__(self, capability: str, message: str) -> None:
        super().__init__(
            ErrorCode.INVALID_INPUT_REFS,
            f"能力 {capability} 的输入引用不合法：{message}",
            retryable=False,
            suggested_action=SuggestedAction.FIX_INPUT,
        )


class InvalidParametersError(CapabilityError):
    def __init__(self, capability: str, message: str) -> None:
        super().__init__(
            ErrorCode.INVALID_PARAMETERS,
            f"能力 {capability} 的参数不合法：{message}",
            retryable=False,
            suggested_action=SuggestedAction.FIX_INPUT,
        )
