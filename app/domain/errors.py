"""错误码表（设计文档 8.3）。

设计文档要求「错误返回统一错误码、是否可重试及建议动作」。关键设计决定：
**这三项由错误码唯一决定，raise 点不能自行填写。**

原因是 T0-2 的写法留了个口子——同一个 `provider_timeout` 在 A 处写
`retryable=True`、B 处写 `retryable=False` 都能编译通过，而重试语义不一致
会直接导致「该重试的没重试」或「不该重试的把费用翻倍」。既然重试语义是错误
码的固有属性，就让它跟着码走；需要不同语义时应当换一个码，而不是换一个参数。

HTTP 状态码同样收在这张表里，避免 API 层维护第二份映射并忘记同步。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SuggestedAction(StrEnum):
    """告诉调用方下一步该做什么，避免每个调用点自己猜。"""

    RETRY = "retry"
    """稍后重试即可，问题是瞬时的。"""

    FIX_INPUT = "fix_input"
    """调用方的请求有问题，重试多少次都一样。"""

    MANUAL_REVIEW = "manual_review"
    """需要人介入判断，不要自动重试（设计文档 3.6 人工审核是一等能力）。"""

    SWITCH_PROVIDER = "switch_provider"
    """换供应商或换生成策略（设计文档 5.1 的 N→P→L 分支）。"""

    CONTACT_ADMIN = "contact_admin"
    """配置、密钥或预算问题，运营与管理员处理。"""


class ErrorCategory(StrEnum):
    CLIENT = "client"
    """调用方请求不合法。"""

    RESOURCE = "resource"
    """引用的对象不存在。"""

    CONFLICT = "conflict"
    """对象存在但当前状态不允许该操作。"""

    UPSTREAM = "upstream"
    """第三方模型或视频服务侧的问题（设计文档 18 章「第三方不稳定」）。"""

    GOVERNANCE = "governance"
    """合规、版权、预算等治理规则拦截，一律需要人介入。"""

    INTERNAL = "internal"
    """系统自身缺陷或基础设施故障。"""


class ErrorCode(StrEnum):
    # --- CLIENT ---------------------------------------------------------
    INVALID_INPUT_REFS = "invalid_input_refs"
    INVALID_PARAMETERS = "invalid_parameters"
    MISSING_REQUIRED_PROMPT = "missing_required_prompt"
    UNKNOWN_PROMPT_SUPPLIED = "unknown_prompt_supplied"
    PROMPT_VARIABLE_MISSING = "prompt_variable_missing"
    PROMPT_NOT_USABLE = "prompt_not_usable"
    ARTIFACT_RIGHTS_UNCLEARED = "artifact_rights_uncleared"

    # --- RESOURCE -------------------------------------------------------
    CAPABILITY_NOT_FOUND = "capability_not_found"
    JOB_NOT_FOUND = "job_not_found"
    ARTIFACT_NOT_FOUND = "artifact_not_found"
    PROMPT_NOT_FOUND = "prompt_not_found"

    # --- CONFLICT -------------------------------------------------------
    JOB_NOT_CANCELLABLE = "job_not_cancellable"
    JOB_NOT_RETRYABLE = "job_not_retryable"
    IDEMPOTENCY_KEY_REUSED = "idempotency_key_reused"
    ARTIFACT_VERSION_STALE = "artifact_version_stale"
    ARTIFACT_IMMUTABLE = "artifact_immutable"
    ARTIFACT_STATUS_TRANSITION_INVALID = "artifact_status_transition_invalid"
    REVIEW_ALREADY_DECIDED = "review_already_decided"

    # --- UPSTREAM -------------------------------------------------------
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_REJECTED_REQUEST = "provider_rejected_request"
    MODEL_OUTPUT_UNPARSEABLE = "model_output_unparseable"

    # --- GOVERNANCE -----------------------------------------------------
    FORBIDDEN_CLAIM_DETECTED = "forbidden_claim_detected"
    EVIDENCE_MISSING = "evidence_missing"
    SIMILARITY_RISK_EXCEEDED = "similarity_risk_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"

    # --- INTERNAL -------------------------------------------------------
    STORAGE_FAILURE = "storage_failure"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True, slots=True)
class ErrorDefinition:
    category: ErrorCategory
    retryable: bool
    suggested_action: SuggestedAction
    http_status: int
    description: str


def _client(action: SuggestedAction, status: int, description: str) -> ErrorDefinition:
    return ErrorDefinition(ErrorCategory.CLIENT, False, action, status, description)


ERROR_CATALOG: dict[ErrorCode, ErrorDefinition] = {
    # --- CLIENT ---------------------------------------------------------
    ErrorCode.INVALID_INPUT_REFS: _client(
        SuggestedAction.FIX_INPUT, 422, "输入产物类型不在该能力的 accepts 白名单内"
    ),
    ErrorCode.INVALID_PARAMETERS: _client(
        SuggestedAction.FIX_INPUT, 422, "参数未通过该能力的 parameters_model 校验"
    ),
    ErrorCode.MISSING_REQUIRED_PROMPT: _client(
        SuggestedAction.FIX_INPUT, 422, "缺少能力声明的必需提示词，不允许静默降级"
    ),
    ErrorCode.UNKNOWN_PROMPT_SUPPLIED: _client(
        SuggestedAction.FIX_INPUT, 422, "注入了能力未声明的提示词，通常是键位写错"
    ),
    ErrorCode.PROMPT_VARIABLE_MISSING: _client(
        SuggestedAction.FIX_INPUT, 422, "渲染提示词时缺少模板声明的变量"
    ),
    ErrorCode.PROMPT_NOT_USABLE: _client(
        SuggestedAction.CONTACT_ADMIN,
        409,
        "引用的提示词版本处于 draft 或已停用，不允许参与生产（计划 1.3 节）",
    ),
    ErrorCode.ARTIFACT_RIGHTS_UNCLEARED: _client(
        SuggestedAction.MANUAL_REVIEW,
        409,
        "热点视频未标记 rights_status，禁止进入下游生产（设计文档 13.3）",
    ),
    # --- RESOURCE -------------------------------------------------------
    ErrorCode.CAPABILITY_NOT_FOUND: ErrorDefinition(
        ErrorCategory.RESOURCE, False, SuggestedAction.FIX_INPUT, 404, "能力未注册"
    ),
    ErrorCode.JOB_NOT_FOUND: ErrorDefinition(
        ErrorCategory.RESOURCE, False, SuggestedAction.FIX_INPUT, 404, "任务不存在"
    ),
    ErrorCode.ARTIFACT_NOT_FOUND: ErrorDefinition(
        ErrorCategory.RESOURCE, False, SuggestedAction.FIX_INPUT, 404, "产物或产物版本不存在"
    ),
    ErrorCode.PROMPT_NOT_FOUND: ErrorDefinition(
        ErrorCategory.RESOURCE, False, SuggestedAction.CONTACT_ADMIN, 404, "提示词键位未登记"
    ),
    # --- CONFLICT -------------------------------------------------------
    ErrorCode.JOB_NOT_CANCELLABLE: ErrorDefinition(
        ErrorCategory.CONFLICT, False, SuggestedAction.FIX_INPUT, 409, "任务已处于终态，无法取消"
    ),
    ErrorCode.JOB_NOT_RETRYABLE: ErrorDefinition(
        ErrorCategory.CONFLICT, False, SuggestedAction.FIX_INPUT, 409, "任务当前状态不允许重试"
    ),
    ErrorCode.IDEMPOTENCY_KEY_REUSED: ErrorDefinition(
        ErrorCategory.CONFLICT,
        False,
        SuggestedAction.FIX_INPUT,
        409,
        "幂等键相同但请求指纹不同，调用方复用了幂等键",
    ),
    ErrorCode.ARTIFACT_VERSION_STALE: ErrorDefinition(
        ErrorCategory.CONFLICT,
        False,
        SuggestedAction.MANUAL_REVIEW,
        409,
        "上游产物已产生新版本，当前引用被标记为可能过期（设计文档 9.2）",
    ),
    ErrorCode.ARTIFACT_IMMUTABLE: ErrorDefinition(
        ErrorCategory.CONFLICT,
        False,
        SuggestedAction.FIX_INPUT,
        409,
        "产物版本内容不可变，修改必须产生新版本（设计文档 11.2）",
    ),
    ErrorCode.ARTIFACT_STATUS_TRANSITION_INVALID: ErrorDefinition(
        ErrorCategory.CONFLICT,
        False,
        SuggestedAction.FIX_INPUT,
        409,
        "产物状态不允许该跃迁",
    ),
    ErrorCode.REVIEW_ALREADY_DECIDED: ErrorDefinition(
        ErrorCategory.CONFLICT, False, SuggestedAction.FIX_INPUT, 409, "该审核节点已有结论"
    ),
    # --- UPSTREAM -------------------------------------------------------
    ErrorCode.PROVIDER_UNAVAILABLE: ErrorDefinition(
        ErrorCategory.UPSTREAM, True, SuggestedAction.RETRY, 503, "第三方服务不可用"
    ),
    ErrorCode.PROVIDER_RATE_LIMITED: ErrorDefinition(
        ErrorCategory.UPSTREAM, True, SuggestedAction.RETRY, 429, "第三方限流"
    ),
    ErrorCode.PROVIDER_TIMEOUT: ErrorDefinition(
        ErrorCategory.UPSTREAM, True, SuggestedAction.RETRY, 504, "第三方任务超时"
    ),
    ErrorCode.PROVIDER_REJECTED_REQUEST: ErrorDefinition(
        ErrorCategory.UPSTREAM,
        False,
        SuggestedAction.SWITCH_PROVIDER,
        422,
        "第三方拒绝该请求（内容策略或参数不受支持），重试无用",
    ),
    ErrorCode.MODEL_OUTPUT_UNPARSEABLE: ErrorDefinition(
        ErrorCategory.UPSTREAM,
        True,
        SuggestedAction.MANUAL_REVIEW,
        502,
        "模型输出结构校验失败。有限次自动修复后仍失败则转人工，"
        "不允许把不完整内容传给下游（设计文档第 12 章）",
    ),
    # --- GOVERNANCE -----------------------------------------------------
    ErrorCode.FORBIDDEN_CLAIM_DETECTED: ErrorDefinition(
        ErrorCategory.GOVERNANCE,
        False,
        SuggestedAction.MANUAL_REVIEW,
        422,
        "命中禁用表达或禁用声明（设计文档 13.3）",
    ),
    ErrorCode.EVIDENCE_MISSING: ErrorDefinition(
        ErrorCategory.GOVERNANCE,
        False,
        SuggestedAction.MANUAL_REVIEW,
        422,
        "产品声明缺少可核验证据，未证实信息禁止生成（设计文档 18 章）",
    ),
    ErrorCode.SIMILARITY_RISK_EXCEEDED: ErrorDefinition(
        ErrorCategory.GOVERNANCE,
        False,
        SuggestedAction.MANUAL_REVIEW,
        422,
        "与来源内容相似度超过阈值，存在版权风险（设计文档 13.3）",
    ),
    ErrorCode.BUDGET_EXCEEDED: ErrorDefinition(
        ErrorCategory.GOVERNANCE,
        False,
        SuggestedAction.CONTACT_ADMIN,
        402,
        "超出任务或流程预算上限（设计文档 18 章「成本失控」）",
    ),
    # --- INTERNAL -------------------------------------------------------
    ErrorCode.STORAGE_FAILURE: ErrorDefinition(
        ErrorCategory.INTERNAL, True, SuggestedAction.RETRY, 503, "对象存储或数据库写入失败"
    ),
    ErrorCode.INTERNAL_ERROR: ErrorDefinition(
        ErrorCategory.INTERNAL, False, SuggestedAction.CONTACT_ADMIN, 500, "未预期的系统错误"
    ),
}

# 新增错误码时必须同时定义它的语义，否则连导入都会失败。
# 这比「运行到那一行才发现没定义」早得多。
_undefined = set(ErrorCode) - ERROR_CATALOG.keys()
if _undefined:
    raise RuntimeError(f"以下错误码未在 ERROR_CATALOG 中定义：{sorted(_undefined)}")


def describe(code: ErrorCode) -> ErrorDefinition:
    return ERROR_CATALOG[code]


class CapabilityError(Exception):
    """受控失败。未被包装成本异常的错误一律视为缺陷。

    只接收错误码与人可读消息；`retryable` / `suggested_action` / `http_status`
    一律从错误码表派生，调用点无法覆盖。
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        definition = ERROR_CATALOG[code]
        self.code = code
        self.message = message
        self.category = definition.category
        self.retryable = definition.retryable
        self.suggested_action = definition.suggested_action
        self.http_status = definition.http_status


class MissingPromptError(CapabilityError):
    def __init__(self, capability: str, keys: tuple[str, ...]) -> None:
        super().__init__(
            ErrorCode.MISSING_REQUIRED_PROMPT,
            f"能力 {capability} 缺少必需的提示词：{', '.join(keys)}",
        )
        self.keys = keys


class UnknownPromptError(CapabilityError):
    def __init__(self, capability: str, keys: tuple[str, ...]) -> None:
        super().__init__(
            ErrorCode.UNKNOWN_PROMPT_SUPPLIED,
            f"能力 {capability} 未声明这些提示词，拒绝接收：{', '.join(keys)}",
        )
        self.keys = keys


class InvalidInputRefsError(CapabilityError):
    def __init__(self, capability: str, message: str) -> None:
        super().__init__(
            ErrorCode.INVALID_INPUT_REFS,
            f"能力 {capability} 的输入引用不合法：{message}",
        )


class InvalidParametersError(CapabilityError):
    def __init__(self, capability: str, message: str) -> None:
        super().__init__(
            ErrorCode.INVALID_PARAMETERS,
            f"能力 {capability} 的参数不合法：{message}",
        )
