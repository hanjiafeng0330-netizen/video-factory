"""T0-3 验收：错误码表（设计文档 8.3）。

设计文档要求「错误返回统一错误码、是否可重试及建议动作」。这些测试守住的核心
性质是：**同一个错误码在任何地方的重试语义都一致**，因为不一致会直接导致
「该重试的没重试」或「不该重试的把第三方费用翻倍」。
"""

from __future__ import annotations

import pytest

from app.domain.errors import (
    ERROR_CATALOG,
    CapabilityError,
    ErrorCategory,
    ErrorCode,
    SuggestedAction,
)


def test_every_code_is_defined() -> None:
    assert set(ErrorCode) == set(ERROR_CATALOG)


def test_error_semantics_are_determined_by_code_alone() -> None:
    """raise 点无法覆盖重试语义——构造器只收 code 和 message。"""
    a = CapabilityError(ErrorCode.PROVIDER_TIMEOUT, "甲处触发")
    b = CapabilityError(ErrorCode.PROVIDER_TIMEOUT, "乙处触发")
    assert (a.retryable, a.suggested_action, a.http_status) == (
        b.retryable,
        b.suggested_action,
        b.http_status,
    )


@pytest.mark.parametrize("code", list(ErrorCode))
def test_http_status_is_a_sane_error_status(code: ErrorCode) -> None:
    assert 400 <= ERROR_CATALOG[code].http_status <= 599


@pytest.mark.parametrize("code", list(ErrorCode))
def test_description_is_present(code: ErrorCode) -> None:
    """错误码表要能当参考文档用，空描述等于没登记。"""
    assert ERROR_CATALOG[code].description.strip()


@pytest.mark.parametrize("code", list(ErrorCode))
def test_retryable_implies_retry_or_manual_review(code: ErrorCode) -> None:
    """可重试的错误不应建议「修请求」，那是自相矛盾的指引。"""
    definition = ERROR_CATALOG[code]
    if definition.retryable:
        assert definition.suggested_action in (
            SuggestedAction.RETRY,
            SuggestedAction.MANUAL_REVIEW,
        )


@pytest.mark.parametrize("code", list(ErrorCode))
def test_client_and_governance_errors_are_never_retryable(code: ErrorCode) -> None:
    """请求不合法或被治理规则拦截时重试无意义，只会放大成本。"""
    definition = ERROR_CATALOG[code]
    if definition.category in (ErrorCategory.CLIENT, ErrorCategory.GOVERNANCE):
        assert definition.retryable is False


@pytest.mark.parametrize("code", list(ErrorCode))
def test_governance_errors_always_require_a_human(code: ErrorCode) -> None:
    """设计文档 3.6：人工审核是一等能力。合规与预算问题不能由系统自行放行。"""
    definition = ERROR_CATALOG[code]
    if definition.category is ErrorCategory.GOVERNANCE:
        assert definition.suggested_action in (
            SuggestedAction.MANUAL_REVIEW,
            SuggestedAction.CONTACT_ADMIN,
        )


def test_resource_errors_are_404() -> None:
    for code, definition in ERROR_CATALOG.items():
        if definition.category is ErrorCategory.RESOURCE:
            assert definition.http_status == 404, code


def test_conflict_errors_are_4xx_conflict_family() -> None:
    for code, definition in ERROR_CATALOG.items():
        if definition.category is ErrorCategory.CONFLICT:
            assert definition.http_status in (402, 409), code


def test_unknown_code_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="not a valid"):
        CapabilityError(ErrorCode("nonexistent_code"), "x")
