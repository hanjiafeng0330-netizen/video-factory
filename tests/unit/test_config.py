"""T0-4 验收：三环境配置与密钥隔离守卫。

这些测试守住的不是「配置能读出来」，而是「配错了会被拒绝」——设计文档 15.3
要求各环境的数据、对象存储和第三方密钥必须隔离，而隔离只有在配错时会失败才
算真的存在。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from app.platform.config import (
    DatabaseSettings,
    Environment,
    ObjectStorageSettings,
    Settings,
    env_file_for,
)


def prod_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "environment": Environment.PROD,
        "database": {
            "user": "vf",
            "password": SecretStr("a-real-long-generated-secret"),
            "name": "vf_prod",
        },
        "storage": {
            "secret_key": SecretStr("another-real-generated-secret"),
            "bucket_raw": "vf-prod-raw",
            "bucket_generated": "vf-prod-generated",
        },
        "video_provider": {"name": "provider_a", "api_key": SecretStr("sk-live-xxxxxxxx")},
        "budget": {"max_cost_per_video": 15.0, "max_cost_per_workflow_run": 200.0},
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------- 环境隔离


def test_dev_defaults_are_valid() -> None:
    settings = Settings(environment=Environment.DEV)
    assert settings.database.name == "vf_dev"
    assert settings.video_provider.name == "mock"


def test_database_name_must_carry_environment() -> None:
    """配错环境应当连不上，而不是静默连到别的环境。"""
    with pytest.raises(ValidationError, match="库名必须以 vf_test 开头"):
        Settings(
            environment=Environment.TEST,
            database=DatabaseSettings(name="vf_dev"),
            storage=ObjectStorageSettings(
                bucket_raw="vf-test-raw", bucket_generated="vf-test-generated"
            ),
        )


def test_bucket_must_carry_environment() -> None:
    with pytest.raises(ValidationError, match="bucket_raw 必须以 vf-test- 开头"):
        Settings(
            environment=Environment.TEST,
            database=DatabaseSettings(name="vf_test"),
            storage=ObjectStorageSettings(
                bucket_raw="vf-prod-raw", bucket_generated="vf-test-generated"
            ),
        )


def test_raw_and_generated_buckets_must_differ() -> None:
    """设计文档 11.1：原始资产和生成资产分开管理。"""
    with pytest.raises(ValidationError, match="必须分桶"):
        Settings(
            environment=Environment.DEV,
            storage=ObjectStorageSettings(bucket_raw="vf-dev-same", bucket_generated="vf-dev-same"),
        )


def test_each_environment_reads_its_own_file() -> None:
    """密钥文件物理分离，不共用一个 .env。"""
    names = {env: env_file_for(env).name for env in Environment}
    assert names == {
        Environment.DEV: ".env.dev",
        Environment.TEST: ".env.test",
        Environment.PROD: ".env.prod",
    }
    assert len(set(names.values())) == len(Environment)


# --------------------------------------------------------------- 生产守卫


def test_valid_production_config_is_accepted() -> None:
    settings = Settings(**prod_kwargs())
    assert settings.environment.is_production


def test_production_rejects_mock_provider() -> None:
    with pytest.raises(ValidationError, match="禁止使用 mock"):
        Settings(**prod_kwargs(video_provider={"name": "mock"}))


def test_production_requires_provider_key() -> None:
    with pytest.raises(ValidationError, match="必须配置视频供应商密钥"):
        Settings(**prod_kwargs(video_provider={"name": "provider_a"}))


@pytest.mark.parametrize("weak", ["", "postgres", "changeme", "MinioAdmin", "123456"])
def test_production_rejects_weak_database_password(weak: str) -> None:
    """这类口令通常是复制示例配置忘了改，不是有意设置的。"""
    with pytest.raises(ValidationError, match="弱口令"):
        Settings(
            **prod_kwargs(database={"user": "vf", "password": SecretStr(weak), "name": "vf_prod"})
        )


def test_production_rejects_weak_storage_secret() -> None:
    with pytest.raises(ValidationError, match="弱口令"):
        Settings(
            **prod_kwargs(
                storage={
                    "secret_key": SecretStr("minioadmin"),
                    "bucket_raw": "vf-prod-raw",
                    "bucket_generated": "vf-prod-generated",
                }
            )
        )


@pytest.mark.parametrize(
    "budget",
    [
        {"max_cost_per_video": 0, "max_cost_per_workflow_run": 200.0},
        {"max_cost_per_video": 15.0, "max_cost_per_workflow_run": 0},
    ],
)
def test_production_requires_cost_ceilings(budget: dict[str, float]) -> None:
    """设计文档 15.3：生产环境必须有费用限制。"""
    with pytest.raises(ValidationError, match="成本上限"):
        Settings(**prod_kwargs(budget=budget))


def test_dev_may_omit_cost_ceilings() -> None:
    assert Settings(environment=Environment.DEV).budget.max_cost_per_video == 0.0


# --------------------------------------------------------------- 脱敏


def test_redacted_summary_contains_no_secret_values() -> None:
    """设计文档 13.2：日志自动脱敏。"""
    settings = Settings(**prod_kwargs())
    dumped = repr(settings.redacted())
    for secret in (
        "a-real-long-generated-secret",
        "another-real-generated-secret",
        "sk-live-xxxxxxxx",
    ):
        assert secret not in dumped
    assert settings.redacted()["video_provider"]["api_key_configured"] is True


def test_dsn_is_not_part_of_redacted_summary() -> None:
    """DSN 内嵌口令，绝不能出现在可打印摘要里。"""
    settings = Settings(**prod_kwargs())
    assert "a-real-long-generated-secret" in settings.database.dsn()
    assert "dsn" not in repr(settings.redacted())


def test_unknown_setting_is_rejected() -> None:
    """拼错的配置项应当报错，而不是被静默忽略后走默认值。"""
    with pytest.raises(ValidationError):
        Settings(environmnet="dev")  # type: ignore[call-arg]
