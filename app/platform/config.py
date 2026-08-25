"""配置与密钥管理（设计文档 13.2、15.3）。

设计文档要求「各环境的数据、对象存储和第三方密钥必须隔离」。「隔离」如果只靠
「大家记得配对」，迟早会出现开发环境连到生产桶、或测试跑数据把生产库写脏。
这里把隔离做成启动时的硬校验：

1. **物理分离**：每个环境读各自的 `.env.{environment}`，密钥不共享一个文件；
2. **命名带环境**：库名与桶名必须带环境标识，配错了连不上而不是静默连对；
3. **生产守卫**：生产环境禁止 mock 供应商、禁止弱口令、必须设费用上限；
4. **日志脱敏**：密钥一律 `SecretStr`，`redacted()` 是唯一允许打印的形式。

第 3 条对应设计文档 15.3「生产环境：使用正式密钥、审计和费用限制」，第 4 条
对应 13.2「日志自动脱敏」。
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    """允许模拟供应商和小样本（设计文档 15.3）。"""

    TEST = "test"
    """使用隔离账号和固定回归数据。"""

    PROD = "prod"
    """使用正式密钥、审计和费用限制。"""

    @property
    def is_production(self) -> bool:
        return self is Environment.PROD


# 常见默认口令。生产环境命中任意一个即拒绝启动——这类口令通常是复制示例配置
# 时忘了改，而不是有意设置的。
_WEAK_SECRETS = frozenset(
    {
        "",
        "postgres",
        "password",
        "passwd",
        "changeme",
        "change-me",
        "secret",
        "minioadmin",
        "root",
        "admin",
        "test",
        "123456",
    }
)


def _reject_weak(env: Environment, field: str, secret: SecretStr) -> None:
    if env.is_production and secret.get_secret_value().lower() in _WEAK_SECRETS:
        raise ValueError(f"生产环境的 {field} 使用了默认或弱口令，拒绝启动")


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "video_factory"
    password: SecretStr = SecretStr("")
    name: str = "vf_dev"

    def dsn(self) -> str:
        pwd = self.password.get_secret_value()
        return f"postgresql://{self.user}:{pwd}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0

    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class ObjectStorageSettings(BaseModel):
    """对象存储。原始资产与生成资产分桶管理（设计文档 11.1）。"""

    endpoint: str = "http://localhost:9000"
    access_key: str = "minioadmin"
    secret_key: SecretStr = SecretStr("")
    bucket_raw: str = "vf-dev-raw"
    bucket_generated: str = "vf-dev-generated"
    signed_url_ttl_seconds: Annotated[int, Field(ge=60, le=86_400)] = 900


class VideoProviderSettings(BaseModel):
    """第三方视频生成供应商（设计文档 10 章）。

    一期只接一家，通过显式配置选择（设计文档 10.3）。
    """

    name: str = "mock"
    api_key: SecretStr | None = None
    timeout_seconds: Annotated[int, Field(ge=30, le=7200)] = 3600


class BudgetSettings(BaseModel):
    """成本上限（设计文档 18 章「成本失控」）。"""

    max_cost_per_video: Annotated[float, Field(ge=0)] = 0.0
    """单条成片成本上限。0 表示不限制，生产环境不允许为 0。"""

    max_cost_per_workflow_run: Annotated[float, Field(ge=0)] = 0.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VF_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    environment: Environment = Environment.DEV
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: ObjectStorageSettings = Field(default_factory=ObjectStorageSettings)
    video_provider: VideoProviderSettings = Field(default_factory=VideoProviderSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)

    @model_validator(mode="after")
    def _enforce_environment_isolation(self) -> Self:
        """库名与桶名必须带环境标识。

        这样配错环境的结果是「连不上」，而不是「静默连到了别的环境」。后者在
        开发环境写脏生产库之后才会被发现。
        """
        env = self.environment.value
        expected_db_prefix = f"vf_{env}"
        if not self.database.name.startswith(expected_db_prefix):
            raise ValueError(
                f"{env} 环境的库名必须以 {expected_db_prefix} 开头，实际为 {self.database.name}"
            )

        expected_bucket_prefix = f"vf-{env}-"
        for field, bucket in (
            ("bucket_raw", self.storage.bucket_raw),
            ("bucket_generated", self.storage.bucket_generated),
        ):
            if not bucket.startswith(expected_bucket_prefix):
                raise ValueError(
                    f"{env} 环境的 {field} 必须以 {expected_bucket_prefix} 开头，实际为 {bucket}"
                )

        if self.storage.bucket_raw == self.storage.bucket_generated:
            raise ValueError("原始资产与生成资产必须分桶（设计文档 11.1）")

        return self

    @model_validator(mode="after")
    def _enforce_production_guards(self) -> Self:
        if not self.environment.is_production:
            return self

        _reject_weak(self.environment, "database.password", self.database.password)
        _reject_weak(self.environment, "storage.secret_key", self.storage.secret_key)

        if self.video_provider.name == "mock":
            raise ValueError("生产环境禁止使用 mock 视频供应商")
        if self.video_provider.api_key is None:
            raise ValueError("生产环境必须配置视频供应商密钥")
        _reject_weak(self.environment, "video_provider.api_key", self.video_provider.api_key)

        if self.budget.max_cost_per_video <= 0:
            raise ValueError("生产环境必须设置单条成片成本上限（设计文档 15.3）")
        if self.budget.max_cost_per_workflow_run <= 0:
            raise ValueError("生产环境必须设置单次流程成本上限")

        return self

    def redacted(self) -> dict[str, Any]:
        """可安全写入日志的配置摘要。

        `SecretStr` 的 `repr` 已经是掩码，但这里显式只导出非密字段，避免将来
        有人给某个密钥字段用了普通 `str` 就悄悄泄进日志。
        """
        return {
            "environment": self.environment.value,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "user": self.database.user,
                "name": self.database.name,
            },
            "redis": {"host": self.redis.host, "port": self.redis.port, "db": self.redis.db},
            "storage": {
                "endpoint": self.storage.endpoint,
                "bucket_raw": self.storage.bucket_raw,
                "bucket_generated": self.storage.bucket_generated,
                "signed_url_ttl_seconds": self.storage.signed_url_ttl_seconds,
            },
            "video_provider": {
                "name": self.video_provider.name,
                "api_key_configured": self.video_provider.api_key is not None,
                "timeout_seconds": self.video_provider.timeout_seconds,
            },
            "budget": {
                "max_cost_per_video": self.budget.max_cost_per_video,
                "max_cost_per_workflow_run": self.budget.max_cost_per_workflow_run,
            },
        }


def env_file_for(environment: Environment, env_dir: Path | None = None) -> Path:
    """各环境的密钥文件物理分离，不共用一个 `.env`。"""
    base = env_dir if env_dir is not None else Path.cwd()
    return base / f".env.{environment.value}"


def load_settings(env_dir: Path | None = None) -> Settings:
    """按 `VF_ENVIRONMENT` 加载对应环境的配置文件。

    环境变量优先于文件，便于 CI 与容器注入密钥而不落盘。
    """
    environment = Environment(os.environ.get("VF_ENVIRONMENT", Environment.DEV.value))
    path = env_file_for(environment, env_dir)
    return Settings(
        _env_file=path if path.exists() else None,
        environment=environment,
    )
