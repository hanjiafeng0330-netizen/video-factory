"""T0-4 验收：本地栈起来后集成测试可跑通。

这些测试验证的不是「基础设施能连上」这种废话，而是几条一期真正依赖的能力是否
就绪——缺任何一条，M1 的资产中心和 M3 的向量去重都会在实现到一半时卡住：

- PostgreSQL 的 pgvector 与 pg_trgm 扩展（一期不引入独立向量库和 ES）
- dev 与 test 使用**不同的库**（测试反复清库，共用会洗掉开发数据）
- Redis 可读写（任务队列与事件投递）
- MinIO 的原始桶与生成桶都已存在且分离（设计文档 11.1）

整个文件标记为 integration，默认 `pytest` 不会跑，需要 `pytest -m integration`。
"""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
import redis
from httpx import AsyncClient

from app.platform.config import Environment, Settings

pytestmark = pytest.mark.integration


def _stack_is_up() -> bool:
    """本地栈是否可达。

    跑不起来时整体跳过而不是报红：「中间件没起来」和「代码写错了」是两类问题，
    前者报成失败会淹没后者，久了就没人看集成测试的结果了。
    """
    import socket

    for host, port in (("localhost", 5432), ("localhost", 6379), ("localhost", 9000)):
        with socket.socket() as probe:
            probe.settimeout(1)
            if probe.connect_ex((host, port)) != 0:
                return False
    return True


if not _stack_is_up():  # pragma: no cover
    pytest.skip(
        "本地栈未启动。执行 docker compose up -d 后重试。",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def dev() -> Settings:
    return Settings(environment=Environment.DEV)


def _connect(settings: Settings, database: str) -> psycopg.Connection[tuple[Any, ...]]:
    return psycopg.connect(
        host=settings.database.host,
        port=settings.database.port,
        user=settings.database.user,
        password=settings.database.password.get_secret_value(),
        dbname=database,
        connect_timeout=5,
    )


# --------------------------------------------------------------- PostgreSQL


@pytest.mark.parametrize("database", ["vf_dev", "vf_test"])
def test_both_environment_databases_exist(dev: Settings, database: str) -> None:
    """dev 与 test 必须是不同的库，否则测试清库会洗掉开发数据。"""
    with _connect(dev, database) as conn, conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        row = cur.fetchone()
        assert row is not None
        assert row[0] == database


@pytest.mark.parametrize("database", ["vf_dev", "vf_test"])
@pytest.mark.parametrize("extension", ["vector", "pg_trgm"])
def test_required_extensions_are_installed(dev: Settings, database: str, extension: str) -> None:
    """一期用 pgvector + PG 全文顶住相似检索，不引入独立向量库与 ES。"""
    with _connect(dev, database) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (extension,))
        assert cur.fetchone() is not None, f"{database} 缺少扩展 {extension}"


def test_vector_type_is_usable(dev: Settings) -> None:
    """扩展装上了不等于能用，直接跑一次向量距离计算。"""
    with _connect(dev, "vf_dev") as conn, conn.cursor() as cur:
        cur.execute("SELECT '[1,0,0]'::vector <-> '[0,1,0]'::vector")
        row = cur.fetchone()
        assert row is not None
        assert float(row[0]) > 0


# --------------------------------------------------------------- Redis


def test_redis_read_write(dev: Settings) -> None:
    client = redis.Redis.from_url(dev.redis.url(), socket_connect_timeout=5)
    try:
        client.set("vf:selftest", "ok", ex=30)
        assert client.get("vf:selftest") == b"ok"
    finally:
        client.delete("vf:selftest")
        client.close()


# --------------------------------------------------------------- MinIO


async def test_minio_is_live(dev: Settings) -> None:
    async with AsyncClient(base_url=dev.storage.endpoint, timeout=5) as ac:
        resp = await ac.get("/minio/health/live")
        assert resp.status_code == 200


async def test_both_buckets_exist_and_differ(dev: Settings) -> None:
    """设计文档 11.1：原始资产和生成资产分开管理。"""
    assert dev.storage.bucket_raw != dev.storage.bucket_generated
    async with AsyncClient(base_url=dev.storage.endpoint, timeout=5) as ac:
        for bucket in (dev.storage.bucket_raw, dev.storage.bucket_generated):
            resp = await ac.head(f"/{bucket}")
            # 桶存在时 MinIO 对未签名请求返回 200 或 403（存在但无权限），
            # 桶不存在时返回 404。这里只需要区分「存在」与「不存在」。
            assert resp.status_code != 404, f"桶 {bucket} 不存在，请检查 minio_init 是否执行"
