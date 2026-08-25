"""T0-2 验收：统一能力接口协议的契约测试。

对应验收标准：
1. `GET /capabilities/{cap}/schema` 返回 JSON Schema 且包含所需提示词键位；
2. 同步与异步两种调用均通；
3. 声明了 required_prompts 的能力在缺少 resolved_prompts 时报错而非静默降级。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.bootstrap.dev import create_dev_app
from app.domain.jobs import JobStatus

PROMPT = {"key": "echo.system", "version": 1, "text": "PREFIX"}


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    app = create_dev_app(tmp_path / "assets")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def body(**overrides: Any) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "parameters": {"message": "hello", "repeat": 2},
        "resolved_prompts": [PROMPT],
        "idempotency_key": "echo-contract-001",
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------- schema


async def test_schema_exposes_parameters_and_prompt_keys(client: AsyncClient) -> None:
    resp = await client.get("/capabilities/echo/schema")
    assert resp.status_code == 200
    spec = resp.json()

    assert spec["name"] == "echo"
    assert spec["parameters_schema"]["properties"].keys() >= {"message", "repeat"}

    # 配置管理台依赖这一段来展示「这个环节用了哪些提示词」
    prompts = spec["required_prompts"]
    assert [p["key"] for p in prompts] == ["echo.system"]
    assert prompts[0]["purpose"]


async def test_capability_list_covers_every_registered_capability(client: AsyncClient) -> None:
    resp = await client.get("/capabilities")
    assert resp.status_code == 200
    names = [c["name"] for c in resp.json()["capabilities"]]
    assert "echo" in names


async def test_unknown_capability_returns_structured_error(client: AsyncClient) -> None:
    resp = await client.get("/capabilities/nope/schema")
    assert resp.status_code == 404
    assert resp.json() == {
        "code": "capability_not_found",
        "category": "resource",
        "message": "未注册的能力：nope",
        "retryable": False,
        "suggested_action": "fix_input",
    }


# --------------------------------------------------------------- 同步调用


async def test_sync_execute(client: AsyncClient) -> None:
    resp = await client.post("/capabilities/echo/execute", json=dict(body()))
    assert resp.status_code == 200
    result = resp.json()
    assert result["notes"] == ["PREFIX :: hello", "PREFIX :: hello"]
    assert result["metrics"]["lines"] == 2


async def test_sync_execute_rejects_unknown_parameter(client: AsyncClient) -> None:
    """坏参数必须以结构化 422 结束，而不是逃成 500。"""
    resp = await client.post(
        "/capabilities/echo/execute",
        json=dict(body(parameters={"message": "hi", "bogus": 1})),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_parameters"


async def test_async_job_rejects_bad_parameters_before_queueing(client: AsyncClient) -> None:
    resp = await client.post(
        "/capabilities/echo/jobs",
        json=dict(body(parameters={"message": "hi", "repeat": 999})),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_parameters"


# --------------------------------------------------------------- 异步调用


async def test_async_job_reaches_succeeded(client: AsyncClient) -> None:
    resp = await client.post("/capabilities/echo/jobs", json=dict(body()))
    assert resp.status_code == 202
    job = resp.json()
    assert job["status"] in (JobStatus.QUEUED, JobStatus.RUNNING)
    assert job["request_fingerprint"].startswith("sha256:")

    for _ in range(50):
        await asyncio.sleep(0.01)
        job = (await client.get(f"/jobs/{job['id']}")).json()
        if JobStatus(job["status"]).is_terminal:
            break

    assert job["status"] == JobStatus.SUCCEEDED
    assert job["result"]["notes"] == ["PREFIX :: hello", "PREFIX :: hello"]


async def test_async_job_validates_contract_before_queueing(client: AsyncClient) -> None:
    """必然失败的请求不应该先排进队列再失败。"""
    resp = await client.post("/capabilities/echo/jobs", json=dict(body(resolved_prompts=[])))
    assert resp.status_code == 422
    assert resp.json()["code"] == "missing_required_prompt"


async def test_cancel_then_retry(client: AsyncClient) -> None:
    job_id = (await client.post("/capabilities/echo/jobs", json=dict(body()))).json()["id"]

    cancelled = await client.post(f"/jobs/{job_id}/cancel")
    # 进程内执行极快，任务可能已经终态；两种结果都合法，但不能是 500。
    assert cancelled.status_code in (200, 409)

    if cancelled.status_code == 200:
        assert cancelled.json()["status"] == JobStatus.CANCELLED
        retried = await client.post(f"/jobs/{job_id}/retry")
        assert retried.status_code == 200
        assert retried.json()["attempt"] == 2


async def test_unknown_job_returns_404(client: AsyncClient) -> None:
    resp = await client.get("/jobs/job_does_not_exist")
    assert resp.status_code == 404
    assert resp.json()["code"] == "job_not_found"


# --------------------------------------------------------------- 提示词注入


async def test_missing_required_prompt_is_rejected(client: AsyncClient) -> None:
    """核心验收点：提示词缺失必须报错，不允许静默降级为空提示词。"""
    resp = await client.post("/capabilities/echo/execute", json=dict(body(resolved_prompts=[])))
    assert resp.status_code == 422
    payload = resp.json()
    assert payload["code"] == "missing_required_prompt"
    assert "echo.system" in payload["message"]
    assert payload["retryable"] is False
    assert payload["suggested_action"] == "fix_input"


async def test_undeclared_prompt_is_rejected(client: AsyncClient) -> None:
    """多传提示词通常意味着键位写错，静默忽略会让排查变成猜谜。"""
    resp = await client.post(
        "/capabilities/echo/execute",
        json=dict(body(resolved_prompts=[PROMPT, {**PROMPT, "key": "echo.typo"}])),
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "unknown_prompt_supplied"


async def test_prompt_version_is_mandatory(client: AsyncClient) -> None:
    """提示词必须带确切版本号，禁止隐式使用最新版。"""
    resp = await client.post(
        "/capabilities/echo/execute",
        json=dict(body(resolved_prompts=[{"key": "echo.system", "text": "PREFIX"}])),
    )
    assert resp.status_code == 422
