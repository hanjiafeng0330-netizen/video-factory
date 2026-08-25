"""统一能力路由（设计文档 8.1）。

所有能力共用这一套路由，新增能力不需要新增任何 endpoint。API 层只做转换与聚合，
不含业务逻辑，也不直接触碰第三方适配层。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.schemas import (
    CapabilityListBody,
    ExecuteRequestBody,
    JobBody,
    ResultBody,
)
from app.capabilities.registry import get_capability, list_capabilities
from app.domain.capability import CapabilityRequest, CapabilitySpec
from app.domain.jobs import JobRecord
from app.workflow.jobs import InMemoryJobStore

router = APIRouter()


def _job_store(request: Request) -> InMemoryJobStore:
    store = request.app.state.job_store
    assert isinstance(store, InMemoryJobStore)
    return store


def _to_domain(body: ExecuteRequestBody) -> CapabilityRequest:
    return CapabilityRequest(
        input_refs=body.input_refs,
        parameters=body.parameters,
        resolved_prompts=body.resolved_prompts,
        idempotency_key=body.idempotency_key,
    )


def _to_job_body(job: JobRecord) -> JobBody:
    result = (
        ResultBody(
            output_refs=job.result.output_refs,
            metrics=job.result.metrics,
            notes=job.result.notes,
        )
        if job.result is not None
        else None
    )
    return JobBody(
        id=job.id,
        capability=job.capability,
        status=job.status,
        attempt=job.attempt,
        request_fingerprint=job.request.fingerprint(),
        result=result,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/capabilities", response_model=CapabilityListBody)
async def list_all() -> CapabilityListBody:
    """列出全部能力及其自描述。

    配置管理台（T2-6）用它来渲染「各环节分别使用了哪些提示词」。
    """
    return CapabilityListBody(capabilities=tuple(c.spec() for c in list_capabilities()))


@router.get("/capabilities/{capability}/schema", response_model=CapabilitySpec)
async def get_schema(capability: str) -> CapabilitySpec:
    return get_capability(capability).spec()


@router.post("/capabilities/{capability}/execute", response_model=ResultBody)
async def execute(capability: str, body: ExecuteRequestBody) -> ResultBody:
    """同步执行，用于预览、调试和短时任务（设计文档 3.1）。"""
    result = await get_capability(capability).execute(_to_domain(body))
    return ResultBody(
        output_refs=result.output_refs,
        metrics=result.metrics,
        notes=result.notes,
    )


@router.post("/capabilities/{capability}/jobs", response_model=JobBody, status_code=202)
async def submit_job(capability: str, body: ExecuteRequestBody, request: Request) -> JobBody:
    """异步提交，用于视频分析、批量生成等耗时操作（设计文档 3.1）。

    契约校验在提交时就做，避免把一个必然失败的请求排进队列。
    """
    cap = get_capability(capability)
    domain_request = _to_domain(body)
    cap.validate_request(domain_request)
    return _to_job_body(_job_store(request).submit(cap, domain_request))


@router.get("/jobs/{job_id}", response_model=JobBody)
async def get_job(job_id: str, request: Request) -> JobBody:
    return _to_job_body(_job_store(request).get(job_id))


@router.post("/jobs/{job_id}/cancel", response_model=JobBody)
async def cancel_job(job_id: str, request: Request) -> JobBody:
    return _to_job_body(_job_store(request).cancel(job_id))


@router.post("/jobs/{job_id}/retry", response_model=JobBody)
async def retry_job(job_id: str, request: Request) -> JobBody:
    store = _job_store(request)
    cap = get_capability(store.get(job_id).capability)
    return _to_job_body(store.retry(cap, job_id))
