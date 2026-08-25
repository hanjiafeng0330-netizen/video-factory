"""FastAPI 应用装配。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.capabilities import router as capabilities_router
from app.api.schemas import ErrorBody
from app.domain.errors import CapabilityError, ErrorCode
from app.workflow.jobs import InMemoryJobStore

# 错误码 → HTTP 状态码。集中一处映射，避免每个路由自己决定状态码导致对外行为不一致。
_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_PARAMETERS: 422,
    ErrorCode.CAPABILITY_NOT_FOUND: 404,
    ErrorCode.JOB_NOT_FOUND: 404,
    ErrorCode.INVALID_INPUT_REFS: 422,
    ErrorCode.MISSING_REQUIRED_PROMPT: 422,
    ErrorCode.UNKNOWN_PROMPT_SUPPLIED: 422,
    ErrorCode.JOB_NOT_CANCELLABLE: 409,
    ErrorCode.JOB_NOT_RETRYABLE: 409,
}

# 新增错误码时必须同时决定它的 HTTP 语义，否则会静默退化成 500。
_unmapped = set(ErrorCode) - _STATUS_BY_CODE.keys()
if _unmapped:
    raise RuntimeError(f"以下错误码未映射 HTTP 状态码：{sorted(_unmapped)}")


def create_app() -> FastAPI:
    app = FastAPI(title="营销视频智能生产系统", version="0.1.0")
    app.state.job_store = InMemoryJobStore()

    @app.exception_handler(CapabilityError)
    async def _handle_capability_error(_: Request, exc: CapabilityError) -> JSONResponse:
        body = ErrorBody(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            suggested_action=exc.suggested_action,
        )
        return JSONResponse(
            status_code=_STATUS_BY_CODE.get(exc.code, 500),
            content=body.model_dump(),
        )

    app.include_router(capabilities_router)
    return app
