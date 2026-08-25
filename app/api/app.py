"""FastAPI 应用装配。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.capabilities import router as capabilities_router
from app.api.schemas import ErrorBody
from app.domain.errors import CapabilityError
from app.workflow.jobs import InMemoryJobStore


def create_app() -> FastAPI:
    app = FastAPI(title="营销视频智能生产系统", version="0.1.0")
    app.state.job_store = InMemoryJobStore()

    @app.exception_handler(CapabilityError)
    async def _handle_capability_error(_: Request, exc: CapabilityError) -> JSONResponse:
        # 状态码与重试语义全部来自错误码表，API 层不做二次判断，
        # 避免出现「同一个错误在不同路由返回不同状态码」。
        body = ErrorBody(
            code=exc.code,
            category=exc.category,
            message=exc.message,
            retryable=exc.retryable,
            suggested_action=exc.suggested_action,
        )
        return JSONResponse(status_code=exc.http_status, content=body.model_dump())

    app.include_router(capabilities_router)
    return app
