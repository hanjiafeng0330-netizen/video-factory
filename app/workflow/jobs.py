"""异步任务通路的进程内实现。

**这是 T0-2 的占位实现，T2-1 会用数据库支撑的任务中心替换它。** 它存在的唯一
目的是让统一能力协议的异步分支在 M0 阶段就真实可跑（而不是留个 TODO），因此
刻意不实现持久化、重试退避、并发控制和完整状态机——那些是 T2-1 的交付物，
提前在内存里写一版只会造成之后要推翻的资产。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from app.domain.capability import Capability, CapabilityRequest
from app.domain.errors import CapabilityError, ErrorCode, SuggestedAction
from app.domain.jobs import JobRecord, JobStatus


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        # 持有强引用，否则事件循环可能在任务完成前把它回收掉。
        self._running: set[asyncio.Task[None]] = set()

    def get(self, job_id: str) -> JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise CapabilityError(
                ErrorCode.JOB_NOT_FOUND,
                f"任务不存在：{job_id}",
                retryable=False,
                suggested_action=SuggestedAction.FIX_INPUT,
            ) from None

    def submit(self, capability: Capability[Any], request: CapabilityRequest) -> JobRecord:
        now = _now()
        job = JobRecord(
            id=f"job_{uuid.uuid4().hex[:16]}",
            capability=capability.name,
            status=JobStatus.QUEUED,
            request=request,
            created_at=now,
            updated_at=now,
        )
        self._jobs[job.id] = job
        self._spawn(capability, job)
        return job

    def cancel(self, job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job.status.is_terminal:
            raise CapabilityError(
                ErrorCode.JOB_NOT_CANCELLABLE,
                f"任务 {job_id} 已处于终态 {job.status}，无法取消",
                retryable=False,
                suggested_action=SuggestedAction.FIX_INPUT,
            )
        job.status = JobStatus.CANCELLED
        job.updated_at = _now()
        return job

    def retry(self, capability: Capability[Any], job_id: str) -> JobRecord:
        job = self.get(job_id)
        if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise CapabilityError(
                ErrorCode.JOB_NOT_RETRYABLE,
                f"任务 {job_id} 当前状态为 {job.status}，只有 FAILED 或 CANCELLED 可重试",
                retryable=False,
                suggested_action=SuggestedAction.FIX_INPUT,
            )
        job.status = JobStatus.RETRYING
        job.attempt += 1
        job.error_code = None
        job.error_message = None
        job.updated_at = _now()
        self._spawn(capability, job)
        return job

    def _spawn(self, capability: Capability[Any], job: JobRecord) -> None:
        task = asyncio.create_task(self._execute(capability, job))
        self._running.add(task)
        task.add_done_callback(self._running.discard)

    async def _execute(self, capability: Capability[Any], job: JobRecord) -> None:
        if job.status is JobStatus.CANCELLED:
            return
        job.status = JobStatus.RUNNING
        job.updated_at = _now()
        try:
            job.result = await capability.execute(job.request)
        except CapabilityError as exc:
            job.status = JobStatus.FAILED
            job.error_code = exc.code
            job.error_message = exc.message
        else:
            job.status = JobStatus.SUCCEEDED
        job.updated_at = _now()

    async def drain(self) -> None:
        """等待所有在跑的任务结束。测试用，生产环境不依赖它。"""
        while self._running:
            await asyncio.gather(*tuple(self._running), return_exceptions=True)
