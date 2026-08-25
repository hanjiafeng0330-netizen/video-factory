"""开发用测试页的后端。

**这不是 T2-5 的运营工作台。** 它是一个只在 dev/test 环境挂载的手动验收工具，
目的是让人能用浏览器把「上传一条视频 → 看到镜头切分、关键帧、血缘」这条链走一遍，
而不必写脚本。真正的工作台有权限、审核、版本对比、任务视图，属于 M2。

刻意保留的取舍：
- 上传直接同步跑完整条链。真实工作台必须走异步任务（一条 60 秒视频要几十秒），
  但那需要 T2-1 的任务中心；在这里同步跑能让手动验收看到最直接的结果。
- 资产内容通过本接口直读。生产必须走对象存储的签名链接（设计文档 11.1），
  本地文件系统实现没有真正的签名机制。
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from app.api.deps import container_of
from app.domain.assets import RightsStatus
from app.domain.capability import CapabilityRequest
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.hot_video import SourcePlatform
from app.domain.media import ShotList
from app.domain.refs import ArtifactRef
from app.domain.shot_script import build_shot_script, render_for_analysis
from app.domain.transcript import Transcript

router = APIRouter(prefix="/dev", tags=["dev-ui"])

_PAGE = Path(__file__).parent / "dev_ui.html"

# 手动验收工具，但仍然设个上限：没有上限的话一次误传能把临时目录写满，
# 而写满磁盘的症状和代码错误完全不同，排查会绕远路。
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@router.get("", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))


@router.post("/analyze")
async def analyze(
    request: Request,
    file: Annotated[UploadFile, File()],
    source_platform: Annotated[str, Form()] = SourcePlatform.DOUYIN.value,
    rights_status: Annotated[str, Form()] = RightsStatus.REFERENCE_ONLY.value,
    registered_by: Annotated[str, Form()] = "dev_tester",
    selection_reason: Annotated[str, Form()] = "开发环境手动验收",
    shot_sensitivity: Annotated[float, Form()] = 0.5,
    transcribe: Annotated[bool, Form()] = True,
    language: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """上传一条视频，跑完入库 + 预处理，返回可视化所需的全部数据。"""
    container = container_of(request)
    workdir = Path(tempfile.mkdtemp(prefix="vf_upload_"))
    try:
        target = workdir / (Path(file.filename or "upload.mp4").name or "upload.mp4")
        written = 0
        with target.open("wb") as sink:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise CapabilityError(
                        ErrorCode.INVALID_PARAMETERS,
                        f"文件超过 {_MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限",
                    )
                sink.write(chunk)

        ingest = container.capabilities.get("hot_video_ingest")
        run_id = uuid.uuid4().hex[:12]
        ingest_result = await ingest.execute(
            CapabilityRequest(
                parameters={
                    "file_path": str(target),
                    "source_platform": source_platform,
                    "rights_status": rights_status,
                    "registered_by": registered_by,
                    "selection_reason": selection_reason,
                    "mime_type": file.content_type or "video/mp4",
                },
                idempotency_key=f"dev-ingest-{run_id}",
            )
        )
        hot_ref = ingest_result.output_refs[0]

        preprocess = container.capabilities.get("video_preprocess")
        pre_result = await preprocess.execute(
            CapabilityRequest(
                input_refs=(hot_ref,),
                parameters={"shot_sensitivity": shot_sensitivity},
                idempotency_key=f"dev-preprocess-{run_id}",
            )
        )
        out_ref = pre_result.output_refs[0]

        transcript_ref = None
        transcript_notes: tuple[str, ...] = ()
        transcript_error: str | None = None
        if transcribe:
            try:
                asr = container.capabilities.get("audio_transcribe")
                asr_result = await asr.execute(
                    CapabilityRequest(
                        input_refs=(out_ref,),
                        parameters={"language": language or None},
                        idempotency_key=f"dev-transcribe-{run_id}",
                    )
                )
                transcript_ref = asr_result.output_refs[0]
                transcript_notes = asr_result.notes
            except CapabilityError as exc:
                # 转写失败不应让整条链失败：镜头切分的结果仍然有价值，
                # 而且「转写失败」和「确实无人说话」必须能区分开。
                transcript_error = f"{exc.code}: {exc.message}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    body = dict(container.artifacts.get(out_ref).body)
    hot_body = dict(container.artifacts.get(hot_ref).body)

    # 镜头脚本是 (预处理, 转写) 的纯函数计算视图，不落成产物——存下来就会出现
    # 「上游改了但视图还是旧的」这种无声漂移。
    transcript = (
        container.artifacts.get(transcript_ref).body_as(Transcript)
        if transcript_ref is not None
        else None
    )
    script = build_shot_script(
        ShotList.model_validate(body["shots"]),
        transcript,
        keyframe_asset_ids=tuple(body["keyframe_asset_ids"]),
        keyframe_at_ms=tuple(body["keyframe_at_ms"]),
    )

    return {
        "hot_video": {"ref": str(hot_ref), "body": hot_body, "notes": ingest_result.notes},
        "preprocess": {
            "ref": str(out_ref),
            "metadata": body["metadata"],
            "shots": body["shots"]["shots"],
            "audio_asset_id": body["audio_asset_id"],
            "keyframe_asset_ids": body["keyframe_asset_ids"],
            "keyframe_at_ms": body["keyframe_at_ms"],
            "truncated_keyframes": body["truncated_keyframes"],
            "notes": pre_result.notes,
        },
        "transcript": {
            "ref": str(transcript_ref) if transcript_ref else None,
            "error": transcript_error,
            "notes": transcript_notes,
            "language": transcript.language if transcript else None,
            "model": transcript.model_name if transcript else None,
            "line_count": len(transcript.lines) if transcript else 0,
        },
        "shot_script": {
            "has_transcript": script.has_transcript,
            "speech_ratio": script.speech_ratio,
            "silent_shot_count": script.silent_shot_count,
            "entries": [
                {
                    "index": entry.index,
                    "start_ms": entry.start_ms,
                    "end_ms": entry.end_ms,
                    "duration_ms": entry.duration_ms,
                    "keyframe_asset_id": entry.keyframe_asset_id,
                    "keyframe_at_ms": entry.keyframe_at_ms,
                    "visual_description": entry.visual_description,
                    "speech_ratio": entry.speech_ratio,
                    "is_silent": entry.is_silent,
                    "lines": [
                        {
                            "text": line.text,
                            "start_ms": line.start_ms,
                            "end_ms": line.end_ms,
                            "spans_shot_boundary": line.spans_shot_boundary,
                        }
                        for line in entry.lines
                    ],
                }
                for entry in script.entries
            ],
            "rendered": render_for_analysis(script),
        },
        "lineage": [
            {
                "ref": str(node.ref),
                "depth": node.depth,
                "relation": node.relation_from_child,
            }
            for node in container.lineage.trace(transcript_ref or out_ref)
        ],
        "versions": {
            "hot_video": len(container.artifacts.history(hot_ref.type, hot_ref.id)),
            "preprocess": len(container.artifacts.history(out_ref.type, out_ref.id)),
        },
    }


@router.get("/assets/{asset_id}")
async def asset_content(asset_id: str, request: Request) -> FileResponse:
    container = container_of(request)
    asset = container.assets.get(asset_id)
    return FileResponse(container.assets.open_path(asset_id), media_type=asset.mime_type)


@router.get("/artifacts/{artifact_type}/{artifact_id}")
async def artifact_history(
    artifact_type: str, artifact_id: str, request: Request
) -> list[dict[str, Any]]:
    """版本历史。手动验收「重跑产生新版本而不是覆盖」时用得上。"""
    container = container_of(request)
    from app.domain.refs import ArtifactType

    versions = container.artifacts.history(ArtifactType(artifact_type), artifact_id)
    return [
        {
            "ref": str(version.ref),
            "status": version.status,
            "stale": version.stale.reason if version.stale else None,
            "created_at": version.created_at.isoformat(),
            "created_by": version.created_by,
            "sources": [str(source) for source in version.sources],
        }
        for version in versions
    ]


@router.get("/capabilities")
async def capabilities(request: Request) -> list[dict[str, Any]]:
    """已注册能力及其所需提示词。

    配置管理台（T2-6）会读同一份数据来渲染「各环节用了哪些提示词」。
    """
    container = container_of(request)
    return [
        {
            "name": spec.name,
            "stage": spec.stage,
            "summary": spec.summary,
            "accepts": list(spec.accepts),
            "produces": list(spec.produces),
            "required_prompts": [
                {"key": prompt.key, "purpose": prompt.purpose} for prompt in spec.required_prompts
            ],
        }
        for spec in (capability.spec() for capability in container.capabilities.all())
    ]


def parse_ref(raw: str) -> ArtifactRef:
    """`type:id@vN` → ArtifactRef。页面回传引用时用。"""
    from app.domain.refs import ArtifactType

    head, _, version = raw.partition("@v")
    artifact_type, _, artifact_id = head.partition(":")
    return ArtifactRef(type=ArtifactType(artifact_type), id=artifact_id, version=int(version))
