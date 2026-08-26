"""开发用测试页的后端 - 步骤化重构版。

把视频分析流程拆成4个独立步骤，每步可单独触发和重跑：
1. 上传视频
2. 预处理（镜头切分 + 关键帧提取）
3. 语音转写（可选）
4. 视频理解（可选，调用LLM）
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
from app.domain.shot_script import ShotScript, build_shot_script, render_for_analysis
from app.domain.transcript import Transcript

router = APIRouter(prefix="/dev", tags=["dev-ui"])

_PAGE = Path(__file__).parent / "dev_ui.html"
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024


@router.get("", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))


# ============================================================================
# 步骤1：上传视频
# ============================================================================
@router.post("/step1/upload")
async def step1_upload(
    request: Request,
    file: Annotated[UploadFile, File()],
    source_platform: Annotated[str, Form()] = SourcePlatform.DOUYIN.value,
    rights_status: Annotated[str, Form()] = RightsStatus.REFERENCE_ONLY.value,
    registered_by: Annotated[str, Form()] = "dev_tester",
    selection_reason: Annotated[str, Form()] = "开发环境手动验收",
) -> dict[str, Any]:
    """步骤1：上传视频并入库。"""
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
        hot_body = dict(container.artifacts.get(hot_ref).body)

        return {
            "hot_video": {
                "ref": str(hot_ref),
                "body": hot_body,
                "notes": ingest_result.notes,
            }
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ============================================================================
# 步骤2：预处理（镜头切分 + 关键帧提取 + 语音转写）
# ============================================================================
@router.post("/step2/preprocess")
async def step2_preprocess(
    request: Request,
    hot_video_ref: Annotated[str, Form()],
    shot_sensitivity: Annotated[float, Form()] = 0.5,
    run_transcription: Annotated[bool, Form()] = True,
    language: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """步骤2：预处理，包括镜头切分、关键帧提取、音轨提取、语音转写。"""
    container = container_of(request)

    # 解析引用
    hot_ref = _parse_ref(hot_video_ref)

    # 调用预处理能力（镜头切分 + 关键帧 + 音轨）
    preprocess = container.capabilities.get("video_preprocess")
    run_id = uuid.uuid4().hex[:12]
    pre_result = await preprocess.execute(
        CapabilityRequest(
            input_refs=(hot_ref,),
            parameters={"shot_sensitivity": shot_sensitivity},
            idempotency_key=f"dev-preprocess-{run_id}",
        )
    )
    out_ref = pre_result.output_refs[0]
    body = dict(container.artifacts.get(out_ref).body)

    # 如果有音轨且要求转写，则调用 ASR
    transcript = None
    transcript_ref = None
    transcript_notes: tuple[str, ...] = ()
    transcript_error: str | None = None

    if run_transcription and body.get("audio_asset_id"):
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
            transcript = container.artifacts.get(transcript_ref).body_as(Transcript)
            transcript_notes = asr_result.notes
        except CapabilityError as exc:
            transcript_error = f"{exc.code}: {exc.message}"

    # 构建镜头脚本（带转写）
    script = build_shot_script(
        ShotList.model_validate(body["shots"]),
        transcript,
        keyframes=tuple(tuple(f) for f in body["keyframes"]),
        keyframe_timestamps=tuple(tuple(t) for t in body["keyframe_timestamps"]),
        frames_per_shot=body.get("frames_per_shot", 0),
    )

    result = {
        "preprocess": {
            "ref": str(out_ref),
            "metadata": body["metadata"],
            "shots": body["shots"]["shots"],
            "audio_asset_id": body["audio_asset_id"],
            "keyframes": body["keyframes"],
            "keyframe_timestamps": body["keyframe_timestamps"],
            "frames_per_shot": body.get("frames_per_shot", 0),
            "truncated_shots": body.get("truncated_shots", False),
            "notes": pre_result.notes,
        },
        "transcript": {
            "ref": str(transcript_ref) if transcript_ref else None,
            "error": transcript_error,
            "notes": transcript_notes,
            "language": transcript.language if transcript else None,
            "model": transcript.model_name if transcript else None,
            "line_count": len(transcript.lines) if transcript else 0,
        } if run_transcription else None,
        "shot_script": {
            "has_transcript": script.has_transcript,
            "speech_ratio": script.speech_ratio,
            "silent_shot_count": script.silent_shot_count,
            "frames_per_shot": body.get("frames_per_shot", 0),
            "entries": [
                {
                    "index": entry.index,
                    "start_ms": entry.start_ms,
                    "end_ms": entry.end_ms,
                    "duration_ms": entry.duration_ms,
                    "keyframe_asset_ids": entry.keyframe_asset_ids,
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
    }

    return result


# ============================================================================
# 步骤3：视频理解（可选，调用LLM）
# ============================================================================
@router.post("/step3/video_understand")
async def step3_video_understand(
    request: Request,
    preprocess_ref: Annotated[str, Form()],
) -> dict[str, Any]:
    """步骤3：视频理解，调用LLM分析每个镜头。"""
    container = container_of(request)

    # 解析引用
    pre_ref = _parse_ref(preprocess_ref)

    # 从提示词注册表获取提示词
    shot_visual_prompt = container.prompts.resolve(
        "video_understand.shot_visual",
        variables={
            "shot_index": "0",  # 占位符，实际值在能力内部替换
            "start_ms": "0",
            "end_ms": "0",
            "duration_ms": "0",
            "spoken_text": "",
            "platform": "douyin",
        },
    )

    # 调用视频理解能力
    vu = container.capabilities.get("video_understand")
    run_id = uuid.uuid4().hex[:12]
    vu_result = await vu.execute(
        CapabilityRequest(
            input_refs=(pre_ref,),
            resolved_prompts=(shot_visual_prompt,),
            idempotency_key=f"dev-vu-{run_id}",
        )
    )
    video_understand_ref = vu_result.output_refs[0]

    # 获取带画面描述的镜头脚本（来自 video_understand）
    vu_body = container.artifacts.get(video_understand_ref).body
    vu_script = ShotScript.model_validate(vu_body)

    # 获取原始预处理产物，以获取镜头/关键帧/时间戳信息
    pre_body = container.artifacts.get(pre_ref).body

    # 查找是否有转写产物（可能被步骤2创建）
    transcript = None
    try:
        from app.domain.refs import ArtifactType
        for relation in container.lineage.relations_out_of(pre_ref):
            if relation.target.type == ArtifactType.TRANSCRIPT:
                transcript = container.artifacts.get(relation.target).body_as(Transcript)
                break
    except Exception:
        pass

    # 用原始预处理数据重新构建镜头脚本（包含台词）
    script = build_shot_script(
        ShotList.model_validate(pre_body["shots"]),
        transcript,
        keyframes=tuple(tuple(f) for f in pre_body["keyframes"]),
        keyframe_timestamps=tuple(tuple(t) for t in pre_body["keyframe_timestamps"]),
        frames_per_shot=pre_body.get("frames_per_shot", 0),
    )

    # 将视频理解的画面描述回填到脚本中
    for entry, vu_entry in zip(script.entries, vu_script.entries):
        if vu_entry.visual_description:
            object.__setattr__(entry, 'visual_description', vu_entry.visual_description)

    return {
        "video_understand": {
            "ref": str(video_understand_ref),
            "total_shots": len(script.entries),
            "shots_with_visual_description": sum(
                1 for e in script.entries if e.visual_description
            ),
            "notes": vu_result.notes,
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
                    "keyframe_asset_ids": entry.keyframe_asset_ids,
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
    }


# ============================================================================
# 辅助接口
# ============================================================================
@router.get("/assets/{asset_id}")
async def asset_content(asset_id: str, request: Request) -> FileResponse:
    container = container_of(request)
    asset = container.assets.get(asset_id)
    return FileResponse(container.assets.open_path(asset_id), media_type=asset.mime_type)


@router.get("/capabilities")
async def capabilities(request: Request) -> list[dict[str, Any]]:
    """已注册能力及其所需提示词。"""
    container = container_of(request)
    return [
        {
            "name": spec.name,
            "stage": spec.stage,
            "summary": spec.summary,
            "accepts": list(spec.accepts),
            "produces": list(spec.produces),
            "required_prompts": [
                {"key": prompt.key, "purpose": prompt.purpose}
                for prompt in spec.required_prompts
            ],
        }
        for spec in (capability.spec() for capability in container.capabilities.all())
    ]


def _parse_ref(raw: str) -> "ArtifactRef":
    """`type:id@vN` → ArtifactRef。"""
    from app.domain.refs import ArtifactRef, ArtifactType

    head, _, version = raw.partition("@v")
    artifact_type, _, artifact_id = head.partition(":")
    return ArtifactRef(
        type=ArtifactType(artifact_type), id=artifact_id, version=int(version)
    )
