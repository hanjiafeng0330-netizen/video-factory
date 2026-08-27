"""开发用测试页的后端 - 步骤化重构版。

把视频分析流程拆成3个独立步骤，每步可单独触发和重跑：
1. 上传视频
2. 预处理（镜头切分 + 关键帧提取 + 语音转写）
3. 视频理解（可选，调用LLM）
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

from app.api.deps import container_of
from app.api.xlsx_export import build_analysis_workbook
from app.capabilities.registry import CapabilityRegistry
from app.domain.assets import RightsStatus
from app.domain.capability import CapabilityRequest
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.hot_video import SourcePlatform
from app.domain.media import ShotList
from app.domain.model_catalog import ModelAbility, models_with, require_ability
from app.domain.refs import ArtifactRef, ArtifactType
from app.domain.shot_script import ShotScript, build_shot_script, render_for_analysis
from app.domain.transcript import Transcript

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dev", tags=["dev-ui"])

_PAGE = Path(__file__).parent / "dev_ui.html"
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024
_ANALYSES_DIR = Path(".local_analyses")  # 分析结果持久化目录


def _init_analyses_dir() -> None:
    """初始化分析结果目录。"""
    _ANALYSES_DIR.mkdir(exist_ok=True)


def _save_analysis(analysis_id: str, data: dict[str, Any]) -> Path:
    """保存分析结果到磁盘。"""
    _init_analyses_dir()
    path = _ANALYSES_DIR / f"{analysis_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return path


def _load_analysis(analysis_id: str) -> dict[str, Any] | None:
    """加载分析结果。"""
    path = _ANALYSES_DIR / f"{analysis_id}.json"
    if path.exists():
        return cast(dict[str, Any], json.loads(path.read_text()))
    return None


def _llm_capabilities_for_request(request: Request, *, zcode_plan_key: str) -> CapabilityRegistry:
    """选择仅用于本次 LLM 调用的 key，绝不写入能力请求或分析结果。"""
    container = container_of(request)
    if zcode_plan_key == "5688":
        configured_key = container.settings.llm.api_key
        if configured_key is None:
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "默认 LLM API key 未配置")
        api_key = configured_key.get_secret_value()
    elif not zcode_plan_key:
        raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "请填写 zcode plan key")
    else:
        api_key = zcode_plan_key
    return container.llm_capabilities_for(api_key)


def _list_analyses() -> list[dict[str, Any]]:
    """列出所有分析结果。"""
    _init_analyses_dir()
    results = []
    for path in sorted(_ANALYSES_DIR.glob("*.json"), reverse=True):
        # 跳过 macOS 的 AppleDouble 文件
        if path.name.startswith("._"):
            continue
        data = json.loads(path.read_text())
        results.append(
            {
                "id": path.stem,
                "created_at": data.get("created_at", ""),
                "video_name": data.get("video_name", ""),
                "hot_video_ref": data.get("hot_video", {}).get("ref", ""),
                "shot_count": len(data.get("shot_script", {}).get("entries", [])),
                "has_transcript": data.get("shot_script", {}).get("has_transcript", False),
                "has_visual_description": data.get("video_understand", {}).get("ref") is not None,
            }
        )
    return results


@router.get("", response_class=HTMLResponse)
async def page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"))


# ============================================================================
# 热点视频库
# ============================================================================
@router.post("/hot-videos/upload")
async def upload_hot_video_to_library(
    request: Request,
    file: Annotated[UploadFile, File()],
    source_platform: Annotated[str, Form()] = SourcePlatform.DOUYIN.value,
    rights_status: Annotated[str, Form()] = RightsStatus.REFERENCE_ONLY.value,
    registered_by: Annotated[str, Form()] = "library_user",
    selection_reason: Annotated[str, Form()] = "热点视频库上传",
) -> dict[str, Any]:
    """从热点视频库上传并入库，成功后不自动启动后续分析。"""
    container = container_of(request)
    workdir = Path(tempfile.mkdtemp(prefix="vf_library_upload_"))
    try:
        target = workdir / (Path(file.filename or "upload.mp4").name or "upload.mp4")
        written = 0
        with target.open("wb") as sink:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "文件超过 500MB 上限")
                sink.write(chunk)
        result = await container.capabilities.get("hot_video_ingest").execute(
            CapabilityRequest(
                parameters={
                    "file_path": str(target),
                    "original_filename": Path(file.filename or "upload.mp4").name,
                    "source_platform": source_platform,
                    "rights_status": rights_status,
                    "registered_by": registered_by,
                    "selection_reason": selection_reason,
                    "mime_type": file.content_type or "video/mp4",
                },
                idempotency_key=f"library-ingest-{uuid.uuid4().hex[:12]}",
            )
        )
        ref = result.output_refs[0]
        return {"ref": str(ref), "notes": result.notes}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _analysis_package_summary(analysis_id: str, data: dict[str, Any]) -> dict[str, Any]:
    stages = {
        "preprocess": bool((data.get("preprocess") or {}).get("ref")),
        "video_understand": bool((data.get("video_understand") or {}).get("ref")),
        "marketing_analysis": bool((data.get("marketing_analysis") or {}).get("ref")),
    }
    status = (
        "marketing_complete"
        if stages["marketing_analysis"]
        else "video_understand_complete"
        if stages["video_understand"]
        else "preprocess_complete"
    )
    return {
        "id": analysis_id,
        "created_at": data.get("created_at", ""),
        "video_name": data.get("video_name", ""),
        "shot_count": len((data.get("shot_script") or {}).get("entries", [])),
        "stages": stages,
        "status": status,
    }


@router.get("/hot-videos")
async def list_hot_videos(request: Request) -> list[dict[str, Any]]:
    """热点视频库：逻辑视频、入库版本与关联分析包摘要。"""
    container = container_of(request)
    latest_by_id: dict[str, Any] = {}
    for version in container.artifacts.list_by_type(ArtifactType.HOT_VIDEO):
        latest_by_id.setdefault(version.id, version)

    packages_by_hot: dict[str, list[dict[str, Any]]] = {}
    for session in _list_analyses():
        data = _load_analysis(session["id"])
        if data:
            hot_ref = (data.get("hot_video") or {}).get("ref", "")
            packages_by_hot.setdefault(hot_ref, []).append(
                _analysis_package_summary(session["id"], data)
            )

    videos: list[dict[str, Any]] = []
    for version in latest_by_id.values():
        body = dict(version.body)
        ref = str(version.ref)
        packages = packages_by_hot.get(ref, [])
        has_preprocess = any(p.get("stages", {}).get("preprocess", False) for p in packages)
        has_understand = any(p.get("stages", {}).get("video_understand", False) for p in packages)
        has_marketing = any(p.get("stages", {}).get("marketing_analysis", False) for p in packages)
        if has_marketing:
            processing_state = "market_analyzed"
        elif has_understand:
            processing_state = "understood"
        elif has_preprocess:
            processing_state = "preprocessed"
        else:
            processing_state = "unprocessed"
        videos.append(
            {
                "ref": ref,
                "logical_id": version.id,
                "version": version.version,
                "original_filename": body.get("original_filename", version.id),
                "mime_type": container.assets.get(str(body.get("asset_id"))).mime_type
                if body.get("asset_id")
                else "video/mp4",
                "created_at": version.created_at.isoformat(),
                "source_platform": body.get("source_platform", ""),
                "author": body.get("author"),
                "source_url": body.get("source_url"),
                "tags": body.get("tags", []),
                "rights_status": body.get("rights_status"),
                "selection_reason": body.get("selection_reason"),
                "asset_id": body.get("asset_id"),
                "analysis_packages": packages,
                "processing_state": processing_state,
                "available_actions": [
                    action
                    for action, allowed in (
                        ("preprocess", not has_preprocess),
                        ("video_understand", has_preprocess and not has_understand),
                        ("marketing_analysis", has_understand and not has_marketing),
                    )
                    if allowed
                ],
            }
        )
    # 未处理优先；同一状态内按上传/登记时间倒序。
    videos.sort(key=lambda item: item["created_at"], reverse=True)
    videos.sort(key=lambda item: item["processing_state"] != "unprocessed")
    return videos


# ============================================================================
# 分析历史与导出
# ============================================================================
@router.get("/analyses")
async def list_analyses(request: Request) -> list[dict[str, Any]]:
    """列出所有分析结果。"""
    return _list_analyses()


@router.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str, request: Request) -> dict[str, Any]:
    """获取指定分析的详细数据。"""
    data = _load_analysis(analysis_id)
    if data is None:
        raise CapabilityError(ErrorCode.ARTIFACT_NOT_FOUND, f"分析结果不存在：{analysis_id}")
    return data


@router.get("/analyses/{analysis_id}/export.xlsx")
async def export_analysis_xlsx(analysis_id: str, request: Request) -> Response:
    """导出完整视频分析包：视频分析 + 营销分析两个 Sheet，嵌入关键帧图片。"""
    data = _load_analysis(analysis_id)
    if data is None:
        raise CapabilityError(ErrorCode.ARTIFACT_NOT_FOUND, f"分析结果不存在：{analysis_id}")
    content = build_analysis_workbook(data, container_of(request).assets)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="video_analysis_{analysis_id}.xlsx"'
        },
    )


# ============================================================================
# 可复用结果目录
# ============================================================================
@router.get("/preprocess-results")
async def list_preprocess_results(request: Request) -> list[dict[str, Any]]:
    container = container_of(request)
    results: list[dict[str, Any]] = []
    for session in _list_analyses():
        data = _load_analysis(session["id"])
        if not data:
            continue
        pre = data.get("preprocess", {})
        ref = pre.get("ref")
        if not ref:
            continue
        try:
            container.artifacts.get(_parse_ref(ref))
            available = True
        except CapabilityError:
            available = False
        results.append(
            {
                "analysis_id": session["id"],
                "preprocess_ref": ref,
                "transcript_ref": (data.get("transcript") or {}).get("ref"),
                "video_name": data.get("video_name", ""),
                "created_at": data.get("created_at", ""),
                "shot_count": len(data.get("shot_script", {}).get("entries", [])),
                "available": available,
            }
        )
    return results


@router.get("/video-understand-results")
async def list_video_understand_results(request: Request) -> list[dict[str, Any]]:
    container = container_of(request)
    results: list[dict[str, Any]] = []
    for session in _list_analyses():
        data = _load_analysis(session["id"])
        if not data:
            continue
        vu = data.get("video_understand") or {}
        ref = vu.get("ref")
        if not ref:
            continue
        try:
            script = container.artifacts.get(_parse_ref(ref)).body_as(ShotScript)
            available = True
            has_transcript = script.has_transcript
            coverage = sum(1 for entry in script.entries if entry.visual_description)
        except CapabilityError:
            available = False
            has_transcript = False
            coverage = 0
        results.append(
            {
                "analysis_id": session["id"],
                "shot_script_ref": ref,
                "preprocess_ref": (data.get("preprocess") or {}).get("ref"),
                "video_name": data.get("video_name", ""),
                "created_at": data.get("created_at", ""),
                "available": available,
                "has_transcript": has_transcript,
                "visual_coverage": coverage,
            }
        )
    return results


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
                    "original_filename": Path(file.filename or "upload.mp4").name,
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
        "analysis_id": run_id,
        "created_at": __import__("datetime").datetime.now().isoformat(),
        "video_name": str(hot_ref.id),
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
        }
        if run_transcription
        else None,
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
        "hot_video": {"ref": str(hot_ref)},
        "video_understand": {"ref": None},
    }

    # 保存分析结果
    _save_analysis(run_id, result)

    return result


# ============================================================================
# 步骤3：视频理解（可选，调用LLM）
# ============================================================================
@router.post("/step3/video_understand")
async def step3_video_understand(
    request: Request,
    preprocess_ref: Annotated[str, Form()],
    analysis_id: Annotated[str, Form()] = "",
    vision_model: Annotated[str, Form()] = "",
    transcript_ref: Annotated[str, Form()] = "",
    zcode_plan_key: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """步骤3：视频理解，调用LLM分析每个镜头。"""
    container = container_of(request)

    # 解析引用
    pre_ref = _parse_ref(preprocess_ref)
    selected_model = vision_model or container.settings.llm.vision_model
    try:
        require_ability(selected_model, ModelAbility.VISION)
    except ValueError as exc:
        raise CapabilityError(ErrorCode.INVALID_PARAMETERS, str(exc)) from exc

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

    # 调用视频理解能力。用户可以选择步骤2关联的 exact transcript 版本。
    selected_transcript_ref = _parse_ref(transcript_ref) if transcript_ref else None
    input_refs = (pre_ref,) + ((selected_transcript_ref,) if selected_transcript_ref else ())
    vu = _llm_capabilities_for_request(
        request,
        zcode_plan_key=zcode_plan_key,
    ).get("video_understand")
    run_id = uuid.uuid4().hex[:12]
    vu_result = await vu.execute(
        CapabilityRequest(
            input_refs=input_refs,
            parameters={"vision_model": selected_model},
            resolved_prompts=(shot_visual_prompt,),
            idempotency_key=f"dev-vu-{run_id}",
        )
    )
    video_understand_ref = vu_result.output_refs[0]

    # 此为 canonical artifact：包含 selected transcript 的台词与视觉描述，
    # 页面展示、CSV 导出和营销分析都必须使用同一份内容。
    script = container.artifacts.get(video_understand_ref).body_as(ShotScript)
    pre_body = container.artifacts.get(pre_ref).body
    analysis_id = analysis_id or pre_body.get("analysis_id", "")

    result = {
        "analysis_id": analysis_id,
        "video_understand": {
            "ref": str(video_understand_ref),
            "total_shots": len(script.entries),
            "shots_with_visual_description": sum(1 for e in script.entries if e.visual_description),
            "notes": vu_result.notes,
        },
        "shot_script": {
            "has_transcript": script.has_transcript,
            "speech_ratio": script.speech_ratio,
            "silent_shot_count": script.silent_shot_count,
            "frames_per_shot": pre_body.get("frames_per_shot", 0),
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

    # 更新已保存的分析结果
    if analysis_id:
        saved_data = _load_analysis(analysis_id)
        if saved_data:
            saved_data["video_understand"] = {
                "ref": str(video_understand_ref),
                "total_shots": len(script.entries),
                "shots_with_visual_description": sum(
                    1 for e in script.entries if e.visual_description
                ),
                "notes": list(vu_result.notes),
            }
            saved_data["shot_script"] = result["shot_script"]
            _save_analysis(analysis_id, saved_data)

    return result


# ============================================================================
# 步骤4：营销分析（可选，调用 LLM 提取营销口径）
# ============================================================================
@router.post("/step4/marketing_analysis")
async def step4_marketing_analysis(
    request: Request,
    shot_script_ref: Annotated[str, Form()],
    analysis_id: Annotated[str, Form()] = "",
    text_model: Annotated[str, Form()] = "",
    zcode_plan_key: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """步骤 4：营销分析，调用 LLM 提取营销口径（钩子/痛点/卖点结构/视觉风格）。"""
    container = container_of(request)

    # 解析引用与模型能力校验
    ss_ref = _parse_ref(shot_script_ref)
    selected_model = text_model or container.settings.llm.text_model
    try:
        require_ability(selected_model, ModelAbility.TEXT)
    except ValueError as exc:
        raise CapabilityError(ErrorCode.INVALID_PARAMETERS, str(exc)) from exc

    # 获取提示词
    marketing_prompt = container.prompts.resolve(
        "marketing_analysis.prompt",
        variables={"shot_script": ""},  # 占位符，实际值在能力内部填充
    )

    # 调用营销分析能力
    ma = _llm_capabilities_for_request(
        request,
        zcode_plan_key=zcode_plan_key,
    ).get("marketing_analysis")
    run_id = uuid.uuid4().hex[:12]
    ma_result = await ma.execute(
        CapabilityRequest(
            input_refs=(ss_ref,),
            parameters={"text_model": selected_model},
            resolved_prompts=(marketing_prompt,),
            idempotency_key=f"dev-ma-{run_id}",
        )
    )
    ma_ref = ma_result.output_refs[0]

    # 获取分析结果
    ma_body = container.artifacts.get(ma_ref).body

    # 构建响应
    result = {
        "analysis_id": analysis_id,
        "marketing_analysis": {
            "ref": str(ma_ref),
            "model": selected_model,
            "hook": ma_body.get("hook", ""),
            "pain_points": ma_body.get("pain_points", ""),
            "selling_point_structure": ma_body.get("selling_point_structure", ""),
            "visual_style": ma_body.get("visual_style", ""),
            "notes": ma_body.get("notes", ""),
            "raw_output": ma_body.get("raw_output", ""),
        },
    }

    # 更新已保存的分析结果
    if analysis_id:
        saved_data = _load_analysis(analysis_id)
        if saved_data:
            saved_data["marketing_analysis"] = result["marketing_analysis"]
            _save_analysis(analysis_id, saved_data)

    return result


# ============================================================================
# 模型目录（步骤3/步骤4的下拉选项）
# ============================================================================
@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """按能力返回可选模型，前端不展示不匹配的模型。"""

    def serialize(ability: ModelAbility) -> list[dict[str, Any]]:
        return [
            {
                "id": spec.id,
                "family": spec.family,
                "multiplier": spec.budget_multiplier,
                "verification": spec.vision_verification,
                "note": spec.note,
            }
            for spec in models_with(ability)
        ]

    return {
        "vision_models": serialize(ModelAbility.VISION),
        "text_models": serialize(ModelAbility.TEXT),
    }


# ============================================================================
# 提示词配置接口
# ============================================================================
@router.get("/prompts")
async def list_prompts(request: Request) -> dict[str, Any]:
    """列出所有提示词及其版本历史。"""
    container = container_of(request)
    prompts_data = []
    for template in container.prompts.templates():
        versions = []
        for v in container.prompts.history(template.key):
            versions.append(
                {
                    "version": v.version,
                    "status": v.status,
                    "body": v.body,
                    "change_note": v.change_note,
                    "author": v.author,
                    "created_at": v.created_at.isoformat(),
                    "activated_at": v.activated_at.isoformat() if v.activated_at else None,
                }
            )
        prompts_data.append(
            {
                "key": template.key,
                "stage": template.stage,
                "purpose": template.purpose,
                "variables": list(template.variables),
                "versions": versions,
            }
        )
    return {"prompts": prompts_data}


@router.post("/prompts/{prompt_key}/activate")
async def activate_prompt(
    prompt_key: str,
    request: Request,
    version: Annotated[int, Form()],
    actor: Annotated[str, Form()] = "web_user",
) -> dict[str, Any]:
    """激活指定版本的提示词。"""
    container = container_of(request)
    activated = container.prompts.activate(prompt_key, version, actor=actor)
    return {
        "key": prompt_key,
        "version": activated.version,
        "status": activated.status,
        "activated_at": activated.activated_at.isoformat() if activated.activated_at else None,
    }


@router.post("/prompts/add_version")
async def add_prompt_version(
    request: Request,
    key: Annotated[str, Form()],
    body: Annotated[str, Form()],
    change_note: Annotated[str, Form()],
    author: Annotated[str, Form()] = "web_user",
) -> dict[str, Any]:
    """添加新版本的提示词（草案状态）。"""
    container = container_of(request)
    version = container.prompts.add_version(key, body, change_note=change_note, author=author)
    return {
        "key": key,
        "version": version.version,
        "status": version.status,
        "change_note": version.change_note,
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
                {"key": prompt.key, "purpose": prompt.purpose} for prompt in spec.required_prompts
            ],
        }
        for spec in (capability.spec() for capability in container.capabilities.all())
    ]


def _parse_ref(raw: str) -> ArtifactRef:
    """`type:id@vN` → ArtifactRef。"""
    from app.domain.refs import ArtifactRef, ArtifactType

    head, _, version = raw.partition("@v")
    artifact_type, _, artifact_id = head.partition(":")
    return ArtifactRef(type=ArtifactType(artifact_type), id=artifact_id, version=int(version))
