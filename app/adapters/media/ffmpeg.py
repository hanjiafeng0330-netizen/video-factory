"""ffmpeg / ffprobe 适配器。

实现 domain 里的媒体探测、抽音轨与抽帧三个协议。放在适配层是因为它包裹的是一个外部工具，业务
代码不应该知道 ffmpeg 的参数长什么样——换成别的工具或云端转码服务时，只换
这一个文件。

镜头切分**不在这里**：ffmpeg 的 scene 滤镜只比较亮度平面，会漏掉亮度接近而颜色
不同的硬切，原因与实测见 `scene_detect.py`。本文件只保留探测、抽音轨、抽帧。

`build_shot_list()` 留在本文件，因为它是与检测器无关的纯逻辑（切点 → 首尾相接的
镜头列表）；两个检测器实现共用它，以保证产出满足同样的不变量。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.domain.errors import CapabilityError, ErrorCode
from app.domain.media import MediaMetadata, Shot, ShotList

_FFPROBE = "ffprobe"
_FFMPEG = "ffmpeg"

# 短于此长度的片段不单独成镜头。快速剪辑里连续的几帧闪切会被 scene 滤镜识别成
# 多个镜头，那些「镜头」抽出来的关键帧没有分析价值，反而把帧数放大好几倍。
_MIN_SHOT_MS = 400


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            args,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
    except FileNotFoundError as exc:
        raise CapabilityError(
            ErrorCode.INTERNAL_ERROR, f"未找到可执行文件 {args[0]}，请确认已安装 ffmpeg"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise CapabilityError(ErrorCode.PROVIDER_TIMEOUT, f"{args[0]} 执行超时") from exc
    except subprocess.CalledProcessError as exc:
        raise CapabilityError(
            ErrorCode.STORAGE_FAILURE,
            f"{args[0]} 执行失败：{(exc.stderr or '').strip()[:400]}",
        ) from exc


def _parse_fps(raw: str | None) -> float:
    """ffprobe 的帧率是 `30000/1001` 这种分数形式。"""
    if not raw or raw == "0/0":
        return 0.0
    if "/" in raw:
        numerator, _, denominator = raw.partition("/")
        return float(numerator) / float(denominator) if float(denominator) else 0.0
    return float(raw)


class FfmpegMediaTool:
    """一个类同时实现探测、抽音轨、抽帧三个协议。

    合并是因为它们共享同一套子进程调用与错误映射；拆开只会让三处重复处理
    「ffmpeg 不存在 / 超时 / 返回非零」这三种情况。
    """

    def probe(self, path: Path) -> MediaMetadata:
        result = _run(
            [
                _FFPROBE,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]
        )
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if video is None:
            raise CapabilityError(
                ErrorCode.INVALID_PARAMETERS, f"文件不含视频流，无法作为热点视频处理：{path.name}"
            )

        duration_s = float(payload.get("format", {}).get("duration") or 0.0)
        return MediaMetadata(
            duration_ms=int(duration_s * 1000),
            width=int(video.get("width") or 0),
            height=int(video.get("height") or 0),
            fps=_parse_fps(video.get("avg_frame_rate")),
            video_codec=video.get("codec_name"),
            audio_codec=audio.get("codec_name") if audio else None,
            has_audio=audio is not None,
        )

    def extract_audio(self, source: Path, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        # 16k 单声道 wav：几乎所有 ASR 服务都要求或推荐这个格式，在这里统一转好，
        # 免得每接一家 ASR 供应商都自己转一遍。
        _run(
            [
                _FFMPEG,
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(target),
            ]
        )
        return target

    def extract_frame(self, source: Path, target: Path, *, at_ms: int) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        # -ss 放在 -i 之前是输入端定位，比输出端定位快几个数量级；
        # 对关键帧抽取来说这点精度损失无关紧要。
        _run(
            [
                _FFMPEG,
                "-y",
                "-ss",
                f"{at_ms / 1000:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(target),
            ]
        )
        if not target.is_file() or target.stat().st_size == 0:
            raise CapabilityError(
                ErrorCode.STORAGE_FAILURE, f"在 {at_ms}ms 处抽帧失败：{source.name}"
            )
        return target


def build_shot_list(cut_points_ms: list[int], duration_ms: int) -> ShotList:
    """把切点列表变成首尾相接、覆盖全片的镜头列表。

    独立成函数是为了让「切点 → 镜头」这段纯逻辑能脱离 ffmpeg 单独测试：
    切点重复、切点在 0ms、切点超出时长、末尾残留极短片段，这几种情况都出现过，
    而用真实视频去覆盖它们既慢又难以构造。
    """
    if duration_ms <= 0:
        raise CapabilityError(ErrorCode.INVALID_PARAMETERS, "视频时长为 0，无法切分镜头")

    boundaries = [0]
    for point in cut_points_ms:
        # 丢弃越界切点与过近切点。过近的切点来自快速闪切，单独成镜头没有分析价值。
        if 0 < point < duration_ms and point - boundaries[-1] >= _MIN_SHOT_MS:
            boundaries.append(point)

    # 末尾残片并回上一个镜头，而不是留一个几十毫秒的镜头。
    if duration_ms - boundaries[-1] < _MIN_SHOT_MS and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(duration_ms)

    shots = tuple(
        Shot(index=i, start_ms=boundaries[i], end_ms=boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    )
    # 检测不到切换时返回单个覆盖全片的镜头，而不是空列表——空列表会让下游到处判空。
    return ShotList(shots=shots)
