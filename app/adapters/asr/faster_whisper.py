"""语音转写适配器（faster-whisper，本地推理）。

**为什么一期默认用本地模型而不是云 ASR。** 热点视频分析是内部动作，且需要反复
重跑（换灵敏度重切、改提示词重新理解都会连带重跑）。云 ASR 按时长计费，会让
「重跑一次」变成一件要花钱的事，于是没人敢重跑——而设计文档 3.1 明确要求独立重跑。
另外本地推理不外传素材，直接规避了设计文档 13.3 关于外部素材使用范围的一类风险。

云 ASR 仍然会需要（长素材、更高准确率、更好的中文标点），因此它走同一个
`SpeechRecognizer` 协议接入，通过配置切换，不影响任何业务代码。

模型是**懒加载**的：首次转写时才下载与初始化。放在构造函数里会让整个进程启动
被一次几百 MB 的下载卡住，而多数请求根本不做转写。

**繁简归一化。** whisper 对中文常输出繁体（实测「很多人以為洗臉要用熱水」）。
对大陆营销素材这是错的：转写会流进 LLM 分析和人工审核，繁体字形会让脚本模板里
出现混杂字形，人工审核也会被迫先做一遍字形转换。这里用 OpenCC 做确定性转换，
**不是**用 initial_prompt 去引导模型——后者会改变模型行为且属于提示词，按项目规则
必须先经业务评审再落库；而繁简转换是纯文本归一化，可测且不影响识别结果本身。

素材确实来自港台平台时应当关掉归一化，因此它是参数而不是硬编码。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from app.domain.errors import CapabilityError, ErrorCode
from app.domain.transcript import TranscriptLine

# small 是中文短视频口播的性价比拐点：tiny/base 在中文上错字明显，medium 以上
# 在 CPU 上慢到不实用。需要更高准确率时应当切云 ASR，而不是在本地堆模型尺寸。
_DEFAULT_MODEL = "small"

# int8 在 Apple Silicon CPU 上比 float32 快数倍，中文口播的可懂度差异可忽略。
_DEFAULT_COMPUTE_TYPE = "int8"


class FasterWhisperRecognizer:
    """实现 `app.domain.transcript.SpeechRecognizer`。"""

    def __init__(
        self,
        model_size: str = _DEFAULT_MODEL,
        *,
        device: str = "cpu",
        compute_type: str = _DEFAULT_COMPUTE_TYPE,
        normalize_to_simplified: bool = True,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._normalize_to_simplified = normalize_to_simplified
        self._converter: Any = None
        self._model: Any = None
        # 多个请求并发触发首次加载时，只允许一个线程真正下载与初始化。
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        suffix = "+t2s" if self._normalize_to_simplified else ""
        # 归一化影响转写文本，因此必须体现在模型标识里，否则无法回答
        # 「这条脚本依据的转写是否做过繁简转换」。
        return f"faster-whisper:{self._model_size}{suffix}"

    def _normalize(self, text: str) -> str:
        if not self._normalize_to_simplified or not text:
            return text
        if self._converter is None:
            from opencc import OpenCC

            self._converter = OpenCC("t2s")
        return str(self._converter.convert(text))

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:  # pragma: no cover
                    raise CapabilityError(
                        ErrorCode.INTERNAL_ERROR,
                        "未安装 faster-whisper，无法进行本地转写",
                    ) from exc
                self._model = WhisperModel(
                    self._model_size, device=self._device, compute_type=self._compute_type
                )
        return self._model

    def transcribe(
        self, audio_path: Path, *, language: str | None = None
    ) -> tuple[tuple[TranscriptLine, ...], str, str]:
        if not audio_path.is_file():
            raise CapabilityError(ErrorCode.INVALID_PARAMETERS, f"音频文件不存在：{audio_path}")

        model = self._ensure_model()
        try:
            segments, info = model.transcribe(
                str(audio_path),
                language=language,
                # VAD 过滤掉静音段，避免模型在长静音上产生幻觉重复句
                # （whisper 系列的典型失败模式）。
                vad_filter=True,
                beam_size=5,
            )
            raw = list(segments)
        except Exception as exc:
            raise CapabilityError(
                ErrorCode.MODEL_OUTPUT_UNPARSEABLE, f"转写失败：{audio_path.name}（{exc}）"
            ) from exc

        lines: list[TranscriptLine] = []
        for segment in raw:
            text = self._normalize(str(segment.text).strip())
            if not text:
                continue
            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)
            if end_ms <= start_ms:
                # VAD 边界偶尔产出零长片段，给它一个最小可见时长而不是丢掉——
                # 丢掉会让转写出现无声的空洞。
                end_ms = start_ms + 1
            lines.append(
                TranscriptLine(
                    index=len(lines),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text=text,
                    confidence=_probability_of(segment),
                )
            )

        return (
            tuple(lines),
            str(getattr(info, "language", None) or language or "unknown"),
            (self.model_name),
        )


def _to_simplified(text: str) -> str:
    """繁体转简体。已是简体、英文、数字均不受影响。"""
    from opencc import OpenCC

    return str(OpenCC("t2s").convert(text))


def _probability_of(segment: Any) -> float | None:
    """faster-whisper 给的是平均对数概率，转成 0~1 便于在审核页展示。"""
    raw = getattr(segment, "avg_logprob", None)
    if raw is None:
        return None
    import math

    return max(0.0, min(1.0, math.exp(float(raw))))
