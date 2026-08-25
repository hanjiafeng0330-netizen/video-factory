"""ASR 文本归一化。

whisper 对中文常输出繁体。转写会流进 LLM 分析和人工审核，繁体字形会让脚本模板里
出现混杂字形，审核也被迫先做一遍字形转换。所以在适配器出口做确定性归一化。
"""

from __future__ import annotations

import pytest

from app.adapters.asr.faster_whisper import FasterWhisperRecognizer


def test_traditional_is_converted() -> None:
    recognizer = FasterWhisperRecognizer("small")
    assert recognizer._normalize("很多人以為洗臉要用熱水") == "很多人以为洗脸要用热水"


def test_simplified_is_untouched() -> None:
    recognizer = FasterWhisperRecognizer("small")
    text = "这是简体中文，不应被改动"
    assert recognizer._normalize(text) == text


@pytest.mark.parametrize("text", ["Hello World", "123 456", "SPF50+", ""])
def test_non_chinese_is_untouched(text: str) -> None:
    assert FasterWhisperRecognizer("small")._normalize(text) == text


def test_normalization_can_be_disabled() -> None:
    """素材确实来自港台平台时应当保留繁体。"""
    recognizer = FasterWhisperRecognizer("small", normalize_to_simplified=False)
    assert recognizer._normalize("洗臉") == "洗臉"


def test_model_name_records_normalization() -> None:
    """归一化影响转写文本，必须体现在模型标识里。

    否则无法回答「这条脚本依据的转写是否做过繁简转换」。
    """
    assert FasterWhisperRecognizer("small").model_name == "faster-whisper:small+t2s"
    assert (
        FasterWhisperRecognizer("small", normalize_to_simplified=False).model_name
        == "faster-whisper:small"
    )
