"""产物引用。

模块之间只传引用，不传内容。任何引用都必须带确切版本号——设计文档 8.3 要求
「禁止隐式读取最新版参与正式生产」，因此这里把 version 设为必填而非可选，
让「忘记指定版本」变成一个类型错误而不是一个线上事故。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ArtifactType(StrEnum):
    """可被能力模块引用的标准产物类型。"""

    HOT_VIDEO = "hot_video"
    MEDIA_ASSET = "media_asset"
    PREPROCESS_RESULT = "preprocess_result"
    TRANSCRIPT = "transcript"
    SHOT_SCRIPT = "shot_script"
    MARKETING_ANALYSIS = "marketing_analysis"
    VIDEO_ANALYSIS = "video_analysis"
    SCRIPT_PATTERN = "script_pattern"
    PRODUCT_PROFILE = "product_profile"
    SELLING_POINT_SET = "selling_point_set"
    MARKETING_SCRIPT = "marketing_script"
    STORYBOARD = "storyboard"
    VIDEO_JOB = "video_job"
    VIDEO_OUTPUT = "video_output"
    QUALITY_REPORT = "quality_report"


class ArtifactRef(BaseModel):
    """指向某个产物的某个确切版本。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ArtifactType
    id: str = Field(min_length=1, max_length=64)
    version: Annotated[int, Field(ge=1)]

    def __str__(self) -> str:
        return f"{self.type}:{self.id}@v{self.version}"
