"""能力注册表。

从「模块级单例」改成「按依赖构建」：视频类能力需要仓储、资产存储和媒体适配器，
这些必须在组合根（`app.api.app`）装配后注入。能力模块自己不认识任何具体实现，
只认 domain 里的协议——这既满足「能力模块不得 import app.storage / app.adapters」
的边界契约，也让同一个能力在测试里可以用假实现驱动。

注册表位于 `app.capabilities` 包内但不属于任何单个能力，因此不受「能力模块互不
依赖」契约约束——它只做汇总，不在能力之间建立依赖。

新增能力时在 `build_capabilities()` 里加一行即可，路由、schema、任务通路自动生效。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.capabilities.echo.capability import EchoCapability
from app.capabilities.ingest_hot_video.capability import IngestHotVideoCapability
from app.capabilities.marketing_analysis.capability import MarketingAnalysisCapability
from app.capabilities.preprocess_video.capability import PreprocessVideoCapability
from app.capabilities.transcribe.capability import TranscribeCapability
from app.capabilities.video_understand.capability import VideoUnderstandCapability
from app.domain.assets import AssetStore
from app.domain.capability import Capability
from app.domain.errors import CapabilityError, ErrorCode
from app.domain.llm import LLMClient
from app.domain.media import AudioExtractor, FrameExtractor, MediaProbe, ShotDetector
from app.domain.prompt_registry import PromptRegistry
from app.domain.transcript import SpeechRecognizer
from app.domain.versioning import ArtifactRepository


@dataclass(frozen=True)
class CapabilityDeps:
    """能力模块的全部外部依赖。

    集中成一个对象而不是散在各处的参数，是为了让「新增一种依赖」在类型上可见——
    否则很容易出现某个能力偷偷从全局取了个单例，那会让它无法独立测试。
    """

    artifacts: ArtifactRepository
    assets: AssetStore
    probe: MediaProbe
    audio: AudioExtractor
    shots: ShotDetector
    frames: FrameExtractor
    recognizer: SpeechRecognizer
    llm: LLMClient
    prompts: PromptRegistry
    vision_model: str
    text_model: str


class CapabilityRegistry:
    def __init__(self, capabilities: tuple[Capability[Any], ...]) -> None:
        by_name = {c.name: c for c in capabilities}
        if len(by_name) != len(capabilities):
            raise RuntimeError("能力名称重复，注册表构建失败")
        self._by_name = by_name
        self._ordered = capabilities

    def all(self) -> tuple[Capability[Any], ...]:
        return self._ordered

    def get(self, name: str) -> Capability[Any]:
        try:
            return self._by_name[name]
        except KeyError:
            raise CapabilityError(ErrorCode.CAPABILITY_NOT_FOUND, f"未注册的能力：{name}") from None


def build_capabilities(deps: CapabilityDeps) -> CapabilityRegistry:
    return CapabilityRegistry(
        (
            EchoCapability(),
            IngestHotVideoCapability(
                artifacts=deps.artifacts, assets=deps.assets, probe=deps.probe
            ),
            PreprocessVideoCapability(
                artifacts=deps.artifacts,
                assets=deps.assets,
                probe=deps.probe,
                audio=deps.audio,
                shots=deps.shots,
                frames=deps.frames,
            ),
            TranscribeCapability(
                artifacts=deps.artifacts,
                assets=deps.assets,
                recognizer=deps.recognizer,
            ),
            VideoUnderstandCapability(
                artifacts=deps.artifacts,
                assets=deps.assets,
                llm=deps.llm,
                prompts=deps.prompts,
                vision_model=deps.vision_model,
            ),
            MarketingAnalysisCapability(
                artifacts=deps.artifacts,
                assets=deps.assets,
                llm=deps.llm,
                prompts=deps.prompts,
                text_model=deps.text_model,
            ),
        )
    )
