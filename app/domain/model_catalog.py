"""可用模型目录。

配置管理台（T2-7）按环节选模型时读这份目录；配置校验也用它，防止把不具备某项
能力的模型选到那个位置上。

## 为什么必须做能力校验，而不是「配错了自然会报错」

实测发现：网关会为**纯文本模型静默丢掉图片输入**并照常返回 HTTP 200。
`deepseek-v4-pro` 收到一张纯红色图片加「请描述这张图片」，回答「我看不到图片」，
而 `input_tokens` 只有 100（Claude 带同一张图是 1222）——图片在链路上就被丢了。

如果把它配成视觉模型，后果不是报错，而是**产出一份看起来合理但完全凭空编造的画面
描述**。这类错会一路流进脚本模板和成片，且没有任何症状。所以能力标记必须在配置
加载时校验，不能指望运行时暴露。

## 倍率的用途

`cost_multiplier` 是相对计费倍率。它进目录是为了让 T7-3 的预算控制能在**任务创建
前**估算成本——设计文档 18 章要求「预算预估、任务上限、审批阈值」，而预估必须发生
在调用之前，事后统计拦不住已经花掉的钱。

区间型倍率（如 `1~4x`）按上界估算：预算控制宁可高估。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ModelAbility(StrEnum):
    TEXT = "text"
    VISION = "vision"
    IMAGE_GENERATION = "image_generation"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"
    """本项目实测确认过。"""

    ASSUMED = "assumed"
    """按同系列模型推断，未逐个实测。"""


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    family: str = Field(min_length=1, max_length=32)
    abilities: tuple[ModelAbility, ...]
    cost_multiplier_min: float = Field(ge=0)
    cost_multiplier_max: float = Field(ge=0)
    self_hosted: bool = False
    vision_verification: VerificationStatus = VerificationStatus.ASSUMED
    note: str | None = None

    def supports(self, ability: ModelAbility) -> bool:
        return ability in self.abilities

    @property
    def budget_multiplier(self) -> float:
        """预算估算用上界。宁可高估，低估会让成本上限失效。"""
        return self.cost_multiplier_max


_TEXT_VISION = (ModelAbility.TEXT, ModelAbility.VISION)
_TEXT_ONLY = (ModelAbility.TEXT,)


def _spec(
    model_id: str,
    family: str,
    abilities: tuple[ModelAbility, ...],
    low: float,
    high: float | None = None,
    *,
    self_hosted: bool = False,
    verification: VerificationStatus = VerificationStatus.ASSUMED,
    note: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        family=family,
        abilities=abilities,
        cost_multiplier_min=low,
        cost_multiplier_max=high if high is not None else low,
        self_hosted=self_hosted,
        vision_verification=verification,
        note=note,
    )


MODEL_CATALOG: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        # --- Claude ---
        _spec("claude-fable-5", "claude", _TEXT_VISION, 10),
        _spec("claude-opus-4-6", "claude", _TEXT_VISION, 5),
        _spec("claude-opus-4-7", "claude", _TEXT_VISION, 5),
        _spec("claude-opus-4-8", "claude", _TEXT_VISION, 5),
        _spec("claude-opus-5", "claude", _TEXT_VISION, 5),
        _spec(
            "claude-sonnet-4-6",
            "claude",
            _TEXT_VISION,
            3,
            verification=VerificationStatus.VERIFIED,
            note="视觉已实测；本项目视觉分析的默认选择，倍率与效果的平衡点",
        ),
        _spec("claude-sonnet-5", "claude", _TEXT_VISION, 2.5),
        # --- DeepSeek ---
        _spec(
            "deepseek-v4-pro",
            "deepseek",
            _TEXT_ONLY,
            1.8,
            verification=VerificationStatus.VERIFIED,
            note="实测无视觉：带图请求返回 200 但图片被静默丢弃，模型答「我看不到图片」",
        ),
        _spec(
            "deepseek-v4-pro-zy",
            "deepseek",
            _TEXT_ONLY,
            1.8,
            self_hosted=True,
            note="自部署。视觉能力按同系列推断为无",
        ),
        # --- GPT ---
        _spec("gpt-5.4", "gpt", _TEXT_VISION, 1.5),
        _spec(
            "gpt-5.4-mini",
            "gpt",
            _TEXT_VISION,
            0.35,
            verification=VerificationStatus.VERIFIED,
            note="视觉已实测；倍率最低的可用视觉模型，适合大批量抽帧描述",
        ),
        _spec("gpt-5.5", "gpt", _TEXT_VISION, 2),
        _spec("gpt-5.6-luna", "gpt", _TEXT_VISION, 0.1, 0.15),
        _spec("gpt-5.6-sol", "gpt", _TEXT_VISION, 2, 2.5),
        _spec("gpt-5.6-terra", "gpt", _TEXT_VISION, 1.1, 1.6),
        _spec(
            "gpt-image-2",
            "gpt",
            (ModelAbility.IMAGE_GENERATION,),
            1,
            note="仅生图。不能用于文本或视觉理解位置",
        ),
        # --- Qwen ---
        _spec("qwen3.6-plus", "qwen", _TEXT_VISION, 1, 4),
        _spec("qwen3.7-max", "qwen", _TEXT_VISION, 3),
        _spec(
            "qwen3.7-plus",
            "qwen",
            _TEXT_VISION,
            1,
            3,
            verification=VerificationStatus.VERIFIED,
            note="视觉已实测",
        ),
    )
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError:
        raise ValueError(f"未知模型 {model_id}，可用模型：{sorted(MODEL_CATALOG)}") from None


def models_with(ability: ModelAbility) -> tuple[ModelSpec, ...]:
    """按能力筛选。配置管理台的下拉选项用它，避免把不支持的模型摆出来让人选。"""
    return tuple(
        sorted(
            (spec for spec in MODEL_CATALOG.values() if spec.supports(ability)),
            key=lambda spec: spec.cost_multiplier_max,
        )
    )


def require_ability(model_id: str, ability: ModelAbility) -> ModelSpec:
    spec = get_model(model_id)
    if not spec.supports(ability):
        hint = ", ".join(m.id for m in models_with(ability)[:5])
        raise ValueError(
            f"模型 {model_id} 不具备 {ability} 能力"
            + (f"（{spec.note}）" if spec.note else "")
            + f"。可选：{hint} 等"
        )
    return spec
