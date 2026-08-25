"""提示词草案登记。

**本文件里的提示词全部以 DRAFT 状态入库，跑不起来。** 激活需要业务评审后调用
`registry.activate(key, version, actor=...)`，那是一个会写审计的显式动作
（开发计划 1.3 节）。

把草案写进代码而不是直接写进数据库，是因为一期还没有配置管理台（T2-6）。等后台
上线后，正文的后续修订应当在后台进行——那时本文件只负责登记键位契约与初始草案。
"""

from __future__ import annotations

from app.domain.prompt_registry import PromptTemplate
from app.prompts.registry import InMemoryPromptRegistry

SHOT_VISUAL_KEY = "video_understand.shot_visual"
NARRATIVE_KEY = "video_understand.narrative"


SHOT_VISUAL_TEMPLATE = PromptTemplate(
    key=SHOT_VISUAL_KEY,
    stage="视频理解",
    purpose="基于镜头内多帧综合描述画面内容，关注动作过程与变化，只记录可见事实",
    variables=("shot_index", "start_ms", "end_ms", "duration_ms", "spoken_text", "platform"),
)

# 草案 v2：从单帧升级到多帧。评审要点见 docs/提示词草案_视频理解.md。
SHOT_VISUAL_BODY_V2 = """你在为营销视频分析系统描述**单个镜头**的画面内容。

## 输入
- N 张关键帧图片（**均匀分布在该镜头内**，不是首/中/尾，避开了转场）
- 镜头序号：{{ shot_index }}
- 镜头时间：{{ start_ms }}–{{ end_ms }}ms（时长 {{ duration_ms }}ms）
- 该镜头内的口播台词：{{ spoken_text }}
- 目标平台：{{ platform }}

## 任务
基于所有帧**综合描述画面内容，关注动作过程与变化**。不要逐帧分别描述；把 N 帧看
成一个镜头内连续动作的采样点，给出一个**整体描述**，并在「motion」字段里点出可
见的变化。

## 硬性要求
1. **综合描述，不逐帧分述。** 模型输出应当是「这段镜头里发生了什么」，不是「第
   1 帧…第 2 帧…」。逐帧分述会把动作切碎了，下游叙事还原会把一个连贯动作当成
   几个独立事件。
2. 只写你在帧里确实看到的。看不清就写「看不清」，不要补全。
3. **不要写出帧里没有出现的品牌名、产品名、成分名。** 即使台词提到了，画面上没
   有就不算。
4. 台词仅用于消解歧义（例如判断手里拿的是否就是台词提到的那个东西）。**不得把台
   词内容当作画面内容写进描述。**
5. 画面文字（字幕、贴纸、包装文字）原样摘录，不要改写、不要翻译、不要补全被遮挡
   的部分。任一帧里出现就算出现。
6. **不判断这个镜头「有没有效果」「是不是钩子」。** 那是下一环节的事。
7. **motion 字段只写可见的变化**，例如「人物从画面左侧走到右侧」「手势从摊开到握
   拳」「产品从画面外被拿进画面」。不可见的变化（心理、情绪）不写。

## confidence 的含义
对本次综合描述准确性的自评（基于 N 帧整体信息量）：
- 0.9 以上：所有帧画面清晰，主体、动作过程与要素都明确
- 0.6–0.9：主体清楚，但动作细节或文字不确定
- 0.6 以下：帧整体模糊、严重遮挡、或信息不足

宁可给低分。低置信度会让人工复核这一镜头，虚高的置信度会让错误描述直接流进下游。

## 输出
严格输出 JSON，不要加任何解释文字、不要用代码块包裹：

{
  "visual_summary": "一句话，60 字以内，综合 N 帧可见内容",
  "motion": "动作/变化描述，不超过 40 字；无明显变化写「无明显变化」",
  "people": [{"description": "外观与姿态", "count": 1}],
  "products": [{"appearance": "外观描述", "visible_text": "包装上可见文字，无则空字符串"}],
  "scene": "环境场景，如室内浴室、纯色背景、户外街道",
  "actions": ["综合 N 帧可见的动作，用完成时/进行时表述"],
  "on_screen_text": ["原样摘录的画面文字，任一帧出现即算出现"],
  "camera": {"shot_size": "特写|近景|中景|远景|不确定",
             "angle": "平视|俯视|仰视|不确定"},
  "legibility": "clear|partial|unreadable",
  "uncertainties": ["你不确定的具体内容"],
  "confidence": 0.0
}"""


NARRATIVE_TEMPLATE = PromptTemplate(
    key=NARRATIVE_KEY,
    stage="视频理解",
    purpose="基于镜头脚本还原叙事结构、创意机制与证据",
    variables=("shot_script", "duration_ms", "shot_count", "speech_ratio", "platform"),
)

NARRATIVE_BODY_V1 = """你在为营销视频分析系统还原一条热点视频的**叙事结构**。

## 输入

镜头脚本（每个镜头的时间、画面描述、台词）：

{{ shot_script }}

视频信息：总时长 {{ duration_ms }}ms，{{ shot_count }} 个镜头，
语音占比 {{ speech_ratio }}，目标平台 {{ platform }}

## 任务
判断这条视频「用什么结构讲了什么」，并为每一个判断给出证据。

## 硬性要求

1. **每个 role 判断必须指向具体证据**：镜头序号 + 引用到的台词原文或画面要素。
   拿不出证据的判断不要写。
2. **区分观测与推断。** `observed_facts` 只写镜头脚本里明确写着的内容；
   `segments` 里的判断属于推断，且每条都要能追回到 observed。
3. `role` 只能取以下封闭集合之一，**不要自创**：
   `hook`（抓注意力） / `pain`（点出问题） / `proof`（证明或演示） /
   `product`（介绍产品本身） / `emotion`（情绪或共鸣） /
   `cta`（推动行动） / `transition`（过渡）
4. 一个镜头只归属一个 role。连续且作用相同的镜头可以合并成一个 segment。
5. **不要判断这条视频「值得不值得模仿」，不要对播放量做归因。** 高互动不等于结构
   有效，那是运营结合人群与品类做的判断，不是你的。
6. `creative_mechanism` 写「这条视频靠什么机制起作用」，例如「反常识开场」
   「前后对比」「现场证明」「身份代入」。**只写你能从结构里看出来的，最多 4 条。**
7. `inference_notes` 写你的不确定之处，以及**可能的替代解读**。宁可写「也可能是
   X」——过度自信的单一解读会让下游把偶然当规律。
8. 如果某镜头的画面描述为「（待视觉理解）」或为空，只依据台词判断，并在
   `inference_notes` 里明确写出「镜头 N 缺画面信息」。

## 输出
严格输出 JSON，不要加任何解释文字、不要用代码块包裹：

{
  "segments": [
    {
      "shot_indexes": [0],
      "role": "hook",
      "purpose": "这一段在结构上承担什么作用，一句话",
      "evidence": {
        "quotes": ["直接引用的台词原文"],
        "visual_cues": ["引用的画面要素"]
      },
      "confidence": 0.0
    }
  ],
  "creative_mechanism": ["最多 4 条"],
  "hook": {
    "shot_indexes": [0],
    "how_it_works": "开头靠什么留住人，一句话",
    "relies_on": "visual|verbal|both"
  },
  "observed_facts": ["镜头脚本里明确写着的事实"],
  "inference_notes": ["不确定之处与替代解读"],
  "overall_confidence": 0.0
}"""


def seed_prompt_drafts(registry: InMemoryPromptRegistry) -> None:
    """登记键位契约并写入 v2 草案。不激活。"""
    registry.register(SHOT_VISUAL_TEMPLATE)
    registry.add_version(
        SHOT_VISUAL_KEY,
        SHOT_VISUAL_BODY_V2,
        change_note="v2 草案：从单帧升级到多帧综合描述，新增 motion 字段；禁止逐帧分述",
        author="claude",
    )

    registry.register(NARRATIVE_TEMPLATE)
    registry.add_version(
        NARRATIVE_KEY,
        NARRATIVE_BODY_V1,
        change_note="初版草案：叙事结构还原，强制证据引用并区分观测与推断",
        author="claude",
    )
