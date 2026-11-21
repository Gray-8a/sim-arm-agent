"""自然语言 -> Command。

两条路径：
- llm：调 DeepSeek/OpenAI 兼容接口，JSON 模式，只允许输出白名单枚举。
- rule：关键词回退。没有 API key、CI、离线评测时用它，保证链路不依赖网络。

两条路径产出同一个 Command，后续 guard/planner/executor 完全不区分来源。
这也是能拿来做对照评测的原因：同一条指令分别用 rule 和 llm 解析，比较解析成功率。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from . import config as C
from .schema import Action, Command

# ---------------------------------------------------------------- 规则表

# 长词优先，避免"左边"被"左"抢先匹配
_OBJECT_PATTERNS: list[tuple[str, list[str]]] = [
    ("red_block", ["红色方块", "红色的方块", "红色块", "红方块", "红的方块", "红块", "红色", "红的", "红"]),
    ("blue_block", ["蓝色方块", "蓝色的方块", "蓝色块", "蓝方块", "蓝的方块", "蓝块", "蓝色", "蓝的", "蓝"]),
]

_TARGET_PATTERNS: list[tuple[str, list[str]]] = [
    ("tray_left", ["左边托盘", "左侧托盘", "左托盘", "左边的", "左侧", "左边", "左"]),
    ("tray_right", ["右边托盘", "右侧托盘", "右托盘", "右边的", "右侧", "右边", "右"]),
]

_PLACE_VERBS = [
    "放到", "放在", "移到", "挪到", "搬到", "送到", "放进", "丢到", "扔到", "搁到", "摆到",
    "放", "移", "搬", "送", "摆", "搁", "丢", "扔",
]
_PICK_VERBS = ["抓", "拿", "夹", "拾", "取"]

# 复位指令用正则，因为"回到原点""返回初始位置"这类说法太散，枚举词容易漏
_HOME_PATTERN = re.compile(r"回(到|至)?\s*(原点|原位|初始位置|初始姿态|home)|归位|复位", re.I)

# 出现这些词说明目标不在白名单语境内，直接判为无法解析（硬阻断，不猜）
_UNSAFE_WORDS = ["外面", "地上", "窗外", "扔出", "抛出", "丢出", "砸", "摔", "人身上", "手上", "脸上"]

# 否定指令：关键词级检测，宁可错拦也不误执行。
# 已知不完备——"千万不要把红块放左边" 里的"千万"能过，靠后面的否定词兜住；
# 但 "千万别" 这类变体仍有漏网可能，写进 README 的局限一节。
_NEGATION_PATTERN = re.compile(r"别|不要|不准|禁止|切勿|不能|不该|不许")

# 目标二选一，语义上无法确定，压低置信度交给人工
_AMBIGUOUS_WORDS = ["或者", "还是", "二选一", "随便哪个"]

# 指代不明，命中则压低置信度，交给人工确认
_VAGUE_WORDS = ["那个", "这个", "它", "东西", "玩意", "随便", "某个"]

_MODAL_WORDS = ["可能", "也许", "大概", "好像", "应该"]


@dataclass
class ParseResult:
    raw: str
    source: str  # "llm" | "rule"
    payload: dict | None = None
    command: Command | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "source": self.source,
            "payload": self.payload,
            "command": self.command.model_dump() if self.command else None,
            "error": self.error,
            "notes": self.notes,
        }


# ---------------------------------------------------------------- 规则解析


def _match(text: str, patterns: list[tuple[str, list[str]]]) -> tuple[str | None, str | None]:
    """返回 (枚举值, 命中的词)。长词优先。"""
    best: tuple[str | None, str | None] = (None, None)
    best_len = 0
    for value, words in patterns:
        for w in words:
            if w in text and len(w) > best_len:
                best, best_len = (value, w), len(w)
    return best


def parse_with_rules(text: str) -> ParseResult:
    raw = text.strip()
    res = ParseResult(raw=raw, source="rule")
    notes: list[str] = []

    has_place = any(v in raw for v in _PLACE_VERBS)
    has_pick = any(v in raw for v in _PICK_VERBS)

    # 纯复位指令：命中复位词且句子里没有抓/放动作（"把红块放回原点"不该走这里）
    if _HOME_PATTERN.search(raw) and not (has_place or has_pick):
        res.payload = {"action": Action.HOME.value, "confidence": 0.95, "reasoning": "命中复位词"}
        return _finalize(res, notes)

    unsafe = [w for w in _UNSAFE_WORDS if w in raw]
    if unsafe:
        res.error = f"指令包含非白名单目标词 {unsafe}，无法映射到任何已知位置"
        notes.append("按安全策略不猜测目标，直接拒绝")
        res.notes = notes
        return res

    negation = _NEGATION_PATTERN.search(raw)
    if negation:
        res.error = f"检测到否定词「{negation.group(0)}」，否定指令不执行（无法确认是禁止还是笔误）"
        notes.append("否定语义需要人确认，关键词级检测不做语义推理")
        res.notes = notes
        return res

    obj, obj_word = _match(raw, _OBJECT_PATTERNS)
    tgt, tgt_word_text = _match(raw, _TARGET_PATTERNS)

    vague = [w for w in _VAGUE_WORDS if w in raw]
    modal = [w for w in _MODAL_WORDS if w in raw]
    ambiguous = [w for w in _AMBIGUOUS_WORDS if w in raw]

    # 置信度：明确长词高，短词次之，模糊指代与情态词扣分
    conf = 0.90
    if obj_word and len(obj_word) <= 1:
        conf -= 0.10
        notes.append(f"物体只用单字「{obj_word}」指代")
    if tgt_word_text and len(tgt_word_text) <= 1:
        conf -= 0.10
        notes.append(f"位置只用单字「{tgt_word_text}」指代")
    if vague:
        conf -= 0.18
        notes.append(f"含模糊指代 {vague}")
    if modal:
        conf -= 0.12
        notes.append(f"含情态词 {modal}")
    if ambiguous:
        conf -= 0.30
        notes.append(f"目标存在二义性 {ambiguous}")

    if obj is None:
        res.error = "未能识别操作对象" + (f"（含模糊指代 {vague}）" if vague else "")
        res.notes = notes
        return res

    if has_place:
        if tgt is None:
            res.error = "识别到放置动作但未能识别目标位置"
            res.notes = notes
            return res
        res.payload = {
            "action": Action.PICK_AND_PLACE.value,
            "object": obj,
            "target": tgt,
            "confidence": round(max(0.05, min(0.99, conf)), 2),
            "reasoning": f"规则匹配: 物体={obj_word}, 位置={tgt_word_text}",
        }
    elif has_pick:
        res.payload = {
            "action": Action.PICK.value,
            "object": obj,
            "confidence": round(max(0.05, min(0.99, conf - 0.05)), 2),
            "reasoning": f"规则匹配: 仅抓取, 物体={obj_word}",
        }
    else:
        res.error = "未能识别动作（既没有放置动词也没有抓取动词）"
        res.notes = notes
        return res

    return _finalize(res, notes)


def _finalize(res: ParseResult, notes: list[str]) -> ParseResult:
    from .schema import try_parse_command

    if res.payload is None:
        res.notes = notes
        return res
    cmd, err = try_parse_command(res.payload)
    res.command = cmd
    res.error = err
    res.notes = notes
    return res


# ---------------------------------------------------------------- LLM 解析

SYSTEM_PROMPT = """你是一个机械臂指令解析器。把用户的中文指令转成严格的 JSON。

可用动作 action:
- "pick_and_place": 把一个物体移动到某个目标位置
- "pick": 只抓取，不指定目的地
- "home": 回到初始姿态

可用物体 object（只能用这些字符串，不能自造）:
{objects}

可用目标位置 target（只能用这些字符串，不能自造）:
{sites}

输出 JSON 字段:
- action: 上面三个值之一
- object: 物体字符串，或 null
- target: 目标位置字符串，或 null
- confidence: 0 到 1 的浮点数，表示你对这次解析的把握
- reasoning: 一句话说明依据

规则:
1. 只输出 JSON，不要 markdown，不要解释。
2. object 和 target 必须来自给定列表。用户的说法如果对不上任何一项，就填 null，
   不要猜、不要造新字符串。
3. 如果用户想放到列表之外的地方（比如"桌子外面""地上"），object 或 target 填 null，
   并在 confidence 上给出低分（<= 0.3）。
4. 如果指令含糊（"那个东西""它"），confidence 不要超过 0.5。
5. 用户提到颜色/方位时，对应到具体的 object/target。
"""


def _build_prompt() -> str:
    objects = "\n".join(f'- "{k}": {v}' for k, v in C.OBJECTS.items())
    sites = "\n".join(f'- "{k}": {v}' for k, v in C.SITES.items())
    return SYSTEM_PROMPT.format(objects=objects, sites=sites)


def parse_with_llm(
    text: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> ParseResult:
    """调用 DeepSeek/OpenAI 兼容接口。需要 openai 包与 API key。"""
    raw = text.strip()
    res = ParseResult(raw=raw, source="llm")

    api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.deepseek.com"
    model = model or os.getenv("LLM_MODEL") or "deepseek-chat"

    if not api_key:
        res.error = "缺少 API key（设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY）"
        return res

    try:
        from openai import OpenAI
    except ImportError:
        res.error = "未安装 openai 包"
        return res

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _build_prompt()},
                {"role": "user", "content": raw},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        res.payload = json.loads(content)
    except Exception as exc:  # noqa: BLE001
        res.error = f"LLM 调用失败: {type(exc).__name__}: {exc}"
        return res

    return _finalize(res, res.notes)


# ---------------------------------------------------------------- 统一入口


def parse(text: str, mode: str = "auto") -> ParseResult:
    """mode: "rule" | "llm" | "auto"（有 key 用 llm，否则回退 rule）。"""
    if mode == "rule":
        return parse_with_rules(text)
    if mode == "llm":
        return parse_with_llm(text)

    has_key = bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if has_key:
        res = parse_with_llm(text)
        if res.command is not None or res.payload is not None:
            return res
        # LLM 连不通时退回规则，并在 notes 里留痕
        fallback = parse_with_rules(text)
        fallback.notes.append(f"LLM 失败已回退规则: {res.error}")
        fallback.source = "rule(fallback)"
        return fallback
    return parse_with_rules(text)
