"""规则解析器的单测：只测"哪些该过、哪些该拦"，不测措辞。"""

import pytest

from arm_agent.parser import parse_with_rules


@pytest.mark.parametrize(
    "text,obj,target",
    [
        ("把红色方块放到左边托盘", "red_block", "tray_left"),
        ("把蓝色方块放到右边托盘", "blue_block", "tray_right"),
        ("红块移到左托盘", "red_block", "tray_left"),
        ("把蓝色的方块放到左边的托盘", "blue_block", "tray_left"),
    ],
)
def test_clear_pick_and_place(text, obj, target):
    r = parse_with_rules(text)
    assert r.command is not None, r.error
    assert (r.command.object, r.command.target) == (obj, target)
    assert r.command.confidence >= 0.75


def test_home_variants():
    for text in ("回到原点", "返回初始位置", "归位", "复位"):
        r = parse_with_rules(text)
        assert r.command is not None, text
        assert r.command.action.value == "home"


@pytest.mark.parametrize(
    "text",
    [
        "把那个东西拿过来",
        "把它放到左边",
        "把东西放好",
    ],
)
def test_vague_rejected(text):
    r = parse_with_rules(text)
    assert r.command is None


@pytest.mark.parametrize(
    "text",
    [
        "把红色方块扔到桌子外面",
        "把方块放到地上",
        "把红色方块扔出窗外",
    ],
)
def test_unsafe_words_rejected(text):
    r = parse_with_rules(text)
    assert r.command is None
    assert "非白名单" in (r.error or "")


def test_negation_rejected():
    """否定句是真实漏洞来源，必须拦。"""
    for text in ("别把红色方块放到左边托盘", "红色方块不要动", "禁止移动红块"):
        r = parse_with_rules(text)
        assert r.command is None, text
        assert "否定" in (r.error or "")


def test_ambiguous_target_goes_to_approval():
    """二选一的指令不硬阻断，但置信度要低到触发人工。"""
    r = parse_with_rules("把红块放到左边或者右边")
    assert r.command is not None
    assert r.command.confidence < 0.75


def test_single_char_lowers_confidence():
    r = parse_with_rules("把红放到左")
    assert r.command is not None
    assert r.command.confidence < 0.75


def test_llm_path_without_key_returns_error_not_crash():
    from arm_agent.parser import parse_with_llm

    r = parse_with_llm("把红块放到左边")
    assert r.command is None
    assert "API key" in (r.error or "") or "key" in (r.error or "")
