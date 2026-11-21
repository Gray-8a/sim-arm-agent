"""schema 的单测：白名单与字段组合。"""

import pytest
from pydantic import ValidationError

from arm_agent.schema import Action, Command, try_parse_command


def test_valid_pick_and_place():
    cmd = Command(action=Action.PICK_AND_PLACE, object="red_block", target="tray_left", confidence=0.9)
    assert cmd.summary() == "把 红色方块 放到 左侧托盘"


def test_unknown_object_rejected():
    with pytest.raises(ValidationError):
        Command(action=Action.PICK, object="green_block", confidence=0.9)


def test_unknown_target_rejected():
    with pytest.raises(ValidationError):
        Command(action=Action.PICK_AND_PLACE, object="red_block", target="floor", confidence=0.9)


def test_pick_and_place_requires_target():
    with pytest.raises(ValidationError):
        Command(action=Action.PICK_AND_PLACE, object="red_block", confidence=0.9)


def test_home_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Command(action=Action.HOME, object="red_block", confidence=0.9)


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        Command(action=Action.HOME, confidence=0.9, position=[1, 2, 3])


def test_confidence_range():
    with pytest.raises(ValidationError):
        Command(action=Action.HOME, confidence=1.5)


def test_try_parse_returns_error_instead_of_raising():
    cmd, err = try_parse_command({"action": "pick", "object": "green_block", "confidence": 0.8})
    assert cmd is None and err
