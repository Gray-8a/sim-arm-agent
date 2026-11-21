"""安全闸门的单测。

这里只测"判断逻辑"，不测仿真精度——仿真状态用一次 fixture，避免每条都重建模型。
"""

import numpy as np
import pytest

from arm_agent.env import ArmEnv
from arm_agent.guard import (
    check_l1_structure,
    check_l2_confidence,
    check_l3_workspace,
    check_l4_reachability,
    in_workspace,
    run_guards,
)
from arm_agent.schema import Action, Command


@pytest.fixture(scope="module")
def env():
    return ArmEnv()


def _cmd(**kw) -> Command:
    base = {"action": Action.PICK_AND_PLACE, "object": "red_block", "target": "tray_left", "confidence": 0.9}
    base.update(kw)
    return Command(**base)


def test_workspace_bounds():
    assert in_workspace(np.array([0.5, 0.0, 0.1]))
    assert not in_workspace(np.array([1.5, 0.0, 0.1]))
    assert not in_workspace(np.array([0.5, 0.0, -0.5]))


def test_l1_blocks_when_command_missing():
    r = check_l1_structure(None, "解析失败")
    assert not r.passed and r.hard_block


def test_l1_passes_valid_command():
    r = check_l1_structure(_cmd(), None)
    assert r.passed and not r.hard_block


def test_l2_low_confidence_is_soft():
    r = check_l2_confidence(_cmd(confidence=0.5))
    assert not r.passed
    assert not r.hard_block  # 低置信度可以人工放行


def test_l2_high_confidence_passes():
    assert check_l2_confidence(_cmd(confidence=0.9)).passed


def test_l3_in_range(env):
    assert check_l3_workspace(_cmd(), env).passed


def test_l3_hard_blocks_out_of_range(env):
    """把目标 site 人为挪出工作空间，验证 L3 是硬阻断。"""
    site_id = env._site_ids["tray_left"]
    saved = env.model.site_pos[site_id].copy()
    try:
        env.model.site_pos[site_id] = [1.5, 0.0, 0.1]
        env.forward()
        r = check_l3_workspace(_cmd(), env)
        assert not r.passed
        assert r.hard_block
    finally:
        env.model.site_pos[site_id] = saved
        env.forward()


def test_l4_reachable(env):
    assert check_l4_reachability(_cmd(), env).passed


def test_l4_unreachable_is_soft(env):
    """不可达不硬阻断：可能是被挡住，允许人判断。"""
    site_id = env._site_ids["tray_right"]
    saved = env.model.site_pos[site_id].copy()
    try:
        env.model.site_pos[site_id] = [0.9, 0.0, 0.55]
        env.forward()
        r = check_l4_reachability(_cmd(target="tray_right"), env)
        assert not r.passed
        assert not r.hard_block
    finally:
        env.model.site_pos[site_id] = saved
        env.forward()


def test_run_guards_short_circuits_on_l1(env):
    report = run_guards(None, env, parse_error="boom")
    assert report.hard_blocked
    assert len(report.results) == 1


def test_run_guards_all_pass(env):
    report = run_guards(_cmd(), env)
    assert report.passed
    assert not report.needs_approval
    assert [r.layer for r in report.results] == [
        "L1_structure",
        "L2_confidence",
        "L3_workspace",
        "L4_reachability",
    ]
