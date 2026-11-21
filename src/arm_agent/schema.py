"""指令的数据契约。

这是整条链路唯一的"模型输出形状"。LLM 或规则解析器的产物必须先过这里，
才有资格往下走到 guard / planner。

关键设计：模型不产出坐标。动作、物体、目标位置全是白名单枚举，
几何参数一律由代码查表得到。这样"把方块扔到桌子外面"这种指令在数据结构上
根本无法表达，而不是靠提示词求模型别这么干。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import config as C


class Action(str, Enum):
    HOME = "home"
    PICK = "pick"
    PICK_AND_PLACE = "pick_and_place"


class Command(BaseModel):
    """一条已结构化的执行指令。"""

    model_config = ConfigDict(extra="forbid")

    action: Action
    object: str | None = None
    target: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""

    @field_validator("object")
    @classmethod
    def _known_object(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in C.OBJECTS:
            raise ValueError(f"object 不在白名单内: {v!r}，允许: {sorted(C.OBJECTS)}")
        return v

    @field_validator("target")
    @classmethod
    def _known_target(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in C.SITES:
            raise ValueError(f"target 不在白名单内: {v!r}，允许: {sorted(C.SITES)}")
        return v

    @model_validator(mode="after")
    def _consistent_fields(self) -> Command:
        if self.action is Action.PICK_AND_PLACE:
            if self.object is None or self.target is None:
                raise ValueError("pick_and_place 必须同时给出 object 和 target")
        elif self.action is Action.PICK:
            if self.object is None:
                raise ValueError("pick 必须给出 object")
        elif self.action is Action.HOME:
            if self.object is not None or self.target is not None:
                raise ValueError("home 不接受 object/target")
        return self

    def summary(self) -> str:
        if self.action is Action.HOME:
            return "回 home"
        if self.action is Action.PICK:
            return f"抓取 {C.OBJECTS[self.object]}"
        return f"把 {C.OBJECTS[self.object]} 放到 {C.SITES[self.target]}"


def try_parse_command(payload: dict) -> tuple[Command | None, str | None]:
    """安全地构造 Command，返回 (command, 错误信息)。不抛异常，供 guard L1 使用。"""
    try:
        return Command.model_validate(payload), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
