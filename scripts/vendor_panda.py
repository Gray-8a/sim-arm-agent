"""把 menagerie 的 panda.xml 复制到 assets/panda_local.xml，并修正 meshdir。

上游 panda.xml 用 meshdir="assets"，这个路径是相对它自己所在目录解析的。
我们把场景文件放在 assets/ 里，include 上游文件时 meshdir 会被拼到主文件目录上，
导致找不到 mesh。这里生成一份本地副本，把 meshdir 改成相对本文件的路径。

同时删除上游的 <keyframe>：场景里加了两个 freejoint 方块后 nq 变大，
上游 keyframe 的 qpos 长度对不上，会让编译直接报错。home 位姿改由代码
（config.HOME_QPOS / env.reset）设置，行为一致且更可控。
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
DST = ROOT / "assets" / "panda_local.xml"

MESH_REL = "../third_party/mujoco_menagerie/franka_emika_panda/assets"


def main() -> None:
    txt = SRC.read_text(encoding="utf-8")

    old = 'meshdir="assets"'
    new = f'meshdir="{MESH_REL}"'
    if old not in txt:
        raise SystemExit(f"meshdir marker not found in {SRC}")
    txt = txt.replace(old, new)

    # 删掉整段 <keyframe>...</keyframe>
    txt, n_key = re.subn(r"\s*<keyframe>.*?</keyframe>", "", txt, flags=re.S)
    if n_key != 1:
        raise SystemExit(f"expected 1 keyframe block, removed {n_key}")

    DST.write_text(txt, encoding="utf-8")
    print(f"wrote {DST.relative_to(ROOT)} (keyframe blocks removed: {n_key})")


if __name__ == "__main__":
    main()
