"""探测 SmolVLA 的输入输出契约，只下载 config.json，不下权重。

目的是搞清楚：这个策略期望什么样的 observation，输出什么形状的 action。
这决定了怎么把它和我们自己的仿真接起来（或者说明为什么接不上）。
"""

import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "lerobot/smolvla_base"


def main() -> None:
    path = hf_hub_download(REPO, "config.json")
    cfg = json.loads(open(path, encoding="utf-8").read())

    print("=== 关键字段 ===")
    for k in (
        "type",
        "n_action_steps",
        "chunk_size",
        "max_state_dim",
        "max_action_dim",
        "resize_imgs_with_padding",
        "empty_cameras",
        "freeze_vision_encoder",
        "train_expert_only",
    ):
        if k in cfg:
            print(f"  {k}: {cfg[k]}")

    print("\n=== input_features ===")
    print(json.dumps(cfg.get("input_features"), indent=2, ensure_ascii=False))

    print("\n=== output_features ===")
    print(json.dumps(cfg.get("output_features"), indent=2, ensure_ascii=False))

    print("\n=== 其余顶层键 ===")
    print(sorted(cfg.keys()))


if __name__ == "__main__":
    main()
