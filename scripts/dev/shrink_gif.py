"""把演示 GIF 压小，减少 README 首屏加载量。

坑：每帧独立 quantize 会让 GIF 反而变大（每帧都要存完整调色板）。
正确做法是共用一张调色板 + 抽帧 + 缩小尺寸。

源文件用 artifacts/demo_d2.gif（原始渲染输出），输出覆盖 assets/demo.gif。
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "demo_d2.gif"
DST = ROOT / "assets" / "demo.gif"

SIZE = (480, 360)
COLORS = 256
KEEP_EVERY = 2  # 抽帧：31 帧 -> 16 帧


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"原始 GIF 不存在: {SRC}（先跑 scripts/demo_d2.py --gif）")

    before = DST.stat().st_size if DST.exists() else 0
    im = Image.open(SRC)
    n = getattr(im, "n_frames", 1)

    frames_rgb = []
    for i in range(n):
        if i % KEEP_EVERY and i != n - 1:
            continue
        im.seek(i)
        frames_rgb.append(im.convert("RGB").resize(SIZE, Image.LANCZOS))
    im.close()

    # 用全部帧拼一张长图求全局调色板，所有帧共用这一张
    strip = Image.new("RGB", (SIZE[0], SIZE[1] * len(frames_rgb)))
    for i, f in enumerate(frames_rgb):
        strip.paste(f, (0, i * SIZE[1]))
    palette_img = strip.quantize(colors=COLORS, method=Image.MEDIANCUT)

    frames_q = [f.quantize(palette=palette_img, dither=Image.NONE) for f in frames_rgb]

    frames_q[0].save(
        DST,
        save_all=True,
        append_images=frames_q[1:],
        duration=100,
        loop=0,
        optimize=True,
    )
    after = DST.stat().st_size
    print(
        f"frames {n} -> {len(frames_q)}  size {SIZE[0]}x{SIZE[1]}  colors {COLORS}  "
        f"{before/1024:.0f} KB -> {after/1024:.0f} KB"
    )


if __name__ == "__main__":
    main()
