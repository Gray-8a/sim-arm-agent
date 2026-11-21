"""重写 .gitignore。

起因：早前用 PowerShell 的 Add-Content 追加内容，按 ANSI 编码写入了 UTF-8 文件，
生成了混合编码的坏文件，而且已经提交进仓库。这个脚本用明确的 UTF-8 重写它。

Windows 上不要再用 Set-Content / Add-Content 改含中文的文本文件，
PS 5.1 的默认编码是 ANSI，会静默损坏文件。
"""

from pathlib import Path

CONTENT = """# Python
__pycache__/
*.py[cod]
.venv/
.venv-vla/
venv/
*.egg-info/

# 运行产物
logs/
artifacts/*.png
artifacts/*.gif
artifacts/*.json
eval/results/

# Environment
.env
.env.local

# 上游模型（按需 sparse clone，不入库）
third_party/

# 编辑器
.vscode/
.idea/
.DS_Store

# MuJoCo 运行日志
MUJOCO_LOG.TXT
"""


def main() -> None:
    p = Path(__file__).resolve().parents[2] / ".gitignore"
    p.write_text(CONTENT, encoding="utf-8", newline="\n")
    b = p.read_bytes()
    text = b.decode("utf-8")
    bom = b.startswith(b"\xef\xbb\xbf")
    print(f"wrote {p}  bytes={len(b)}  bom={bom}  lines={len(text.splitlines())}")


if __name__ == "__main__":
    main()
