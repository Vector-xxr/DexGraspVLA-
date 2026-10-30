#!/usr/bin/env python3
"""已并入 `eval_policy.py` —— 那个脚本 ACT 和 π0.5 都能跑，策略类型自动识别。

留这个文件只是为了让旧命令不至于静默跑错东西。
"""
import sys
from pathlib import Path

print(__doc__)
print(f"  改用：\n    python {Path(__file__).with_name('eval_policy.py')} --ckpt ...\n")
sys.exit(2)
