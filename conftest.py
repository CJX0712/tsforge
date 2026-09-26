"""conftest — 将仓库根加入 sys.path，保证 pytest 可 import tsforge。"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
