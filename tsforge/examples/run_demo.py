"""examples/run_demo — 最小可运行示例（clone 后零干预可跑）。

演示：合成数据 -> 多模型基准 -> 打印表格 -> 写 benchmark.json。
运行：python tsforge/examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

# 将仓库根加入 sys.path（无需安装即可运行）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tsforge.pipeline.forecast_pipeline import ForecastPipeline  # noqa: E402


def main() -> None:
    pipe = ForecastPipeline()
    result, dataset = pipe.run()
    print(f"dataset: {dataset.name} ({len(dataset)} series)")
    print(result.to_table())
    out = os.path.join(_REPO_ROOT, "benchmark.json")
    result.save_json(out)
    print(f"\nwritten: {out} ({len(result.rows)} models)")


if __name__ == "__main__":
    main()
