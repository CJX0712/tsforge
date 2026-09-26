"""cli — 命令行入口（argparse）。

用法：
  python -m tsforge.cli demo
  python -m tsforge.cli forecast --series airline --h 12 --model SARIMAX
  python -m tsforge.cli version
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from tsforge.core.config import Config
from tsforge.core.types import TimeSeries
from tsforge.data.synthetic import make_synthetic
from tsforge.forecasters import (
    AutoARIMAForecaster,
    DriftForecaster,
    HoltWintersForecaster,
    MLForecaster,
    NaiveForecaster,
    SARIMAXForecaster,
    SeasonalNaiveForecaster,
)
from tsforge.pipeline.forecast_pipeline import ForecastPipeline

_MODEL_REGISTRY = {
    "Naive": NaiveForecaster,
    "SeasonalNaive": SeasonalNaiveForecaster,
    "Drift": DriftForecaster,
    "HoltWinters": HoltWintersForecaster,
    "SARIMAX": SARIMAXForecaster,
    "ML": MLForecaster,
    "AutoARIMA": AutoARIMAForecaster,
}


def _resolve_model(name: str):
    if name in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name]
    key = name.strip().lower()
    alias = {
        "hw": "HoltWinters",
        "ets": "HoltWinters",
        "arima": "SARIMAX",
        "rf": "ML",
        "sklearn": "ML",
    }
    if key in alias:
        return _MODEL_REGISTRY[alias[key]]
    return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="tsforge", description="tsforge: 时间序列预测系统（作者: 晨星）"
    )
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="运行端到端基准 demo -> benchmark.json")
    sub.add_parser("version", help="显示版本")
    fc = sub.add_parser("forecast", help="对指定序列生成预测")
    fc.add_argument("--series", type=str, default="synth_0", help="序列名(synth_i / airline)")
    fc.add_argument("--h", type=int, default=12, help="预测步长")
    fc.add_argument("--model", type=str, default="SARIMAX")

    args = p.parse_args(argv)
    cfg = Config.from_env()

    if args.cmd == "version" or args.cmd is None:
        from tsforge import __version__

        print(f"tsforge {__version__}")
        return 0

    if args.cmd == "demo":
        pipe = ForecastPipeline(cfg)
        result, _ = pipe.run(save_path="benchmark.json")
        print(result.to_table())
        print(f"\nbenchmark.json written ({len(result.rows)} models)")
        return 0

    if args.cmd == "forecast":
        # 取序列：airline 或合成集中按名匹配
        if args.series == "airline":
            from tsforge.data.loader import load_airline

            series = load_airline()
        else:
            ds = make_synthetic(cfg)
            series = next((s for s in ds.series if s.name == args.series), ds.series[0])
        cls = _resolve_model(args.model)
        if cls is None:
            print(f"[error] unknown model '{args.model}'. available: {', '.join(_MODEL_REGISTRY)}")
            return 2
        model = cls()
        if not model.available():
            print(f"[error] model '{args.model}' backend unavailable")
            return 2
        model.fit(series)
        res = model.forecast(args.h)
        out = {
            "series": series.name,
            "model": model.name,
            "horizon": args.h,
            "mean": [round(float(x), 3) for x in res.mean],
        }
        if res.lower is not None:
            out["lower"] = [round(float(x), 3) for x in res.lower]
            out["upper"] = [round(float(x), 3) for x in res.upper]
        print(json.dumps(out, ensure_ascii=False))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
