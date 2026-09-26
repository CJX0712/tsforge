"""pipeline/forecast_pipeline — 端到端编排。

数据(合成/真实) -> 滚动原点切分 -> 多预测器 -> M4 指标基准 -> 结果表/JSON。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from tsforge.core.config import Config
from tsforge.core.types import BenchmarkResult, DataSet
from tsforge.data.synthetic import make_synthetic
from tsforge.eval.benchmark import benchmark
from tsforge.forecasters import (
    AutoARIMAForecaster,
    DriftForecaster,
    HoltWintersForecaster,
    MLForecaster,
    NaiveForecaster,
    SARIMAXForecaster,
    SeasonalNaiveForecaster,
)


class ForecastPipeline:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    def default_models(self) -> List:
        cfg = self.config
        return [
            NaiveForecaster(),
            SeasonalNaiveForecaster(period=cfg.season_period),
            DriftForecaster(),
            HoltWintersForecaster(seasonal_periods=cfg.season_period),
            SARIMAXForecaster(seasonal_order=(0, 1, 1, cfg.season_period)),
            MLForecaster(lag=cfg.season_period),
            AutoARIMAForecaster(m=cfg.season_period),
        ]

    def run(
        self,
        dataset: Optional[DataSet] = None,
        models: Optional[Sequence] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[BenchmarkResult, DataSet]:
        if dataset is None:
            dataset = make_synthetic(self.config)
        if models is None:
            models = self.default_models()
        result = benchmark(models, dataset, self.config)
        if save_path:
            result.save_json(save_path)
        return result, dataset

    def run_demo(self, save_path: str = "benchmark.json") -> BenchmarkResult:
        res, _ = self.run(save_path=save_path)
        return res
