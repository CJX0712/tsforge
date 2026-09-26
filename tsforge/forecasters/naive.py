"""forecasters/naive — 标准弱基线（Naive / SeasonalNaive / Drift）。

这些不是「自研模型」，而是时间序列评测的通用参照基线（M4 竞赛同样内置），
用于 MASE 缩放与 SOTA 对标。实现为标准教科书公式。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tsforge.core.types import ForecastResult, TimeSeries
from tsforge.forecasters.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    name = "Naive"

    def fit(self, train: TimeSeries) -> "NaiveForecaster":
        self._train = train
        self._last = float(train.values[-1])
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        return ForecastResult(mean=np.full(int(horizon), self._last))


class SeasonalNaiveForecaster(BaseForecaster):
    name = "SeasonalNaive"

    def __init__(self, period: int = 12) -> None:
        super().__init__()
        self.period = int(period)

    def fit(self, train: TimeSeries) -> "SeasonalNaiveForecaster":
        self._train = train
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        v = self._train.values
        h = int(horizon)
        if self.period <= 0 or len(v) <= self.period:
            # 退化为 Naive
            return ForecastResult(mean=np.full(h, float(v[-1])))
        out = [float(v[-self.period + (i % self.period)]) for i in range(h)]
        return ForecastResult(mean=np.asarray(out, dtype=np.float64))


class DriftForecaster(BaseForecaster):
    name = "Drift"

    def fit(self, train: TimeSeries) -> "DriftForecaster":
        self._train = train
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        v = self._train.values
        h = int(horizon)
        n = len(v)
        if n < 2:
            return ForecastResult(mean=np.full(h, float(v[-1])))
        slope = (v[-1] - v[0]) / (n - 1)
        start = float(v[-1])
        out = [start + slope * (i + 1) for i in range(h)]
        return ForecastResult(mean=np.asarray(out, dtype=np.float64))
