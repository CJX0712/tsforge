"""forecasters/autoarima — 复用 pmdarima 的 AutoARIMA（SOTA 自动定阶，可选后端）。

复用 Hyndman `auto.arima` 方法论（自动选择 ARIMA 阶数 + 季节项），工业级 SOTA 自动选型。
**可选依赖**：pmdarima 缺失时 `available()` 返回 False，benchmark 自动跳过，
由 statsmodels/Sklearn 实现兜底 → 保证 clone 后零下载可跑 demo。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tsforge.core.types import ForecastResult, TimeSeries
from tsforge.forecasters.base import BaseForecaster


class AutoARIMAForecaster(BaseForecaster):
    name = "AutoARIMA(pmdarima)"

    def __init__(
        self,
        seasonal: bool = True,
        m: int = 12,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.seasonal = seasonal
        self.m = int(m)
        self.random_state = random_state
        self._model = None

    def available(self) -> bool:
        try:
            import pmdarima  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, train: TimeSeries) -> "AutoARIMAForecaster":
        import pmdarima as pm

        self._train = train
        y = np.asarray(train.values, dtype=np.float64).ravel()
        self._model = pm.auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m,
            stepwise=True,
            suppress_warnings=True,
            random_state=self.random_state,
            error_action="ignore",
        )
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        h = int(horizon)
        out = self._model.predict(h, return_conf_int=True, alpha=0.05)
        if isinstance(out, tuple):
            mean, ci = out
            mean = np.asarray(mean, dtype=np.float64).ravel()
            ci = np.asarray(ci, dtype=np.float64)
            lower = ci[:, 0].ravel()
            upper = ci[:, 1].ravel()
        else:
            mean = np.asarray(out, dtype=np.float64).ravel()
            lower = upper = None
        return ForecastResult(mean=mean, lower=lower, upper=upper)
