"""forecasters/statsmodels_forecasters — 复用工业级 statsmodels（SOTA 统计预测）。

- HoltWintersForecaster: ETS(A,A,A) 指数平滑（趋势+季节）。
- SARIMAXForecaster: 季节性 ARIMA（Box-Jenkins 经典 SOTA）。

均复用 statsmodels 0.15（BSD-3，活跃维护，纯 numpy/scipy），非自研。
`available()` 在 statsmodels 缺失时返回 False → benchmark 自动跳过。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tsforge.core.types import ForecastResult, TimeSeries
from tsforge.forecasters.base import BaseForecaster


def _ets_forecast(model, horizon: int) -> ForecastResult:
    mean = np.asarray(model.forecast(int(horizon)), dtype=np.float64).ravel()
    lower = upper = None
    try:
        ci = model.get_forecast(int(horizon)).conf_int(alpha=0.05)
        lower = np.asarray(ci[:, 0], dtype=np.float64).ravel()
        upper = np.asarray(ci[:, 1], dtype=np.float64).ravel()
    except Exception:
        resid = np.asarray(getattr(model, "resid", np.array([0.0])), dtype=np.float64)
        sigma = float(np.std(resid)) + 1e-9
        lower = mean - 1.96 * sigma
        upper = mean + 1.96 * sigma
    return ForecastResult(mean=mean, lower=lower, upper=upper)


class HoltWintersForecaster(BaseForecaster):
    name = "HoltWinters(ETS)"

    def __init__(
        self,
        seasonal_periods: int = 12,
        trend: str = "add",
        seasonal: str = "add",
    ) -> None:
        super().__init__()
        self.seasonal_periods = int(seasonal_periods)
        self.trend = trend
        self.seasonal = seasonal

    def available(self) -> bool:
        try:
            import statsmodels  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, train: TimeSeries) -> "HoltWintersForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._train = train
        y = np.asarray(train.values, dtype=np.float64).ravel()
        if len(y) < 2 * self.seasonal_periods:
            # 数据不足以支撑季节项 → 退化为加性趋势无季节
            self._model = ExponentialSmoothing(y, trend="add", seasonal=None).fit()
        else:
            self._model = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
            ).fit()
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        return _ets_forecast(self._model, horizon)


class SARIMAXForecaster(BaseForecaster):
    name = "SARIMAX"

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.random_state = random_state

    def available(self) -> bool:
        try:
            import statsmodels  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, train: TimeSeries) -> "SARIMAXForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._train = train
        y = np.asarray(train.values, dtype=np.float64).ravel()
        n = len(y)
        m = self.seasonal_order[3]
        # 自适应：训练足够才用季节项，且用稳定的「季节 MA-only」配置 (0,1,1,m)，
        # 避免季节 AR 项在短序列上估计失败/爆炸（经典 airline 模型即此结构）。
        if m > 0 and n >= 3 * m:
            order = self.order
            seasonal_order = (0, 1, 1, m)
        else:
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
        try:
            self._model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        except Exception:
            # 兜底：最简非季节 ARIMA
            self._model = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        f = self._model.get_forecast(int(horizon))
        mean = np.asarray(f.predicted_mean, dtype=np.float64).ravel()
        ci = np.asarray(f.conf_int(alpha=0.05), dtype=np.float64)
        lower = ci[:, 0].ravel()
        upper = ci[:, 1].ravel()
        return ForecastResult(mean=mean, lower=lower, upper=upper)
