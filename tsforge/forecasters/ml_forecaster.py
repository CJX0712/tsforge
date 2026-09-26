"""forecasters/ml_forecaster — 复用 scikit-learn 的递归多步预测器（SOTA ML 分支）。

用滞后(lag) + 时间索引(趋势) 特征训练回归器（默认 RandomForest），
按递归方式逐点预测 horizon 步。复用 sklearn（BSD，已为依赖，零额外体积、稳定）。
属「集成领先开源」而非自研模型。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tsforge.core.types import ForecastResult, TimeSeries
from tsforge.forecasters.base import BaseForecaster


class MLForecaster(BaseForecaster):
    name = "ML(sklearn RF)"

    def __init__(self, lag: int = 12, estimator=None, random_state: int = 42) -> None:
        super().__init__()
        self.lag = int(lag)
        self._estimator = estimator
        self.random_state = random_state
        self._rf = None
        self._degenerate = False
        self._mean = 0.0
        self._history: Optional[np.ndarray] = None

    def available(self) -> bool:
        try:
            import sklearn  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, train: TimeSeries) -> "MLForecaster":
        from sklearn.ensemble import RandomForestRegressor

        self._train = train
        y = np.asarray(train.values, dtype=np.float64).ravel()
        lag = self.lag
        X, Y = [], []
        for t in range(lag, len(y)):
            X.append(list(y[t - lag : t]) + [t])  # 滞后 + 时间索引(趋势)
            Y.append(y[t])
        if not X:
            self._degenerate = True
            self._mean = float(np.mean(y)) if len(y) else 0.0
            self._fitted = True
            return self
        if self._estimator is None:
            self._rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        else:
            self._rf = self._estimator
        self._rf.fit(np.asarray(X), np.asarray(Y))
        self._history = y.copy()
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self._require_fitted()
        h = int(horizon)
        if self._degenerate:
            return ForecastResult(mean=np.full(h, self._mean))
        yhist = list(self._history)
        preds: list = []
        for _ in range(h):
            feat = list(yhist[-self.lag :]) + [len(yhist)]
            p = float(self._rf.predict(np.asarray(feat, dtype=np.float64).reshape(1, -1))[0])
            preds.append(p)
            yhist.append(p)
        return ForecastResult(mean=np.asarray(preds, dtype=np.float64))
