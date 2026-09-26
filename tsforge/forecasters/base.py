"""forecasters/base — 预测器基类与通用工具。"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tsforge.core.types import ForecastResult, TimeSeries


class BaseForecaster:
    name = "base"

    def __init__(self) -> None:
        self._fitted = False
        self._train: Optional[TimeSeries] = None

    def available(self) -> bool:
        return True

    def fit(self, train: TimeSeries) -> "BaseForecaster":
        self._train = train
        self._fitted = True
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        raise NotImplementedError

    def _require_fitted(self) -> None:
        if not self._fitted or self._train is None:
            raise RuntimeError(f"{self.name} must be fit() before forecast()")
