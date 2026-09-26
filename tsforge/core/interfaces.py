"""interfaces — 解耦用的 Protocol 契约（依赖倒置，便于独立测试与替换实现）。

- Forecaster: 所有预测器统一接口；`available()` 支持可选后端离线降级。
- DatasetSource: 数据来源（合成 / CSV / 数据库）统一抽象。
- Metric: 评测指标；`evaluate(actual, pred, train)` 中 train 用于 MASE 缩放。
- Evaluator: 对单个 forecaster 在数据集上给出指标结果。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from tsforge.core.types import (
    DataSet,
    ForecastResult,
    MetricResult,
    TimeSeries,
)


@runtime_checkable
class Forecaster(Protocol):
    name: str

    def available(self) -> bool: ...

    def fit(self, train: TimeSeries) -> "Forecaster": ...

    def forecast(self, horizon: int) -> ForecastResult: ...


@runtime_checkable
class DatasetSource(Protocol):
    def load(self) -> DataSet: ...


@runtime_checkable
class Metric(Protocol):
    name: str

    def evaluate(self, actual, pred, train) -> float: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self, fc: Forecaster, test: TimeSeries, train: TimeSeries, h: int
    ) -> List[MetricResult]: ...
