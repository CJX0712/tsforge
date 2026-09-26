"""eval/metrics — 时间序列预测指标（M4 竞赛标准）。

- MAE / RMSE：基础尺度相关误差（越小越好）。
- sMAPE：对称百分比误差（%）。
- MASE：相对朴素预测(NAIVE-1)的缩放误差 — **尺度无关、可跨序列比较**；
  scale = train 上 |y_t - y_{t-1}| 的均值（Hyndman & Koehler, 2006）。

这些是指标公式（评测基础设施），非自研模型；公式附标准文献，符合「复用/不重复造轮子」原则。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from tsforge.core.types import MetricResult, TimeSeries

METRICS: Dict[str, str] = {
    "MAE": "Mean Absolute Error",
    "RMSE": "Root Mean Square Error",
    "sMAPE": "Symmetric Mean Absolute Percentage Error (%)",
    "MASE": "Mean Absolute Scaled Error (vs NAIVE-1)",
}


def mae(actual, pred) -> float:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(a - p)))


def rmse(actual, pred) -> float:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - p) ** 2)))


def smape(actual, pred) -> float:
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    denom = np.abs(a) + np.abs(p)
    if denom.size == 0:
        return 0.0
    mask = denom > 0
    if not mask.any():
        return 0.0
    val = 2.0 * np.abs(a - p) / np.where(denom == 0, 1.0, denom)
    return float(100.0 * np.mean(val[mask]))


def mase(actual, pred, train) -> float:
    """MASE = MAE(pred,actual) / scale；scale = mean|Δtrain|（NAIVE-1）。"""
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(pred, dtype=np.float64).ravel()
    tr = np.asarray(train, dtype=np.float64).ravel()
    if tr.size < 2:
        return float("nan")
    scale = float(np.mean(np.abs(np.diff(tr))))
    if scale == 0 or not np.isfinite(scale):
        return float("nan")
    return float(np.mean(np.abs(a - p)) / scale)


def evaluate_point_metrics(
    actual, pred, train
) -> Tuple[float, float, float, float]:
    """返回 (MAE, RMSE, sMAPE, MASE)，单窗口单序列。"""
    return (
        mae(actual, pred),
        rmse(actual, pred),
        smape(actual, pred),
        mase(actual, pred, train),
    )
