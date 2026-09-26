"""tests/test_forecasters — 预测器契约与行为。

覆盖：
- 基线(Naive/SeasonalNaive/Drift)与 SOTA(HoltWinters/SARIMAX/ML) 预测长度 == horizon
- 拟合前预测应抛 RuntimeError
- 可选后端 available() 契约（已安装时 True）
- 预测结果有限且长度正确（基准安全网前提）
"""

import numpy as np
import pytest

from tsforge.core.types import TimeSeries
from tsforge.data.loader import load_airline
from tsforge.forecasters import (
    AutoARIMAForecaster,
    DriftForecaster,
    HoltWintersForecaster,
    MLForecaster,
    NaiveForecaster,
    SARIMAXForecaster,
    SeasonalNaiveForecaster,
)


@pytest.fixture
def airline():
    return load_airline()


@pytest.fixture
def small_series():
    rng = np.random.default_rng(0)
    y = np.arange(60, dtype=np.float64) + 5.0 * np.sin(
        2 * np.pi * np.arange(60) / 12
    ) + rng.normal(0, 0.5, 60)
    return TimeSeries(values=y, name="toy")


def _all_finite(r):
    return np.all(np.isfinite(r.mean)) and (
        r.lower is None or np.all(np.isfinite(r.lower))
    )


def test_naive_forecast_length_and_value(small_series):
    m = NaiveForecaster()
    m.fit(small_series)
    r = m.forecast(12)
    assert len(r.mean) == 12
    assert np.allclose(r.mean, small_series.values[-1])


def test_seasonal_naive_periodic(small_series):
    m = SeasonalNaiveForecaster(period=12)
    m.fit(small_series)
    r = m.forecast(12)
    assert len(r.mean) == 12
    assert _all_finite(r)


def test_drift_linear_extrapolation(small_series):
    m = DriftForecaster()
    m.fit(small_series)
    r = m.forecast(3)
    assert len(r.mean) == 3
    # 斜率恒定：相邻差相等
    diffs = np.diff(r.mean)
    assert np.allclose(diffs, diffs[0], atol=1e-9)


def test_holt_winters_available_and_forecast(airline):
    m = HoltWintersForecaster(seasonal_periods=12)
    assert m.available() is True
    m.fit(airline)
    r = m.forecast(12)
    assert len(r.mean) == 12
    assert _all_finite(r)
    assert r.lower is not None and len(r.lower) == 12


def test_sarimax_available_and_forecast(airline):
    m = SARIMAXForecaster(seasonal_order=(0, 1, 1, 12))
    assert m.available() is True
    m.fit(airline)
    r = m.forecast(12)
    assert len(r.mean) == 12
    assert _all_finite(r)
    assert r.lower is not None and len(r.lower) == 12


def test_ml_forecast_length_and_finite(small_series):
    m = MLForecaster(lag=12, random_state=42)
    assert m.available() is True
    m.fit(small_series)
    r = m.forecast(12)
    assert len(r.mean) == 12
    assert _all_finite(r)


def test_autoarima_available_true_when_installed():
    # 本环境已装 pmdarima；契约：available()==True
    m = AutoARIMAForecaster(m=12)
    assert m.available() is True


def test_forecast_before_fit_raises(small_series):
    with pytest.raises(RuntimeError):
        NaiveForecaster().forecast(5)
    with pytest.raises(RuntimeError):
        SARIMAXForecaster().forecast(5)


def test_ml_degenerate_fallback_on_tiny_series():
    # 序列过短无法构造滞后特征时退化为常数均值（不崩）
    s = TimeSeries(values=[1.0, 2.0, 3.0], name="tiny")
    m = MLForecaster(lag=12)
    m.fit(s)
    r = m.forecast(4)
    assert len(r.mean) == 4
    assert np.allclose(r.mean, r.mean[0])  # 常数
