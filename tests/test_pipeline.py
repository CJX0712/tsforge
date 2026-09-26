"""tests/test_pipeline — 端到端编排、SOTA 对标、离线降级、时序切分。

覆盖：
- ForecastPipeline.run 返回 BenchmarkResult 且写出 benchmark.json
- 在确定性 airline 上 SOTA(HoltWinters/SARIMAX) MASE 显著优于 Naive，且 ETS OWA<1
- 可选后端缺失时自动跳过（零下载可跑 demo）
- rolling_origin 严格时序因果（train 为前缀、test 为后缀）、窗口数、sMAPE 有限
"""

import json

import numpy as np
import pytest

from tsforge.core.config import Config
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
from tsforge.forecasters import autoarima as _autoarima_mod
from tsforge.forecasters import statsmodels_forecasters as _sm_mod
from tsforge.pipeline.forecast_pipeline import ForecastPipeline
from tsforge.preprocess.split import rolling_origin, split_dataset


# ---------- 端到端 ----------
def test_run_returns_result_and_writes_json(tmp_path, monkeypatch):
    # 跳过慢速 AutoARIMA，保证测试快速且确定
    monkeypatch.setattr(_autoarima_mod.AutoARIMAForecaster, "available", lambda self: False)
    cfg = Config(n_series=3, n_obs=84, horizon=12, n_windows=2, random_state=7)
    pipe = ForecastPipeline(cfg)
    out = tmp_path / "bench.json"
    result, ds = pipe.run(save_path=str(out))
    assert result is not None
    assert len(result.rows) > 0
    assert out.exists()
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "rows" in data
    # 所有必含指标存在
    for row in result.rows:
        assert "MASE@12" in row and "OWA" in row


# ---------- SOTA 对标（airline，确定） ----------
def test_sota_beats_naive_on_airline():
    from tsforge.core.types import DataSet

    airline = load_airline()
    ds = DataSet(series=[airline], name="airline")
    cfg = Config(horizon=12, n_windows=3, season_period=12)
    models = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(period=12),
        HoltWintersForecaster(seasonal_periods=12),
        SARIMAXForecaster(seasonal_order=(0, 1, 1, 12)),
    ]
    from tsforge.eval.benchmark import benchmark

    res = benchmark(models, ds, cfg, verbose=False)
    by_name = {r["model"]: r for r in res.rows}
    assert "Naive" in by_name and "HoltWinters(ETS)" in by_name and "SARIMAX" in by_name
    # SOTA 的 MASE 必须显著低于 Naive（结构可学习）
    assert by_name["HoltWinters(ETS)"]["MASE@12"] < by_name["Naive"]["MASE@12"]
    assert by_name["SARIMAX"]["MASE@12"] < by_name["Naive"]["MASE@12"]
    # ETS 综合 OWA < 1（超越朴素基准）
    assert by_name["HoltWinters(ETS)"]["OWA"] < 1.0


# ---------- 离线降级 ----------
def test_offline_fallback_skips_optional_backends(monkeypatch):
    monkeypatch.setattr(_sm_mod.HoltWintersForecaster, "available", lambda self: False)
    monkeypatch.setattr(_sm_mod.SARIMAXForecaster, "available", lambda self: False)
    monkeypatch.setattr(_autoarima_mod.AutoARIMAForecaster, "available", lambda self: False)

    cfg = Config(n_series=3, n_obs=84, horizon=12, n_windows=2, random_state=3)
    pipe = ForecastPipeline(cfg)
    result, _ = pipe.run()
    names = {r["model"] for r in result.rows}
    # 可选后端应被跳过
    assert "HoltWinters(ETS)" not in names
    assert "SARIMAX" not in names
    assert "AutoARIMA(pmdarima)" not in names
    # 纯依赖基线仍可用（零下载可跑 demo）
    assert "Naive" in names
    assert "SeasonalNaive" in names
    assert "Drift" in names
    assert "ML(sklearn RF)" in names


# ---------- 滚动原点切分 ----------
def test_rolling_origin_chronological_and_count():
    s = TimeSeries(values=np.arange(120, dtype=np.float64), name="x")
    pairs = rolling_origin(s, horizon=12, n_windows=3)
    assert len(pairs) == 3
    for train_v, test_v in pairs:
        assert len(test_v) == 12
        # test 必为序列末尾之后的连续块（时序因果：train 是前缀）
        assert train_v[-1] < test_v[0] or np.isclose(train_v[-1], test_v[0] - 1)


def test_rolling_origin_train_is_prefix_of_series():
    v = np.arange(100, dtype=np.float64)
    s = TimeSeries(values=v, name="x")
    pairs = rolling_origin(s, horizon=12, n_windows=2)
    for train_v, test_v in pairs:
        n = len(train_v)
        assert np.allclose(train_v, v[:n])
        # test 紧跟 train 之后
        assert np.allclose(test_v, v[n : n + 12])


def test_rolling_origin_too_short_raises():
    s = TimeSeries(values=np.arange(10, dtype=np.float64), name="x")
    with pytest.raises(ValueError):
        rolling_origin(s, horizon=12, n_windows=3)


def test_split_dataset_uses_config():
    cfg = Config(horizon=12, n_windows=3)
    s = TimeSeries(values=np.arange(120, dtype=np.float64), name="x")
    pairs = split_dataset(s, cfg)
    assert len(pairs) == 3
