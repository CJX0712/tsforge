"""tests/test_core — 核心数据类型、配置、错误码契约。

覆盖：
- TimeSeries / DataSet / ForecastResult / BenchmarkResult 结构不变量
- Config.from_env 环境变量覆盖
- 分层错误码
"""

import json
import os

import numpy as np
import pytest

from tsforge.core.config import Config, ENV_PREFIX
from tsforge.core.errors import (
    ConfigError,
    DataError,
    EvalError,
    ModelError,
    TSForgeError,
)
from tsforge.core.types import (
    BenchmarkResult,
    DataSet,
    ForecastResult,
    MetricResult,
    TimeSeries,
)


# ---------- TimeSeries ----------
def test_timeseries_coerces_to_float64_and_ravel():
    s = TimeSeries(values=[1, 2, 3, 4], name="x")
    assert s.values.dtype == np.float64
    assert s.values.shape == (4,)
    assert s.nobs == 4
    assert s.name == "x"


def test_timeseries_to_numpy_is_copy():
    s = TimeSeries(values=[1.0, 2.0], name="x")
    arr = s.to_numpy()
    arr[0] = 999.0
    assert s.values[0] == 1.0  # 副本隔离


# ---------- DataSet ----------
def test_dataset_len_and_names():
    ds = DataSet(
        series=[
            TimeSeries(values=[1, 2], name="a"),
            TimeSeries(values=[3, 4], name="b"),
        ]
    )
    assert len(ds) == 2
    assert ds.names == ["a", "b"]


def test_dataset_default_name_fallback():
    ds = DataSet(series=[TimeSeries(values=[1]), TimeSeries(values=[2])])
    assert ds.names == ["s0", "s1"]


# ---------- ForecastResult ----------
def test_forecast_result_ravel_horizon_and_interval():
    mean = [1.0, 2.0, 3.0]
    lo = [0.5, 1.5, 2.5]
    hi = [1.5, 2.5, 3.5]
    r = ForecastResult(mean=mean, lower=lo, upper=hi)
    assert r.mean.shape == (3,)
    assert r.horizon == 3
    assert r.lower.shape == (3,)
    assert r.upper.shape == (3,)


def test_forecast_result_default_no_interval():
    r = ForecastResult(mean=[1.0, 2.0])
    assert r.lower is None
    assert r.upper is None
    assert r.horizon == 2


# ---------- BenchmarkResult ----------
def test_benchmark_to_table_contains_models_and_metrics():
    br = BenchmarkResult(
        rows=[
            {"model": "Naive", "MASE@12": 2.25, "RMSE@12": 1.40},
            {"model": "HoltWinters(ETS)", "MASE@12": 1.12, "RMSE@12": 0.70},
        ]
    )
    table = br.to_table()
    assert "Naive" in table
    assert "HoltWinters(ETS)" in table
    assert "MASE@12" in table
    assert "RMSE@12" in table


def test_benchmark_to_table_empty():
    br = BenchmarkResult(rows=[])
    assert br.to_table() == "(no results)"


def test_benchmark_save_json_roundtrip(tmp_path):
    br = BenchmarkResult(rows=[{"model": "Naive", "MASE@12": 2.25}])
    p = tmp_path / "bench.json"
    br.save_json(str(p))
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["rows"][0]["model"] == "Naive"


def test_metric_result_str_format():
    m = MetricResult(name="MASE", value=1.23, model="ETS", h=12)
    assert "ETS" in str(m) and "MASE" in str(m) and "@12" in str(m)


# ---------- Config ----------
def test_config_defaults_are_tuned():
    c = Config()
    assert c.n_series == 6
    assert c.n_obs == 120
    assert c.trend == 0.04
    assert c.noise == 0.15
    assert c.season_period == 12
    assert c.horizon == 12
    assert c.n_windows == 3


def test_config_from_env_override(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}N_SERIES", "8")
    monkeypatch.setenv(f"{ENV_PREFIX}NOISE", "0.3")
    monkeypatch.setenv(f"{ENV_PREFIX}LEVEL_SHIFT", "false")
    c = Config.from_env()
    assert c.n_series == 8
    assert abs(c.noise - 0.3) < 1e-12
    assert c.level_shift is False


def test_config_from_env_int_and_float_coercion(monkeypatch):
    monkeypatch.setenv(f"{ENV_PREFIX}HORIZON", "24")
    monkeypatch.setenv(f"{ENV_PREFIX}TREND", "0.1")
    c = Config.from_env()
    assert c.horizon == 24
    assert isinstance(c.trend, float)
    assert abs(c.trend - 0.1) < 1e-12


# ---------- Errors ----------
def test_error_hierarchy_and_codes():
    assert issubclass(ConfigError, TSForgeError)
    assert issubclass(DataError, TSForgeError)
    assert issubclass(ModelError, TSForgeError)
    assert issubclass(EvalError, TSForgeError)
    assert TSForgeError().code == "E000"
    assert ConfigError().code == "E100"
    assert DataError().code == "E200"
    assert ModelError().code == "E300"
    assert EvalError().code == "E400"


def test_error_is_raisable():
    with pytest.raises(TSForgeError):
        raise ModelError("boom")
