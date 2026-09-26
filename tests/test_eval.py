"""tests/test_eval — M4 指标正确性（与 sklearn 交叉验证 + 数学不变量）。

覆盖：
- MAE / RMSE 与 sklearn 一致
- sMAPE 已知手算值
- MASE 缩放性质：pred==actual => 0；train/pred 同比例缩放 => MASE 不变；train 过短 => nan
- evaluate_point_metrics 返回 4 元组
"""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsforge.core.types import TimeSeries
from tsforge.eval.metrics import (
    METRICS,
    mae,
    mase,
    rmse,
    smape,
    evaluate_point_metrics,
)


def test_mae_matches_sklearn():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.5, 2.0, 5.0])
    assert abs(mae(a, p) - mean_absolute_error(a, p)) < 1e-12


def test_rmse_matches_sklearn():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.5, 2.0, 5.0])
    assert abs(rmse(a, p) - np.sqrt(mean_squared_error(a, p))) < 1e-12


def test_smape_known_hand_value():
    # a=[100,200], p=[120,220] -> 13.8528%
    a = [100.0, 200.0]
    p = [120.0, 220.0]
    expected = 100.0 * (
        (2 * 20 / 220 + 2 * 20 / 420) / 2
    )
    assert abs(smape(a, p) - expected) < 1e-6


def test_mase_zero_when_perfect():
    a = np.array([1.0, 2.0, 3.0])
    p = a.copy()
    train = np.array([0.0, 1.0, 2.0, 3.0])
    assert abs(mase(a, p, train)) < 1e-12


def test_mase_scale_invariance_under_triple_scaling():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.2, 1.8, 3.1, 3.9])
    train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    m1 = mase(a, p, train)
    k = 3.7
    m2 = mase(a * k, p * k, train * k)
    assert abs(m1 - m2) < 1e-12


def test_mase_nan_for_too_short_train():
    a = np.array([1.0, 2.0])
    p = np.array([1.0, 2.0])
    train = np.array([1.0])  # < 2 个观测
    assert np.isnan(mase(a, p, train))


def test_metrics_dict_keys():
    assert set(METRICS) == {"MAE", "RMSE", "sMAPE", "MASE"}


def test_evaluate_point_metrics_shape_and_order():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.2, 1.8, 3.1, 3.9])
    train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    res = evaluate_point_metrics(a, p, train)
    assert isinstance(res, tuple) and len(res) == 4
    m_mae, m_rmse, m_smape, m_mase = res
    assert abs(m_mae - mae(a, p)) < 1e-12
    assert abs(m_mase - mase(a, p, train)) < 1e-12
