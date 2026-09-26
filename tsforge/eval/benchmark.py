"""eval/benchmark — 跨模型基准编排。

遍历模型 -> 跳过不可用/失败 -> 每序列滚动原点评估 -> 聚合为 BenchmarkResult。
OWA(Overall Weighted Average, M4 综合) = 0.5*(MASE/MASE_naive) + 0.5*(sMAPE/sMAPE_naive)。
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from tsforge.core.config import Config
from tsforge.core.types import BenchmarkResult, DataSet, MetricResult, TimeSeries
from tsforge.eval.metrics import mae, rmse, smape, mase
from tsforge.preprocess.split import rolling_origin

_BASE_METRICS = ("MAE", "RMSE", "sMAPE", "MASE")


def evaluate_forecaster(
    fc,
    dataset: DataSet,
    config: Config,
) -> List[MetricResult]:
    """对单个 forecaster 在数据集上做滚动原点评估，返回每指标聚合结果。"""
    H = int(config.horizon)
    acc = {m: 0.0 for m in _BASE_METRICS}
    counts = 0
    for s in dataset.series:
        splits = rolling_origin(s, H, config.n_windows)
        for train_v, test_v in splits:
            try:
                fc.fit(TimeSeries(train_v, name=s.name))
                res = fc.forecast(H)
                pred = np.asarray(res.mean, dtype=np.float64).ravel()
                if len(pred) > len(test_v):
                    pred = pred[: len(test_v)]
                if len(pred) == 0:
                    continue
                # 安全网：过滤爆炸/非有限预测（如短序列上 MLE 未收敛）
                if not np.all(np.isfinite(pred)):
                    continue
                train_scale = float(np.abs(train_v).max()) + float(np.std(train_v)) + 1e-6
                if np.any(np.abs(pred) > 1.0e3 * train_scale):
                    continue
                acc["MAE"] += mae(test_v, pred)
                acc["RMSE"] += rmse(test_v, pred)
                acc["sMAPE"] += smape(test_v, pred)
                acc["MASE"] += mase(test_v, pred, train_v)
                counts += 1
            except Exception:
                continue
    if counts == 0:
        return []
    denom = counts
    return [
        MetricResult(name=f"MAE@{H}", value=acc["MAE"] / denom, model=fc.name, h=H),
        MetricResult(name=f"RMSE@{H}", value=acc["RMSE"] / denom, model=fc.name, h=H),
        MetricResult(name=f"sMAPE@{H}", value=acc["sMAPE"] / denom, model=fc.name, h=H),
        MetricResult(name=f"MASE@{H}", value=acc["MASE"] / denom, model=fc.name, h=H),
    ]


def benchmark(
    models: Iterable,
    dataset: DataSet,
    config: Config,
    verbose: bool = True,
) -> BenchmarkResult:
    rows_list: List[dict] = []
    for model in models:
        name = getattr(model, "name", "?")
        available = getattr(model, "available", lambda: True)()
        if not available:
            if verbose:
                print(f"[skip] {name} (optional backend missing)")
            continue
        try:
            res = evaluate_forecaster(model, dataset, config)
        except Exception as e:  # 单模型失败不拖累其它
            if verbose:
                print(f"[error] {name} fit/eval failed: {e}")
            continue
        if not res:
            if verbose:
                print(f"[warn] {name} produced no results")
            continue
        agg = {"model": name}
        for r in res:
            agg[r.name] = round(float(r.value), 4)
        rows_list.append(agg)

    # OWA 推导：以 SeasonNaive(优先) / Naive 为参照
    ref = None
    for r in rows_list:
        if "SeasonalNaive" in r["model"]:
            ref = r
            break
    if ref is None:
        for r in rows_list:
            if "Naive" in r["model"]:
                ref = r
                break
    if ref is not None:
        H = config.horizon
        mref = ref.get(f"MASE@{H}")
        sref = ref.get(f"sMAPE@{H}")
        if mref and sref and mref > 0 and sref > 0:
            for r in rows_list:
                m = r.get(f"MASE@{H}")
                s = r.get(f"sMAPE@{H}")
                if m and s and np.isfinite(m) and np.isfinite(s):
                    r["OWA"] = round(0.5 * (m / mref) + 0.5 * (s / sref), 4)

    return BenchmarkResult(rows=rows_list)
