"""data/synthetic — 合成时间序列（零下载、含可学习结构，便于公平评测）。

生成逻辑：trend + 多谐波季节 + 中段水平跳变 + 高斯噪声。
因结构真实存在，SARIMAX / Holt-Winters 应显著优于朴素基线（MASE<1）。
固定 random_state 保证基准可复现。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tsforge.core.config import Config
from tsforge.core.types import DataSet, TimeSeries


def make_synthetic(
    config: Optional[Config] = None,
    *,
    n_series: Optional[int] = None,
    n_obs: Optional[int] = None,
    trend: Optional[float] = None,
    seasonality: Optional[float] = None,
    season_period: Optional[int] = None,
    level_shift: Optional[bool] = None,
    noise: Optional[float] = None,
    random_state: Optional[int] = None,
) -> DataSet:
    cfg = config or Config()
    n_series = int(n_series if n_series is not None else cfg.n_series)
    n_obs = int(n_obs if n_obs is not None else cfg.n_obs)
    trend = float(trend if trend is not None else cfg.trend)
    seasonality = float(seasonality if seasonality is not None else cfg.seasonality)
    m = int(season_period if season_period is not None else cfg.season_period)
    level_shift = bool(cfg.level_shift if level_shift is None else level_shift)
    noise = float(noise if noise is not None else cfg.noise)
    rs = int(random_state if random_state is not None else cfg.random_state)

    rng = np.random.default_rng(rs)
    series: list = []
    for i in range(n_series):
        t = np.arange(n_obs, dtype=np.float64)
        y = trend * t
        # 主季节 + 二次谐波（更真实的非正弦形态）
        ph1 = rng.uniform(0.0, 2 * np.pi)
        ph2 = rng.uniform(0.0, 2 * np.pi)
        y = y + seasonality * np.sin(2 * np.pi * t / m + ph1)
        y = y + 0.5 * seasonality * np.sin(4 * np.pi * t / m + ph2)
        if level_shift:
            shift_at = int(rng.integers(int(0.3 * n_obs), int(0.7 * n_obs)))
            y[shift_at:] += rng.uniform(1.0, 2.5)
        # 噪声（相对信号尺度）
        scale = float(np.std(y)) + 1e-6
        y = y + rng.normal(0.0, noise * scale, size=n_obs)
        series.append(TimeSeries(values=y, name=f"synth_{i}"))
    return DataSet(series=series, name="synthetic")


def make_real_airline() -> DataSet:
    """内置经典 Box-Jenkins airline 旅客量（144 月，period=12）。"""
    from tsforge.data.loader import load_airline

    return DataSet(series=[load_airline()], name="airline")
