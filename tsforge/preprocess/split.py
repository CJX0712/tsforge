"""preprocess/split — 滚动原点（rolling origin）时序交叉验证切分。

与随机切分根本不同：训练必为历史、测试必为未来，符合时序因果。
每条序列产出 `n_windows` 个 (train, test) 对，基准在其上取均值，估计更稳健。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from tsforge.core.config import Config
from tsforge.core.types import TimeSeries

SplitPair = Tuple[np.ndarray, np.ndarray]


def rolling_origin(
    series: TimeSeries,
    horizon: int,
    n_windows: int = 3,
    min_train: Optional[int] = None,
) -> List[SplitPair]:
    """返回 (train_values, test_values) 列表（test 长度 = horizon）。

    - max_o = n - horizon（测试为最后 horizon 个点时的最大训练终点）
    - min_train 默认 = max(2*horizon, horizon + season_period) 保证训练足量
    - 当窗口退化为 1 或空间不足时，仅返回单一切分（末尾 horizon 点）
    """
    v = series.values
    n = len(v)
    H = int(horizon)
    if n < 2 * H:
        raise ValueError(f"series too short ({n}) for horizon {H}")

    max_o = n - H
    mtrain = int(min_train if min_train is not None else max(3 * H, 3 * H, 3 * 12))
    if n_windows <= 1 or mtrain >= max_o:
        o = max_o
        return [(v[:o].copy(), v[o : o + H].copy())]

    # 在 [mtrain, max_o] 均匀取 n_windows 个互异原点
    idxs = np.linspace(mtrain, max_o, int(n_windows))
    idxs = sorted({int(round(x)) for x in idxs})
    # 去重后若不足，补充分散
    while len(idxs) < int(n_windows) and max_o > mtrain:
        mtrain -= 1
        idxs = sorted({int(round(x)) for x in np.linspace(mtrain, max_o, int(n_windows))})
    out: List[SplitPair] = []
    for o in idxs:
        o = max(mtrain, min(o, max_o))
        out.append((v[:o].copy(), v[o : o + H].copy()))
    return out


def split_dataset(
    series: TimeSeries,
    config: Config,
    min_train: Optional[int] = None,
) -> List[SplitPair]:
    """按 Config 做滚动原点切分。"""
    return rolling_origin(series, config.horizon, config.n_windows, min_train)
