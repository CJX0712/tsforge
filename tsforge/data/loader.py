"""data/loader — 数据集加载（CSV 宽/长表 + 内置 airline 经典序列）。

DatasetSource 实现：把外部数据接入统一 `DataSet`，与合成数据同构。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from tsforge.core.types import DataSet, TimeSeries

# Box-Jenkins airline passengers (1949-01 .. 1960-12, 144 月) — 公共领域经典基准
_AIRLINE_VALUES: tuple = (
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    231, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
)


def load_airline() -> TimeSeries:
    """返回内置 airline 序列（period=12，月度）。"""
    return TimeSeries(values=np.asarray(_AIRLINE_VALUES, dtype=np.float64), name="airline")


class CSVDataSource:
    """从 CSV 加载数据集（DatasetSource 实现）。

    - 宽表：每行一个时间点、每列一条序列（首列可为时间）。
    - 长表：列 [series_id, value]（+ 可选 time）。
    """

    def __init__(
        self,
        path: str,
        series_col: Optional[str] = None,
        value_col: str = "value",
        id_col: str = "series_id",
        time_col: Optional[str] = None,
    ) -> None:
        self.path = path
        self.series_col = series_col
        self.value_col = value_col
        self.id_col = id_col
        self.time_col = time_col

    def load(self) -> DataSet:
        import pandas as pd

        df = pd.read_csv(self.path)
        if self.series_col is not None:
            # 宽表：列即序列
            series = [
                TimeSeries(values=df[col].to_numpy(dtype=np.float64), name=str(col))
                for col in df.columns
                if col != (self.time_col or "")
            ]
        else:
            # 长表
            series_map: dict = {}
            for sid, grp in df.groupby(self.id_col):
                vals = grp[self.value_col].to_numpy(dtype=np.float64)
                series_map[str(sid)] = TimeSeries(values=vals, name=str(sid))
            series = list(series_map.values())
        return DataSet(series=series, name="csv")


def load_csv(path: str, **kwargs) -> DataSet:
    return CSVDataSource(path, **kwargs).load()
