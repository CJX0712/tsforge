"""core/types — 全局数据类型（dataclass 容器）。

约定：
- 时序预测为单变量外生无（univariate, exogenous-free）；`TimeSeries.values` 为一维 float。
- 评测语义：误差越小越好；`ForecastResult.mean` 为点预测，`lower/upper` 为预测区间（可选）。
- 基准表 `BenchmarkResult.rows` 为每行一个模型、列为 `指标@horizon` 的 dict 列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TimeSeries:
    """单条单变量时间序列。"""

    values: np.ndarray
    name: str = ""
    timestamps: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64).ravel()
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps)

    @property
    def nobs(self) -> int:
        return int(self.values.shape[0])

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()


@dataclass
class DataSet:
    """多条独立时间序列（panel）；基准在序列间取均值。"""

    series: List[TimeSeries]
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.series = list(self.series)

    def __len__(self) -> int:
        return len(self.series)

    @property
    def names(self) -> List[str]:
        return [s.name or f"s{i}" for i, s in enumerate(self.series)]


@dataclass
class ForecastResult:
    """预测结果：点预测 + 可选区间。"""

    mean: np.ndarray
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    horizon: int = 0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).ravel()
        self.horizon = int(self.mean.shape[0])
        if self.lower is not None:
            self.lower = np.asarray(self.lower, dtype=np.float64).ravel()
        if self.upper is not None:
            self.upper = np.asarray(self.upper, dtype=np.float64).ravel()


@dataclass
class MetricResult:
    """单条指标结果。"""

    name: str
    value: float
    model: str
    h: Optional[int] = None

    def __str__(self) -> str:
        if self.h:
            return f"{self.model} {self.name}@{self.h}={self.value:.4f}"
        return f"{self.model} {self.name}={self.value:.4f}"


@dataclass
class BenchmarkResult:
    """跨模型基准结果。"""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"rows": self.rows}

    def to_table(self, metrics_order: Optional[Sequence[str]] = None) -> str:
        """渲染等宽文本表（CLI 友好，固定列宽避免中英文错位）。"""
        if not self.rows:
            return "(no results)"
        all_keys: set = set()
        for r in self.rows:
            for k in r:
                if k != "model":
                    all_keys.add(k)
        if metrics_order:
            ordered = [m for m in metrics_order if m in all_keys]
            ordered += [m for m in all_keys if m not in metrics_order]
        else:
            ordered = sorted(all_keys)
        header = f"{'model':<22}" + "".join(f"{m:>14}" for m in ordered)
        lines = [header, "-" * len(header)]
        for r in self.rows:
            line = f"{str(r.get('model', '')):<22}"
            for m in ordered:
                v = r.get(m, float("nan"))
                try:
                    line += f"{float(v):>14.4f}"
                except (TypeError, ValueError):
                    line += f"{str(v):>14}"
            lines.append(line)
        return "\n".join(lines)

    def save_json(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


from typing import Sequence  # noqa: E402  (used by to_table signature above)
