"""config — 全局配置（支持环境变量覆盖 TSFORGE_*）。

所有可调超参集中于此，模块只读 Config，不自行散落常量，便于复现与基准控制。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

ENV_PREFIX = "TSFORGE_"


@dataclass
class Config:
    # 随机与合成数据
    random_state: int = 42
    n_series: int = 6           # 合成序列条数（panel 基准取均值）
    n_obs: int = 120            # 每条序列长度
    trend: float = 0.04         # 线性趋势斜率
    seasonality: float = 1.0    # 季节振幅
    season_period: int = 12     # 季节周期（默认月度）
    level_shift: bool = True    # 是否注入中段水平跳变
    noise: float = 0.15         # 噪声幅度（相对信号 std）

    # 评测
    horizon: int = 12           # 预测步长 H
    n_windows: int = 3          # 滚动原点窗口数（时序交叉验证）

    # 预测器
    autoarima_enabled: bool = True

    # 输出
    topn: int = 0

    @classmethod
    def from_env(cls) -> "Config":
        """用环境变量覆盖：TSFORGE_N_SERIES=8 等。"""
        overrides: dict = {}
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        for name in fields:
            env = ENV_PREFIX + name.upper()
            if env in os.environ:
                overrides[name] = _coerce(os.environ[env], fields[name].type)
        return cls(**overrides)


def _coerce(raw: str, typ):
    raw = raw.strip()
    t = str(typ)
    if "int" in t:
        return int(raw)
    if "float" in t:
        return float(raw)
    if "bool" in t:
        return raw.lower() in ("1", "true", "yes", "y")
    if "tuple" in t or "list" in t:
        return tuple(int(x) for x in raw.split(",") if x.strip())
    return raw
