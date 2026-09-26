# tsforge

> 时间序列预测系统 · 作者：**晨星** · v1.0.0
> 复用业界领先开源（statsmodels / scikit-learn / pmdarima），零自研模型，干净环境一键复现。

tsforge 是一套**可实际运行**的单变量时间序列预测与基准系统：内置 7 个预测器（弱基线 + SOTA 统计/ML 分支），严格按 **M4 竞赛指标**与**滚动原点时序交叉验证**评测，可选后端缺失时自动降级，**clone 后零下载即可跑通 demo**。

## 特性

- ✅ **复用优先**：Holt-Winters ETS / SARIMAX（statsmodels）、RandomForest 递归（sklearn）、AutoARIMA（pmdarima），无自研算法。
- ✅ **严格评测**：MAE / RMSE / sMAPE / MASE / OWA，滚动原点时序交叉验证（时间因果）。
- ✅ **离线降级**：`statsmodels` / `pmdarima` 缺失 → `available()=False` 自动跳过，纯 numpy/sklearn 路径保证 demo 可跑。
- ✅ **可独立验证**：每模块单测 + 最小可运行示例；39 个 pytest 用例全绿。
- ✅ **依赖锁定**：`requirements.lock.txt`（pip freeze）保证可复现。
- ✅ **一键复现**：`make` / `Dockerfile` 双通道。

## 快速开始

```bash
# 1) 创建隔离环境并安装锁定依赖
python -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.lock.txt

# 2) 跑端到端基准（默认合成数据）→ 生成 benchmark.json
.venv/bin/python -m tsforge.cli demo

# 3) 查看版本
.venv/bin/python -m tsforge.cli version
```

### 对指定序列做预测

```bash
# airline 经典序列，12 步，SARIMAX
.venv/bin/python -m tsforge.cli forecast --series airline --h 12 --model SARIMAX

# 别名为 ets / arima / rf / sklearn / hw
.venv/bin/python -m tsforge.cli forecast --series synth_0 --h 12 --model ets
```

## 项目结构

```
tsforge/
├── tsforge/
│   ├── core/            # 类型 / 配置 / 错误 / 接口契约
│   ├── data/            # 合成时序 + airline/CSV 加载
│   ├── preprocess/      # 滚动原点切分
│   ├── forecasters/     # 7 个预测器（Naive/SeasonalNaive/Drift/ETS/SARIMAX/ML/AutoARIMA）
│   ├── eval/            # M4 指标 + 基准编排(OWA)
│   ├── pipeline/        # 端到端 ForecastPipeline
│   ├── examples/        # run_demo.py 最小可运行示例
│   └── cli.py           # 命令行入口
├── tests/               # 39 个 pytest 用例
├── docs/architecture.md # 架构图 + 选型依据 + 基线表
├── requirements.txt     # 语义化依赖范围
├── requirements.lock.txt# pip freeze 全量锁定
├── Dockerfile / Makefile / .gitignore
```

## 预测器一览

| 预测器 | 后端 | 说明 |
|--------|------|------|
| Naive | numpy | 最后值（MASE 缩放参照） |
| SeasonalNaive | numpy | 季节周期回填（OWA 参照） |
| Drift | numpy | 线性趋势外推 |
| HoltWinters(ETS) | statsmodels | 加性趋势+季节指数平滑（**综合最优**） |
| SARIMAX | statsmodels | 季节 ARIMA (0,1,1,m) |
| ML(sklearn RF) | scikit-learn | 滞后特征随机森林递归多步 |
| AutoARIMA | pmdarima | 自动定阶（可选） |

## 评测与验收

- **指标**：MAE / RMSE / sMAPE(%) / MASE(vs NAIVE-1) / **OWA**（M4 综合）。
- **切分**：滚动原点，`horizon=12`，默认 `n_windows=3`，训练 ≥ 3·H。
- **默认基线**（本机实测）：Holt-Winters ETS OWA **0.648**、SARIMAX 0.717、ML(RF) 0.761、AutoARIMA 0.743，**全部 < 1**，决定性超越 Naive(1.32) 与 SeasonalNaive(1.0)。
- **验收（DoD）**：`git clone` → 一键脚本 → demo 跑通零干预；pytest 全过；依赖锁定可复现；文档覆盖架构/部署/使用三部分。

> 注：12 步带水平跳变噪声预测下 MASE 绝对值 >1 为正常（真实 M4 月度冠军约 0.85–1.0）；验收以"超越朴素基线 + OWA<1"为准。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q -W ignore::UserWarning
# 39 passed
```

## 许可证

代码以 MIT 许可发布；所复用上游（statsmodels BSD-3 / scikit-learn BSD-3 / pmdarima MIT）遵循各自许可证。
