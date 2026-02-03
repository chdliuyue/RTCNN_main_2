# RTCNN_main_2

本项目包含多种 MNL 变体模型的训练与评估脚本，并提供统一的指标报告（Accuracy、F1、Precision、Recall、混淆矩阵、ECE、Brier score、Elasticities、VoT 等）。

## 环境依赖

建议使用 Python 3.8+，并安装以下依赖：

```bash
pip install -r requirements.txt
```

如果没有 `requirements.txt`，可先安装常见依赖（示例）：

```bash
pip install numpy torch scikit-learn matplotlib adjustText
```

## 数据准备

数据由 `SM_data.py` 加载。默认脚本会直接从该模块读取：

- `X_TRAIN`, `Q_TRAIN`, `y_TRAIN`
- `X_TEST`, `Q_TEST`, `y_TEST`
- `X_vars`

请确保数据文件已准备好并被 `SM_data.py` 正确加载。

## 训练与评估

### 1) 训练并输出所有模型的完整指标（训练集 + 测试集）

```bash
python train.py
```

该脚本会依次训练以下模型，并报告每个模型在训练集和测试集上的指标：

- `E_MNL`
- `EL_MNL`
- `L_MNL`
- `TE_MNL`
- `TEL_MNL`

输出包含 Accuracy、F1、Precision、Recall、混淆矩阵、ECE、Brier score、Elasticities、VoT 等。

### 2) 只做一次训练并输出指标汇总

```bash
python report_metrics.py
```

该脚本同样训练所有模型，但以更简洁的报告形式输出指标（包含训练集与测试集）。

### 3) 生成完整本地化输出（指标、混淆矩阵、Elasticities、VoT、敏感性/消融/不确定性分析）

```bash
python full_analysis.py
```

该脚本会训练所有模型（含 MNL 基线），并把完整指标与图表保存到 `analysis_outputs/` 目录中。

## 其他实验脚本

### 高斯噪声鲁棒性评估

```bash
python gaussian_noise_experiment.py
```

该脚本会在测试集上添加不同强度的高斯噪声，并评估模型鲁棒性。

### 训练损失曲线对比

```bash
python plot_loss_comparison.py
```

或使用带有更丰富样式的版本：

```bash
python plot_loss_comparison_styled.py
```

### 统计检验实验

```bash
python t_statistics_experiment.py
```

## 文件命名规范调整说明

为了提升可读性与维护性，以下脚本已重命名：

- `Gaussian noise.py` → `gaussian_noise_experiment.py`
- `t.py` → `plot_loss_comparison.py`
- `tttt.py` → `plot_loss_comparison_styled.py`
- `t-statistics.py` → `t_statistics_experiment.py`

如有自定义脚本或外部调用，请同步更新引用路径。
