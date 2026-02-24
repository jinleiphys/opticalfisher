# D_eff 方法对比分析 - 2026-01-16

## 当前状态摘要

| 项目 | 状态 |
|------|------|
| **PRL 推荐结果** | Numerov + 对数导数：D_eff = **1.48 ± 0.45** |
| **NN 训练** | 第二次训练进行中（第一次因 numpy 1.26 崩溃） |
| **自动微分** | 已实现，与有限差分结果一致（差异 0.3%） |
| **训练状态** | 2026-01-16 14:47 开始，预计 ~20 小时完成 |

---

## 问题发现

在对比 NN 和 Numerov 方法计算的 D_eff 时，发现**最大差异达到 205%**。

经调查，发现两种方法使用了**不同的导数定义**：

| 文件 | 方法 | 导数类型 |
|------|------|----------|
| `deff_nn_9params.json` | NN + 有限差分 | **对数导数** ∂log(σ)/∂log(p) |
| `deff_scan_kd02_9params.json` | Numerov + 有限差分 | **直接导数** ∂σ/∂p |

### 导数定义差异

**直接导数**:
```python
gradients[i] = (sigma_plus - sigma_minus) / (2 * delta)  # ∂σ/∂p_i
```

**对数导数**:
```python
# ∂log(σ)/∂log(p) = p/σ · ∂σ/∂p
gradients[i] = params[i] * (sigma_plus - sigma_minus) / (2 * delta) / sigma_0
```

对数导数的优点：
- 无量纲，不同物理量的参数可以直接比较
- 对应相对灵敏度，物理意义更清晰

## 解决方案

创建了新脚本 `deff_scan_kd02_9params_log.py`，使用对数导数重新计算 Numerov 的 D_eff。

### 关键修改

```python
def compute_fisher_matrix_log(...):
    # ...
    for i in range(n_params):
        # LOG-DERIVATIVE: d log(sigma) / d log(p) = p/sigma * dsigma/dp
        gradients[i] = params[i] * (sigma_plus - sigma_minus) / (2 * delta) / (sigma_0 + 1e-20)

    # Fisher matrix with unit weights (log-derivatives are already dimensionless)
    weights = np.ones(n_angles)
    F[i, j] = np.sum(weights * gradients[i] * gradients[j])
```

## 对比结果

### 使用相同导数定义后的 D_eff 对比

| 方法 | 中子 | 质子 | 综合 |
|------|------|------|------|
| **Numerov (对数导数)** | 1.61 ± 0.48 | 1.36 ± 0.38 | **1.48 ± 0.45** |
| **NN (对数导数)** | 1.67 ± 0.41 | 1.62 ± 0.44 | **1.65 ± 0.43** |

### 差异改善

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 平均差异 | 40.74% | **36.99%** |
| 最大差异 | 205.06% | **198.29%** |
| 中位数差异 | 31.80% | **28.23%** |

### 差异最大的配置

| 排名 | 系统 | D_eff (NN) | D_eff (Numerov) | 差异 |
|------|------|------------|-----------------|------|
| 1 | p+208Pb @ 100 MeV | 3.30 | 1.11 | 198% |
| 2 | n+56Fe @ 150 MeV | 2.52 | 1.03 | 144% |
| 3 | n+56Fe @ 10 MeV | 2.60 | 1.09 | 138% |
| 4 | n+27Al @ 150 MeV | 2.53 | 1.07 | 137% |
| 5 | n+16O @ 100 MeV | 2.74 | 1.16 | 136% |

## 重要澄清：两种方法都使用有限差分

**注意**：尽管文件名为 `deff_autograd.py`，但实际上**两种方法都使用有限差分计算梯度**，并非自动微分。

### NN 方法 (`deff_autograd.py` line 275, 305)
```python
# Compute Fisher Information Matrix using neural network + finite difference.
sigma_plus = compute_cross_section_nn(model, params_plus, ...)
sigma_minus = compute_cross_section_nn(model, params_minus, ...)
gradients[i] = (sigma_plus - sigma_minus) / (2 * delta)  # 有限差分！
```

### Numerov 方法 (`deff_scan_kd02_9params_log.py`)
```python
sigma_plus = compute_cross_section(projectile, A, Z, E_lab, theta_deg, params_plus)
sigma_minus = compute_cross_section(projectile, A, Z, E_lab, theta_deg, params_minus)
gradients[i] = params[i] * (sigma_plus - sigma_minus) / (2 * delta)  # 也是有限差分！
```

### 方法对比

| 方法 | 截面计算 | 梯度计算 | 精度 |
|------|----------|----------|------|
| **Numerov** | ODE 精确求解 | 有限差分 | **更准确** |
| **NN** | 神经网络近似 | 有限差分 | 依赖模型质量 |

### 结论：Numerov 更可靠

**Numerov 方法应该更准确**，原因：
1. Numerov 是精确求解 Schrödinger 方程的标准方法
2. NN 只是近似 Numerov 的结果
3. 两者梯度计算方法相同（都是有限差分）
4. 当前 NN 模型还有训练 bug（不同 l 用不同能量）

**PRL 论文应优先使用 Numerov 结果**：D_eff = **1.48 ± 0.45**

## 剩余差异的原因

统一导数定义后，差异从 40.74% 降到 36.99%，但仍有显著差异。

**主要原因**: NN 模型的训练数据生成有 bug（已在 PRC_Neural_Solver 中修复）：
- 原训练：每个 l 值使用独立随机能量
- 问题：截面计算需要所有 l 值在相同能量
- 结果：模型对未见过的 (l, E) 组合泛化能力差

详见 `../PRC_Neural_Solver/TRAINING_STATUS.md`

## 文件清单

| 文件 | 描述 |
|------|------|
| `deff_nn_9params.json` | NN + 有限差分 (对数导数) - 168 配置 |
| `deff_scan_kd02_9params.json` | Numerov + 有限差分 (直接导数) - 旧版 |
| `deff_scan_kd02_9params_log.json` | Numerov + 有限差分 (对数导数) - **推荐** |
| `deff_nn_9params_autograd.json` | NN + 自动微分 (对数导数) - 验证用 |
| `deff_scan_kd02_9params_log.py` | Numerov 对数导数扫描脚本 |
| `recompute_deff_log.py` | NN 有限差分版本脚本 |
| `recompute_deff_log_autograd.py` | NN **真正自动微分**版本脚本 |

## 物理结论

**PRL 论文应采用 Numerov 结果**（更可靠）：

| 结果 | 数值 |
|------|------|
| **D_eff (综合)** | **1.48 ± 0.45** |
| D_eff (中子) | 1.61 ± 0.48 |
| D_eff (质子) | 1.36 ± 0.38 |

核心物理结论：

1. **D_eff ≈ 1.5 << 9**：弹性散射数据只能约束约 1.5 个参数组合
2. **条件数 > 10^6**：逆问题严重病态
3. **V-rv 强关联**：实部势深度和半径高度简并（Igo 模糊性）

## 后续工作

1. **PRL 使用 Numerov 结果** (`deff_scan_kd02_9params_log.json`)
2. **等待新 NN 模型训练完成**（在 alpha1 服务器上运行中）
3. **验证新 NN 模型与 Numerov 的一致性**

## 自动微分版本

创建了真正的自动微分版本 `recompute_deff_log_autograd.py`，使用 `torch.autograd.functional.jacobian`。

### 验证结果

| 方法 | D_eff | V-rv corr | 时间 |
|------|-------|-----------|------|
| Autograd | 2.0593 | 0.1849 | 5.24s |
| 有限差分 | 2.0647 | 0.1979 | 4.19s |
| **差异** | **0.3%** | - | - |

### 结论

- 自动微分结果与有限差分一致（差异 < 1%）
- 速度上没有优势（jacobian 有开销）
- 主要优势：精度更高（无截断误差），在需要二阶导数时更有用

## 运行脚本

```bash
cd /Users/jinlei/Desktop/code/PINN_CFC/PRL_Information_Limit

# Numerov + 对数导数（推荐用于 PRL）
python deff_scan_kd02_9params_log.py
# 输出: deff_scan_kd02_9params_log.json

# NN + 自动微分
python recompute_deff_log_autograd.py
# 输出: deff_nn_9params_autograd.json
```
