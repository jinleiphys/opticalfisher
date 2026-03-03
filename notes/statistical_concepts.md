# 核物理中常用的统计概念：Fisher Information、Covariance Analysis 与 Bayesian UQ 的关系

**内部笔记** — Jin Lei, 2026-03

---

## 核心等式：一张图说清楚

```
                    同一个矩阵，三个名字
                    ========================

  Fisher Information Matrix (统计学)     Hessian of χ² (优化/拟合)     Covariance Matrix 的逆 (核物理)
         F_{ij}                    =    (1/2) ∂²χ²/∂θ_i∂θ_j          =         C^{-1}_{ij}

  定义:
  F_{ij} = Σ_k (1/σ_k²) (∂y_k/∂θ_i)(∂y_k/∂θ_j)
         = J^T W J

  其中:
    J = Jacobian 矩阵, J_{ki} = ∂y_k/∂θ_i
    W = 权重矩阵, W_{kk} = 1/σ_k²
```

**结论：核物理圈说的 "covariance analysis" 就是 Fisher information matrix 的逆。同一个数学对象，不同学科用不同名字。**

---

## 1. Fisher Information Matrix (FIM)

**来源**: R.A. Fisher, 1925

**定义**: 对于参数 θ 和数据 y，

$$F_{ij} = -E\left[\frac{\partial^2 \ln L}{\partial \theta_i \partial \theta_j}\right] = \sum_k \frac{1}{\sigma_k^2} \frac{\partial y_k}{\partial \theta_i} \frac{\partial y_k}{\partial \theta_j}$$

（第二个等号在高斯误差模型下成立）

**物理含义**: 数据对参数的"信息量"。λ_i 大 = 数据对这个方向敏感（"stiff"），λ_i 小 = 不敏感（"sloppy"）。

**在核物理中的别名**: 很少有人叫它 "Fisher matrix"，通常叫：
- "Hessian matrix"（χ² 的二阶导数矩阵）
- "curvature matrix"
- "sensitivity matrix"（有时候）

---

## 2. Covariance Matrix

**核物理圈最常用的叫法。**

**定义**: 参数不确定度的协方差矩阵

$$C = F^{-1} = (J^T W J)^{-1}$$

**物理含义**: C_{ii} = 参数 θ_i 的方差（即误差棒²），C_{ij} = 参数 θ_i 和 θ_j 的协方差（即相关性）。

**在核物理中的典型用法**:
- UNEDF 合作组（Kortelainen et al., 2010-2014）: 用 χ² Hessian 的逆作为参数协方差
- Reinhard & Nazarewicz (2010, PRC 81): 开创性的 nuclear DFT 协方差分析
- 光学势拟合中的 "error matrix"

**关键限制**: 假设后验分布是高斯的（即 χ² landscape 是抛物面的）。对 sloppy models 来说，这个假设在 sloppy 方向上严重失效。

---

## 3. 后验协方差 = Fisher 逆 (Laplace 近似)

这是连接 frequentist (Fisher/covariance) 和 Bayesian (posterior) 两个世界的桥梁：

**Laplace 近似** (又叫 Gaussian 近似):

在最优参数点 θ̂ 附近，将 log-posterior 展开到二阶：

$$\ln P(\theta|data) \approx \ln P(\hat{\theta}|data) - \frac{1}{2}(\theta - \hat{\theta})^T F (\theta - \hat{\theta})$$

这意味着后验分布近似为高斯：

$$P(\theta|data) \approx \mathcal{N}(\hat{\theta}, F^{-1})$$

所以：

$$\Sigma_\mathrm{post} = F^{-1} = C$$

**三者等价**（在高斯近似下）：
- 后验协方差 Σ_post（Bayesian 语言）
- Fisher 逆 F⁻¹（统计学语言）
- Covariance matrix C（核物理语言）

---

## 4. 为什么核物理圈转向了 Bayesian MCMC？

**时间线**:
- 2010-2014: UNEDF 合作组用 covariance analysis（= Fisher 逆）做 nuclear DFT UQ
- 2015: McDonnell, Schunck et al. (PRL 114, 122501) 转向 Bayesian MCMC + GP emulator
- 2019: King et al. (PRL 122, 232502) 用 MCMC 做光学势 UQ，发现后验强烈非高斯

**转向的原因**: Sloppy models 的后验分布是"香蕉形"的，不是高斯椭球。具体来说：
- Stiff 方向: 近似高斯，covariance 分析 OK
- Sloppy 方向: χ² landscape 几乎平坦，二阶展开失效，高斯近似严重低估不确定度
- King et al. 发现: frequentist 95% CI 的实际覆盖率只有 44-92%（应该是 95%）

**Fisher 分析的价值**（我们论文的立足点）:
- 虽然 F⁻¹ 不能给出准确的后验分布
- 但 F 的**特征值结构**（哪些方向 stiff、哪些 sloppy）是准确的
- D_eff = participation ratio 不依赖高斯假设的具体适用性
- 关键优势: 27 次求解器调用 vs 100,000 次 MCMC 采样（3700 倍加速）

---

## 5. D_eff (Participation Ratio) — 我们论文的核心量

**定义**:

$$D_\mathrm{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

**物理含义**: "有多少个特征值是有效的"
- D_eff = 1: 一个方向主导（极端 sloppy）
- D_eff = N: 所有方向等权（完全约束）

**关键性质**:
- **Scale-invariant**: F → αF 时 D_eff 不变（所有 λ 等比例缩放）
- **物理含义**: 降低实验误差（等比例缩小误差棒）不改变 D_eff
- **只有加入质量上不同的测量**才能改变 D_eff（因为改变了特征值的相对结构）

**核物理没人用过这个量**，这是我们论文的新贡献。之前大家只看：
- Condition number κ = λ_max/λ_min（只看两端，不看整体分布）
- Individual parameter uncertainties（不看参数间的耦合结构）
- PCA / eigenvector decomposition（定性，没有 D_eff 这个定量指标）

---

## 6. Sloppy Models 理论

**来源**: Sethna 组 (Cornell), 2006-2015

**核心发现**: 很多自然科学中的多参数模型都有一个共同特征：FIM 特征值呈近似等间距对数分布

$$\lambda_n \sim \lambda_1 \cdot r^n \quad (r \ll 1)$$

即特征值在对数尺度上近似均匀分布，跨越很多个数量级。

**已发现的 sloppy models 例子**:
- 系统生物学: 信号通路模型（Gutenkunst et al., 2007）
- 粒子物理: 标准模型参数拟合
- 凝聚态: Ising 模型参数
- 核物理 DFT: Nikšić et al., PRC 94, 024333 (2016) — 首次在核物理中明确使用 "sloppy model" 术语
- **核光学势**: 我们的论文 — D_eff ≈ 2/13, 特征值跨 7 个数量级

**物理根源**（对核光学势来说）:
- 弹性散射主要对核表面势敏感
- 不同参数（V, r_v, a_v, W, ...）对表面势的贡献高度相关
- 导致 ∂σ/∂V ∝ ∂σ/∂r_v（Igo ambiguity 的数学表达）

---

## 7. 术语对照表

| 统计学/信息论 | 核物理 (frequentist) | 核物理 (Bayesian) | 我们论文 |
|---|---|---|---|
| Fisher Information Matrix F | Hessian of χ² / Curvature matrix | — | Fisher Information Matrix |
| F⁻¹ | Covariance matrix C | Posterior covariance Σ_post | F⁻¹ (Laplace approx.) |
| Eigenvalues λ_i | — | — | λ_i (信息量 per direction) |
| Condition number κ | Condition number | — | κ = λ_max/λ_min |
| Participation ratio D_eff | **从未使用** | **从未使用** | D_eff（本文首创于核物理） |
| Stiff/sloppy directions | Well/ill-determined combinations | Correlated/uncorrelated params | Stiff/sloppy modes |
| Cramér-Rao bound | "Theoretical minimum uncertainty" | — | — |

---

## 8. 关键参考文献

### Fisher / Covariance 在核物理中的应用
- Reinhard & Nazarewicz, PRC 81, 051303 (2010) — nuclear DFT covariance analysis 的开创性工作
- Kortelainen et al., PRC 85, 024304 (2012) — UNEDF1 参数优化 + covariance
- Dobaczewski, Nazarewicz & Reinhard, J. Phys. G 41, 074001 (2014) — nuclear DFT 误差传播

### Bayesian UQ 在核物理中的兴起
- McDonnell, Schunck et al., PRL 114, 122501 (2015) — 首次大规模 Bayesian MCMC 用于 nuclear DFT
- Schunck & McDonnell, EPJ A 51, 169 (2015) — UQ in nuclear DFT review
- King et al., PRL 122, 232502 (2019) — 光学势的 Bayesian UQ，发现非高斯后验

### Sloppy Models
- Waterfall et al., PRL 97, 150601 (2006) — sloppy model universality class
- Gutenkunst et al., PLoS Comput. Biol. 3, e189 (2007) — sloppy models 在系统生物学中
- Transtrum et al., J. Chem. Phys. 143, 010901 (2015) — sloppy models review
- Nikšić et al., PRC 94, 024333 (2016) — 核 DFT 中的 sloppy models

### 统计学教科书
- Sivia & Skilling, "Data Analysis: A Bayesian Tutorial", 2nd ed., OUP (2006)
- Kay, "Fundamentals of Statistical Signal Processing: Estimation Theory", Prentice Hall (1993)
