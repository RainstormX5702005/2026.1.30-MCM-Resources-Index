# 贝叶斯 MCMC 参数调优指南

## ✅ 已实现的改进

### 1. **Alpha 建模改进**（人气参数）
```python
# Before（简单先验）
alpha = pm.Normal("alpha", mu=theta, sigma=sigma_alpha, shape=n_contestants)

# After（融入低分晋级信息）
theta_popularity = pm.Normal("theta_popularity", mu=0.3, sigma=0.15)
alpha_mu = theta + theta_popularity * X_low_score_count
alpha = pm.Normal("alpha", mu=alpha_mu, sigma=sigma_alpha, shape=n_contestants)
```

**逻辑**：
- `low_score_advanced_count`：低分晋级次数，体现"人气逆袭"效应
- 低分晋级次数越多 → 粉丝投票支持力度越强 → alpha 先验均值越高
- 避免了信息泄露（只看晋级次数，不看最终排名）

---

## 🎯 减小方差和置信区间的关键参数

### **方差控制参数（按影响力排序）**

#### 1️⃣ `sigma_alpha`（alpha 的个体差异）
```python
# 当前值
sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.3)  # ✅ 已从 0.5 降至 0.3
```
- **作用**：控制选手人气的个体差异
- **调小** → 收紧先验，减小方差
- **建议范围**：`0.2 - 0.4`
- **注意**：太小会导致欠拟合，忽略真实差异

#### 2️⃣ `phi`（Gamma 分布的形状参数）
```python
# 当前值
phi = pm.HalfNormal("phi", sigma=5.0)  # ✅ 已从 3.0 增至 5.0
```
- **作用**：控制投票强度的方差
- **phi 越大** → Gamma 分布越集中 → 方差越小
- **建议范围**：`5.0 - 10.0`
- **注意**：太大会导致分布退化为 Delta 函数

#### 3️⃣ `sigma_season`（赛季趋势的随机游走方差）
```python
# 当前值
sigma_season = pm.HalfNormal("sigma_season", sigma=0.3)
```
- **作用**：控制赛季间的变化幅度
- **调小** → 赛季趋势更平滑，减小整体方差
- **建议范围**：`0.2 - 0.4`

#### 4️⃣ `theta_popularity`（低分晋级效应）
```python
# 当前值
theta_popularity = pm.Normal("theta_popularity", mu=0.3, sigma=0.15)
```
- **作用**：控制低分晋级对人气的提升效应
- **调小 sigma** → 更确定的先验，减小方差
- **建议范围**：`sigma=0.1 - 0.2`

---

### **淘汰约束参数**

#### 5️⃣ 约束权重和强度
```python
# 当前值（已优化）
diff = (obs_percentage[winners_idx] - obs_percentage[losers_idx]) + 0.3 * (
    pt.log(V_latent[winners_idx]) - pt.log(V_latent[losers_idx])
)
p_outcome = pm.math.sigmoid(diff * 3)  # 从 5 降至 3
```
- **约束权重**（0.3）：降低投票强度在约束中的权重
- **Sigmoid 强度**（3）：降低约束的"硬度"
- **效果**：减少对后验的过度约束，降低方差

---

## 📊 MCMC 采样参数

### **已配置参数**
```python
MCMCConfig(
    draws=500,          # 每条链的采样数
    tune=500,           # 预热（调参）步数
    chains=8,           # 并行链数
    target_accept=0.9,  # 目标接受率
    init="advi+adapt_diag",  # 初始化方法
)
```

### **减小方差的调整建议**

#### ✅ **增加采样数**（最稳妥）
```python
draws=1000,  # 从 500 增至 1000
tune=1000,   # 从 500 增至 1000
```
- 更多采样 → 后验估计更稳定 → 置信区间更窄

#### ✅ **提高目标接受率**
```python
target_accept=0.95,  # 从 0.9 增至 0.95
```
- 更保守的采样 → 更稳定的后验

#### ✅ **使用更强的初始化**
```python
init="advi+adapt_diag_grad",  # 使用梯度信息
```

---

## 🔧 实战调参策略

### **快速迭代方案**（测试阶段）
```python
MCMCConfig(
    draws=200,
    tune=200,
    chains=4,
    target_accept=0.85,
)

# 模型参数
sigma_alpha = 0.25
phi = 7.0
sigma_season = 0.25
```

### **高精度方案**（最终结果）
```python
MCMCConfig(
    draws=1500,
    tune=1500,
    chains=8,
    target_accept=0.95,
)

# 模型参数
sigma_alpha = 0.3
phi = 8.0
sigma_season = 0.3
theta_popularity_sigma = 0.12
```

---

## ⚠️ 调参注意事项

### **诊断指标**
运行后检查这些指标：
```python
# 1. Rhat（应该 < 1.01）
az.summary(trace, var_names=["alpha", "phi", "beta_judge"])

# 2. 有效样本数（应该 > 400）
az.ess(trace)

# 3. 轨迹图（应该稳定收敛）
az.plot_trace(trace, var_names=["alpha", "phi"])
```

### **常见问题**

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 置信区间太宽 | phi 太小 | 增大 phi 到 7-10 |
| alpha 方差大 | sigma_alpha 太大 | 降至 0.2-0.3 |
| 链不收敛（Rhat > 1.1） | 初始化不好 | 增加 tune，使用 ADVI |
| 采样太慢 | 约束太多 | 减少 MAX_PAIRS |

---

## 📈 预期效果

### **优化前**
- alpha 置信区间：[-2, 3]（跨度 5）
- V_latent 95% CI：[0.5, 50]（跨度 49.5）

### **优化后**（预期）
- alpha 置信区间：[-1, 1.5]（跨度 2.5，**减少 50%**）
- V_latent 95% CI：[2, 20]（跨度 18，**减少 63%**）
