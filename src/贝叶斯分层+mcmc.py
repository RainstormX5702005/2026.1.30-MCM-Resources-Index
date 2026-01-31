import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 屏蔽非关键警告
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 1. 配置与数据加载
# ==========================================
FILE_PATH = "D:/美赛资料/2026/2026_MCM-ICM_Problems/data/bysfc/preprocessed_data_percentage (1).csv" # 请确保文件在当前目录下


def load_and_structure_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 1. 基础索引映射
    # 赛季
    seasons = df['season'].unique()
    season_map = {s: i for i, s in enumerate(sorted(seasons))}
    df['season_idx'] = df['season'].map(season_map)

    # 选手ID (每行一个选手)
    df['contestant_id'] = np.arange(len(df))

    # 2. 提取特征矩阵 (Design Matrix)
    # 行业 One-Hot
    industry_cols = [c for c in df.columns if 'celebrity_industry_' in c]
    X_industry = df[industry_cols].values.astype(float)
    if X_industry.shape[1] == 0:
        X_industry = np.zeros((len(df), 1))  # 兜底

    # 年龄 (已标准化)
    X_age = df['celebrity_age_during_season'].values.astype(float)

    # 低分晋级历史 (作为先验强度)
    if 'low_score_advanced_count_standard' in df.columns:
        prior_strength = df['low_score_advanced_count_standard'].fillna(0).values.astype(float)
    else:
        prior_strength = np.zeros(len(df))

    # 3. 宽表转长表 (Wide to Long) - 用于模型观测
    # 我们需要每一周的观测数据作为一行
    score_cols = [c for c in df.columns if 'week' in c and '_percentage' in c and 'rank' not in c]

    # 保留用于判断淘汰的列
    id_vars = ['contestant_id', 'season_idx', 'weeks_participated', 'placement', 'celebrity_name']

    long_df = df.melt(id_vars=id_vars, value_vars=score_cols,
                      var_name='week_str', value_name='judge_score')

    # 过滤掉 0 分 (代表该周未参赛或已淘汰)
    long_df = long_df[long_df['judge_score'] != 0].copy()

    # 解析周次 (0-based index)
    long_df['week_idx'] = long_df['week_str'].str.extract(r'week(\d+)_percentage').astype(int) - 1

    # 生成全局索引
    long_df = long_df.sort_values(by=['season_idx', 'week_idx', 'contestant_id']).reset_index(drop=True)
    long_df['obs_id'] = long_df.index

    # 4. 构建淘汰对抗对 (Pairwise Constraints)
    # 逻辑：对于每一个淘汰周，所有存活选手的 (Judge + Fan) > 淘汰选手的 (Judge + Fan)
    print("Building pairwise elimination constraints...")
    pairs = []

    # 按赛季和周分组
    groups = long_df.groupby(['season_idx', 'week_idx'])

    # 预先构建存在查找表 (s, w, c_id)
    existence_set = set(zip(long_df['season_idx'], long_df['week_idx'], long_df['contestant_id']))

    for (s, w), group in groups:
        # 检查下一周是否还有任何人参赛 (如果下一周没人参赛，说明本周是赛季最后一周，不计算淘汰)
        # 只要有一人下一周还在，说明本周发生了淘汰/晋级
        
        current_contestants = group['contestant_id'].values
        
        # 找出哪些选手在下一周还在
        survived_contestants = []
        for c in current_contestants:
            if (s, w + 1, c) in existence_set:
                survived_contestants.append(c)
        
        # 如果幸存者为空，说明下周没人了 -> 或者是赛季结束
        if len(survived_contestants) == 0:
            continue
            
        survived_set = set(survived_contestants)
        
        # 那些不在幸存者集合里的，就是本周被淘汰的
        survived_indices = []
        eliminated_indices = []
        
        for idx, row in group.iterrows():
            if row['contestant_id'] in survived_set:
                survived_indices.append(row['obs_id'])
            else:
                eliminated_indices.append(row['obs_id'])

        survived_indices = np.array(survived_indices)
        eliminated_indices = np.array(eliminated_indices)

        if len(eliminated_indices) > 0 and len(survived_indices) > 0:
            # 生成笛卡尔积：所有幸存者 vs 所有淘汰者
            w_grid, l_grid = np.meshgrid(survived_indices, eliminated_indices)
            # 堆叠为 (N, 2) 数组
            pairs.append(np.column_stack((w_grid.ravel(), l_grid.ravel())))

    if pairs:
        constraint_pairs = np.vstack(pairs)
    else:
        constraint_pairs = np.empty((0, 2), dtype=int)

    print(f"Data prepared: {len(long_df)} observations, {len(constraint_pairs)} constraint pairs.")

    return {
        'long_df': long_df,
        'X_industry': X_industry,
        'X_age': X_age,
        'prior_strength': prior_strength,
        'n_contestants': len(df),
        'n_seasons': len(seasons),
        'n_industry_cols': X_industry.shape[1],
        'constraint_pairs': constraint_pairs,
        'seasons_list': sorted(seasons),
        'contestant_names': df['celebrity_name'].values
    }


# ==========================================
# 2. 模型构建与推断
# ==========================================
def build_and_run_model(data, draws=2000, tune=1000):
    long_df = data['long_df']
    obs_c = long_df['contestant_id'].values
    obs_s = long_df['season_idx'].values
    obs_w = long_df['week_idx'].values
    obs_score = long_df['judge_score'].values  # 这是百分比数据

    # 约束索引
    pairs = data['constraint_pairs']
    has_constraints = len(pairs) > 0
    if has_constraints:
        idx_winner = pairs[:, 0]
        idx_loser = pairs[:, 1]

    print("Compiling PyMC model...")

    with pm.Model() as model:
        # --- Priors ---

        # 1. 赛季基准趋势 (Random Walk) - 捕捉评分通胀/紧缩
        sigma_season = pm.HalfNormal("sigma_season", 0.1)
        season_trend = pm.GaussianRandomWalk("season_trend", sigma=sigma_season, shape=data['n_seasons'])

        # 2. 周次效应 (随着周数增加，选手通常表现更好或分数更高)
        beta_week = pm.Normal("beta_week", 0.0, 0.5)

        # 3. 选手潜在能力 (Latent Ability / Popularity)
        # 利用 low_score_advanced_count_standard 作为先验均值的修饰
        # theta > 0 意味着低分晋级次数越多，其潜在人气/能力越高
        theta_prior = pm.Normal("theta_prior", 0.5, 0.5)
        mu_alpha = theta_prior * data['prior_strength']
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)

        # alpha: 每个选手的基准能力
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=data['n_contestants'])

        # 4. 特征回归系数 (行业, 年龄)
        beta_ind = pm.Normal("beta_industry", 0, 1.0, shape=data['n_industry_cols'])
        beta_age = pm.Normal("beta_age", 0, 1.0)

        # 5. 评委评分的权重 (评委分占多大比例反映了总实力)
        beta_judge = pm.HalfNormal("beta_judge", 2.0)

        # --- Model Expected Value (Log-Link) ---
        # 计算特征效应
        # 注意：这里需要矩阵乘法，X_industry 是 (N_contestants, F)
        effect_ind = pm.math.dot(data['X_industry'], beta_ind)

        # 线性组合：选手基准 + 特征 + 赛季趋势 + 周次趋势 + 评委分贡献
        # 我们这里把 judge_score 当作 predictor 之一，来推断 "Total Strength"
        # Total Strength = Latent Fan Base + Judge Score Contribution
        # 但为了拟合观测值，我们通常假设 V_latent ~ Distribution(mu)

        # 构建线性预测子
        log_mu = (
                alpha[obs_c] +
                effect_ind[obs_c] +
                beta_age * data['X_age'][obs_c] +
                season_trend[obs_s] +
                beta_week * obs_w +
                beta_judge * obs_score  # 假设评委分越高，期望的综合实力越高
        )

        # --- Likelihood ---
        # 使用 Gamma 分布建模潜在的 "强度" (必须为正)
        # Gamma 的均值 = mu
        phi = pm.HalfNormal("phi", 5.0)  # 离散参数
        mu_val = pm.math.exp(log_mu)

        # V_latent 是我们需要推断的每一周的“真实综合��分”
        # 我们没有直接观测到 V_latent，但我们有 V_latent 必须满足的约束
        # 在这里，我们把 V_latent 当作一个随机变量，它的先验由上述回归决定
        # 而它的后验由 Elimination Constraint 决定
        V_latent = pm.Gamma("V_latent", alpha=phi, beta=phi / mu_val, shape=len(long_df))

        # --- Constraint Likelihood ---
        if has_constraints:
            # 核心假设：胜者的 V_latent > 败者的 V_latent
            # 这里的 V_latent 已经包含了 Judge Score 的影响 (通过 regression mean)
            # 如果我们要显式分离 Fan Vote，模型会更复杂。这里 V_latent 代表 Total Score (Judge + Fan)

            diff = V_latent[idx_winner] - V_latent[idx_loser]

            # 使用 Sigmoid 概率近似 Heaviside 阶跃函数，使其可微
            # diff > 0 -> p -> 1
            # temperature 系数 10 控制边界的硬度
            p_outcome = pm.math.sigmoid(diff * 5)

            # 观测值为全1（即胜者必须胜）
            pm.Bernoulli("elimination_constraint", p=p_outcome, observed=np.ones(len(pairs)))

        # --- Sampling ---
        print(f"Starting sampling ({draws} draws, {tune} tune)...")
        # 如果有 colab GPU 或 Linux，这里会自动加速。Windows 下使用多核 CPU。
        trace = pm.sample(draws=draws, tune=tune, chains=4, cores=4, target_accept=0.95, return_inferencedata=True)

    return model, trace


# ==========================================
# 3. 可视化与检验
# ==========================================
def analyze_results(data, trace, model):
    print("\n=== Model Diagnostics ===")

    # 1. 诊断统计量
    summary = az.summary(trace, var_names=["sigma_season", "beta_week", "beta_judge", "theta_prior"])
    print(summary)

    # 2. 轨迹图 (Traceplot)
    az.plot_trace(trace, var_names=["beta_judge", "beta_week", "theta_prior"], compact=True)
    plt.tight_layout()
    plt.savefig("model_traceplot.png")
    plt.show()

    # 3. 后验分布图 (Forest Plot for Coefficients)
    plt.figure(figsize=(10, 6))
    az.plot_forest(trace, var_names=["beta_industry"], combined=True)
    plt.title("Impact of Industry on Contestant Strength")
    plt.savefig("industry_effect.png")
    plt.show()

    # 4. 提取潜在强度 (V_latent) 并合并回数据
    # 取后验均值
    v_posterior = trace.posterior["V_latent"].mean(dim=["chain", "draw"]).values
    data['long_df']['estimated_strength'] = v_posterior

    # 5. 赛季趋势可视化
    season_trend_post = trace.posterior["season_trend"].mean(dim=["chain", "draw"]).values
    plt.figure(figsize=(10, 5))
    plt.plot(data['seasons_list'], season_trend_post, marker='o', linestyle='-', color='purple')
    plt.title("Season Baseline Trend (Inflation/Deflation)")
    plt.xlabel("Season")
    plt.ylabel("Baseline Strength Correction")
    plt.grid(True, alpha=0.3)
    plt.savefig("season_trend.png")
    plt.show()

    # 6. 选手排名表 (Top 20 by Average Strength)
    avg_strength = data['long_df'].groupby(['contestant_id', 'celebrity_name'])[
        'estimated_strength'].mean().reset_index()
    top_20 = avg_strength.sort_values('estimated_strength', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='estimated_strength', y='celebrity_name', data=top_20, palette='viridis')
    plt.title("Top 20 Contestants by Estimated Latent Strength (Fan + Judge)")
    plt.xlabel("Latent Strength Score")
    plt.tight_layout()
    plt.savefig("top_contestants.png")
    plt.show()

    # 7. 保存结果
    data['long_df'].to_csv("modeling_results_detailed.csv", index=False)
    print("\nResults saved to 'modeling_results_detailed.csv'")

    return data['long_df']


def predict_eliminations(final_df, seasons_list):
    print("\n=== 执行末尾淘汰预测判别 ===")

    # 1. 准备容器
    prediction_results = []

    # 2. 按赛季和周次分组，模拟每一周的“大逃杀”
    # 只需要分析那些有人被淘汰的周
    groups = final_df.groupby(['season_idx', 'week_idx'])

    correct_predictions = 0
    total_elimination_events = 0

    for (season_idx, week_idx), group in groups:
        # 这一周实际被淘汰的人 (Week_participated == Current Week 且 名次 > 3)
        # 注意：week_idx 是 0-based，week 是 1-based
        current_week = week_idx + 1

        # 找出谁在这一周实际上走了 (Ground Truth)
        # 逻辑：选手的参与周数等于当前周，且他不是前三名（前三名最后一周不算淘汰）
        actual_eliminated = group[
            (group['weeks_participated'] == current_week) &
            (group['placement'] > 3)
            ]

        if len(actual_eliminated) == 0:
            continue  # 这一周没人淘汰（可能是第一周或非淘汰周），跳过

        # 3. 模型的预测
        # 我们的 V_latent (estimated_vote_intensity) 代表综合实力
        # 实力最低的人应该被淘汰

        # 获取本周所有选手的预测强度
        # 按强度从低到高排序
        ranked_contestants = group.sort_values('estimated_vote_intensity', ascending=True)

        # 本周实际淘汰了多少人？（通常是1人，偶尔双淘汰）
        num_to_eliminate = len(actual_eliminated)
        total_elimination_events += num_to_eliminate

        # 模型预测的倒霉蛋（倒数 N 名）
        predicted_eliminated = ranked_contestants.head(num_to_eliminate)

        # 4. 比对结果
        actual_ids = set(actual_eliminated['contestant_id'].values)
        predicted_ids = set(predicted_eliminated['contestant_id'].values)

        # 计算交集（猜对了几个）
        hits = len(actual_ids.intersection(predicted_ids))
        correct_predictions += hits

        # 记录详细日志
        prediction_results.append({
            'Season': seasons_list[season_idx],
            'Week': current_week,
            'Actual_Eliminated_IDs': list(actual_ids),
            'Predicted_Eliminated_IDs': list(predicted_ids),
            'Correct_Count': hits,
            'Total_Eliminated': num_to_eliminate
        })

    # 5. 计算准确率
    accuracy = correct_predictions / total_elimination_events if total_elimination_events > 0 else 0

    print(f"\n预测统计:")
    print(f"总共发生的淘汰事件数: {total_elimination_events}")
    print(f"模型准确预测次数: {correct_predictions}")
    print(f"淘汰预测准确率 (Accuracy): {accuracy:.2%}")

    # 转换为 DataFrame 方便查看
    pred_df = pd.DataFrame(prediction_results)
    pred_df.to_csv("elimination_predictions.csv", index=False)
    print("详细预测结果已保存至 'elimination_predictions.csv'")

    return pred_df


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    try:
        # 1. Load
        data_bundle = load_and_structure_data(FILE_PATH)

        # 2. Run
        model, trace = build_and_run_model(data_bundle, draws=1000, tune=500)

        # 3. Analyze
        final_df = analyze_results(data_bundle, trace, model)

        # 4. Predict
        predict_df = final_df.copy()
        if 'estimated_strength' in predict_df.columns:
            predict_df['estimated_vote_intensity'] = predict_df['estimated_strength']

        predict_eliminations(predict_df, data_bundle['seasons_list'])

        print("Analysis Complete.")

    except FileNotFoundError:
        print(f"Error: Could not find file '{FILE_PATH}'. Please ensure it is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()






