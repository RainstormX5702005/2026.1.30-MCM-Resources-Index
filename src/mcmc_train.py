"""
贝叶斯层次模型 + MCMC 推断粉丝投票强度
完整版本 - 使用 C 编译加速，多核并行采样
"""

import os

os.environ.setdefault(
    "PYTENSOR_FLAGS", "device=cpu,floatX=float64,optimizer=fast_compile"
)

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import multiprocessing as mp
from typing import Dict, List, Tuple
from dataclasses import dataclass

from configs.config import OUTPUT_DIR

# 设置随机种子
np.random.seed(42)


@dataclass
class MCMCConfig:
    """MCMC 采样配置"""

    draws: int = 500
    tune: int = 500
    chains: int = 8
    cores: int = -1
    target_accept: float = 0.85
    init: str = "jitter+adapt_diag"  # 更稳定的初始化方法

    def __post_init__(self):
        if self.cores == -1:
            self.cores = min(mp.cpu_count(), self.chains)


def load_preprocessed_data(
    file_name: str = "preprocessed_data_percentage.csv",
) -> pd.DataFrame:
    """加载预处理后的数据"""
    file_path = OUTPUT_DIR / "preprocessed" / file_name

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        return df
    except FileNotFoundError as e:
        raise


def prepare_indices(df: pd.DataFrame) -> Tuple[Dict, int, int]:
    """
    准备赛季和选手索引

    Returns:
        (season_map, n_seasons, n_contestants)
    """
    seasons = sorted(df["season"].unique())
    season_map = {s: i for i, s in enumerate(seasons)}

    df["season_idx"] = df["season"].map(season_map)
    df["contestant_id"] = range(len(df))

    n_seasons = len(seasons)
    n_contestants = len(df)

    return season_map, n_seasons, n_contestants


def extract_features(
    df: pd.DataFrame, n_contestants: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接从预处理数据中提取特征（不重复编码）

    Returns:
        (X_industry, X_age)
    """
    industry_cols = [c for c in df.columns if "celebrity_industry_" in c]
    if len(industry_cols) > 0:
        X_industry = df[industry_cols].values.astype(np.float64)
    else:
        X_industry = np.zeros((n_contestants, 1), dtype=np.float64)

    if "celebrity_age_during_season" in df.columns:
        X_age = df["celebrity_age_during_season"].fillna(0).values.astype(np.float64)
    else:
        X_age = np.zeros(n_contestants, dtype=np.float64)

    return X_industry, X_age


def build_observation_data(
    df: pd.DataFrame,
    max_weeks: int = 11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    构建长格式观测数据

    Returns:
        (obs_season_idx, obs_week_idx, obs_contestant_idx, obs_percentage, flat_idx_map)
    """
    obs_season_idx = []
    obs_week_idx = []
    obs_contestant_idx = []
    obs_percentage = []
    flat_idx_map = {}
    current_flat_idx = 0

    for idx, row in df.iterrows():
        c_id = row["contestant_id"]
        s_idx = row["season_idx"]

        for w in range(1, max_weeks + 1):
            col_pct = f"week{w}_percentage"

            if col_pct in df.columns:
                pct_val = row[col_pct]

                if pd.notna(pct_val) and pct_val > 0:
                    obs_season_idx.append(s_idx)
                    obs_week_idx.append(w - 1)
                    obs_contestant_idx.append(c_id)
                    obs_percentage.append(pct_val)
                    flat_idx_map[(c_id, w)] = current_flat_idx
                    current_flat_idx += 1

    return (
        np.array(obs_season_idx, dtype=np.int32),
        np.array(obs_week_idx, dtype=np.int32),
        np.array(obs_contestant_idx, dtype=np.int32),
        np.array(obs_percentage, dtype=np.float64),
        flat_idx_map,
    )


def build_elimination_pairs(
    df: pd.DataFrame,
    season_map: Dict,
    flat_idx_map: Dict,
    max_weeks: int = 11,
) -> Tuple[np.ndarray, Dict]:
    """
    构建淘汰约束配对（避免信息泄露）

    逻辑：
    1. **不使用 placement**（最终排名），避免未来信息泄露
    2. 只看选手是否在下周还有数据：
       - 被淘汰：weeks_participated == w 且 w < season_total_weeks
       - 晋级：weeks_participated > w
    3. 约束：在同一周内，晋级者的综合得分 > 被淘汰者

    Returns:
        (elimination_pairs, pair_info)
        - elimination_pairs: [[winner_idx, loser_idx], ...]
        - pair_info: {pair_idx: {"winner": name, "loser": name, "week": w, "season": s}}
    """
    elimination_pairs = []
    pair_info = {}  # 用于调试验证
    pair_idx = 0

    for s in df["season"].unique():
        s_df = df[df["season"] == s]
        season_total_weeks = s_df["season_total_weeks"].iloc[0]

        # 遍历每一周（除了最后一周，因为最后一周没有淘汰）
        for w in range(1, min(max_weeks, season_total_weeks)):
            week_contestants = []

            for _, row in s_df.iterrows():
                c_id = row["contestant_id"]

                # 检查本周是否有观测数据
                if (c_id, w) not in flat_idx_map:
                    continue

                flat_idx = flat_idx_map[(c_id, w)]
                weeks_part = row["weeks_participated"]

                week_contestants.append(
                    {
                        "flat_idx": flat_idx,
                        "weeks_participated": weeks_part,
                        "contestant_id": c_id,
                        "name": row["celebrity_name"],
                    }
                )

            # 分离晋级者和淘汰者（不看 placement！）
            advanced = []  # 晋级者
            eliminated = []  # 淘汰者

            for c in week_contestants:
                if c["weeks_participated"] == w:
                    # 本周是最后一周 = 被淘汰
                    eliminated.append(c)
                elif c["weeks_participated"] > w:
                    # 继续参赛 = 晋级
                    advanced.append(c)

            # 生成配对：每个晋级者 vs 每个被淘汰者
            for winner in advanced:
                for loser in eliminated:
                    elimination_pairs.append([winner["flat_idx"], loser["flat_idx"]])
                    pair_info[pair_idx] = {
                        "winner": winner["name"],
                        "loser": loser["name"],
                        "week": w,
                        "season": s,
                    }
                    pair_idx += 1

    return (
        (
            np.array(elimination_pairs, dtype=np.int32)
            if elimination_pairs
            else np.array([], dtype=np.int32).reshape(0, 2)
        ),
        pair_info,
    )


def build_pymc_model(
    obs_season_idx: np.ndarray,
    obs_week_idx: np.ndarray,
    obs_contestant_idx: np.ndarray,
    obs_percentage: np.ndarray,
    X_industry: np.ndarray,
    X_age: np.ndarray,
    elimination_pairs: np.ndarray,
    n_seasons: int,
    n_contestants: int,
    n_observations: int,
) -> pm.Model:
    """
    构建完整的贝叶斯层次模型

    模型结构：
    - season_trend: 赛季趋势 (Gaussian Random Walk)
    - beta_week: 周次效应
    - alpha: 选手基础人气（简单先验）
    - beta_judge: 评委分权重
    - beta_industry: 职业特征权重
    - beta_age: 年龄权重
    - V_latent: 潜在投票强度 (Gamma 分布)
    - constraint: 淘汰约束 (Bernoulli)
    """
    n_industry_features = X_industry.shape[1]
    n_pairs = len(elimination_pairs)

    with pm.Model() as model:

        # 1. 赛季趋势 (Gaussian Random Walk)
        sigma_season = pm.HalfNormal("sigma_season", sigma=0.3)
        season_trend = pm.GaussianRandomWalk(
            "season_trend",
            sigma=sigma_season,
            shape=n_seasons,
            init_dist=pm.Normal.dist(0, 0.1),
        )

        beta_week = pm.Normal("beta_week", mu=0, sigma=0.1)

        theta = pm.Normal("theta", mu=0, sigma=0.2)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
        alpha = pm.Normal(
            "alpha",
            mu=theta,
            sigma=sigma_alpha,
            shape=n_contestants,
        )

        # 4. 评委分权重
        beta_judge = pm.Normal("beta_judge", mu=0.5, sigma=0.3)

        # 5. 职业特征权重
        beta_ind = pm.Normal("beta_ind", mu=0, sigma=0.3, shape=n_industry_features)

        # 6. 年龄权重
        beta_age = pm.Normal("beta_age", mu=0, sigma=0.3)

        # === Log-Linear 模型（投票强度） ===
        log_mu = (
            alpha[obs_contestant_idx]
            + beta_judge * obs_percentage
            + pm.math.dot(X_industry, beta_ind)[obs_contestant_idx]
            + beta_age * X_age[obs_contestant_idx]
            + season_trend[obs_season_idx]
            + beta_week * obs_week_idx
        )

        # 潜在票数强度 (Gamma 分布)
        phi = pm.HalfNormal("phi", sigma=3.0)
        mu_ = pm.math.exp(log_mu)
        V_latent = pm.Gamma(
            "V_latent",
            alpha=phi,
            beta=phi / mu_,
            shape=n_observations,
        )

        # === 淘汰约束 ===
        if n_pairs > 0:
            winners_idx = elimination_pairs[:, 0]
            losers_idx = elimination_pairs[:, 1]

            # 约束：晋级者的综合得分 > 淘汰者
            diff = (obs_percentage[winners_idx] - obs_percentage[losers_idx]) + 0.5 * (
                pt.log(V_latent[winners_idx]) - pt.log(V_latent[losers_idx])
            )

            # Sigmoid 概率约束
            p_outcome = pm.math.sigmoid(diff * 5)
            pm.Bernoulli(
                "constraint",
                p=p_outcome,
                observed=np.ones(n_pairs, dtype=np.int32),
            )

    return model


def run_mcmc_sampling(model: pm.Model, config: MCMCConfig) -> az.InferenceData:
    """运行 MCMC 采样"""
    print(f"🚀 Starting MCMC sampling with {config.cores} cores...")
    print(f"   Chains: {config.chains}, Draws: {config.draws}, Tune: {config.tune}")
    print(f"   Init: {config.init}")

    with model:
        trace = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            cores=config.cores,
            target_accept=config.target_accept,
            init=config.init,
            return_inferencedata=True,
            progressbar=True,
        )

    return trace


def extract_results(
    trace: az.InferenceData,
    df: pd.DataFrame,
    obs_season_idx: np.ndarray,
    obs_week_idx: np.ndarray,
    obs_contestant_idx: np.ndarray,
    obs_percentage: np.ndarray,
    season_map: Dict,
) -> pd.DataFrame:
    """提取推断结果"""
    # 提取潜在票数后验
    v_samples = trace.posterior["V_latent"].values  # (chains, draws, observations)
    v_mean = v_samples.mean(axis=(0, 1))
    v_std = v_samples.std(axis=(0, 1))
    v_lower = np.percentile(v_samples, 2.5, axis=(0, 1))
    v_upper = np.percentile(v_samples, 97.5, axis=(0, 1))

    # 反转 season_map
    inv_season_map = {v: k for k, v in season_map.items()}

    # 构造结果表
    results = []
    for i in range(len(obs_percentage)):
        c_idx = obs_contestant_idx[i]
        celeb_name = df.loc[df["contestant_id"] == c_idx, "celebrity_name"].values[0]

        results.append(
            {
                "season": inv_season_map[obs_season_idx[i]],
                "week": obs_week_idx[i] + 1,
                "celebrity_name": celeb_name,
                "contestant_id": c_idx,
                "judge_score_pct": obs_percentage[i],
                "vote_intensity_mean": v_mean[i],
                "vote_intensity_std": v_std[i],
                "vote_intensity_lower_95": v_lower[i],
                "vote_intensity_upper_95": v_upper[i],
            }
        )

    result_df = pd.DataFrame(results)

    # 排序
    result_df = result_df.sort_values(
        ["season", "week", "vote_intensity_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return result_df


def save_results(df: pd.DataFrame, output_file: str) -> None:
    """保存结果"""
    output_path = OUTPUT_DIR / "trained" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")


def main():
    """主函数：执行完整的贝叶斯 MCMC 推断流程"""

    INPUT_FILE = "preprocessed_data_percentage.csv"
    OUTPUT_FILE = "bayesian_vote_intensity.csv"
    MAX_WEEKS = 11

    n_cores = mp.cpu_count()
    mcmc_config = MCMCConfig(
        draws=500,
        tune=500,
        chains=min(n_cores, 8),
        cores=min(n_cores, 8),
        target_accept=0.9,
        init="advi+adapt_diag",
    )

    print("=" * 60)
    print("🔥 Bayesian Hierarchical Model + MCMC Inference")
    print(f"   Using {mcmc_config.cores} CPU cores (max available: {n_cores})")
    print("   C++ compilation enabled for acceleration")
    print("=" * 60)

    # [1/8] 加载数据
    print("\n[1/8] Loading preprocessed data...")
    df = load_preprocessed_data(INPUT_FILE)

    # [2/8] 准备索引
    print("\n[2/8] Preparing indices...")
    season_map, n_seasons, n_contestants = prepare_indices(df)
    print(f"      Seasons: {n_seasons}, Contestants: {n_contestants}")

    # [3/8] 提取特征
    print("\n[3/8] Extracting features...")
    X_industry, X_age = extract_features(df, n_contestants)
    print(f"      Industry features: {X_industry.shape[1]}")

    # [4/8] 构建观测数据
    print("\n[4/8] Building observation data...")
    obs_season_idx, obs_week_idx, obs_contestant_idx, obs_percentage, flat_idx_map = (
        build_observation_data(df, MAX_WEEKS)
    )
    n_observations = len(obs_percentage)
    print(f"      Observations: {n_observations}")

    # [5/8] 构建淘汰约束
    print("\n[5/8] Building elimination constraints...")
    elimination_pairs, pair_info = build_elimination_pairs(
        df, season_map, flat_idx_map, MAX_WEEKS
    )

    # 打印验证信息（前5个配对）
    if len(elimination_pairs) > 0:
        print(f"      Total pairs: {len(elimination_pairs)}")
        print(f"      Sample pairs (验证无信息泄露):")
        for i in range(min(5, len(elimination_pairs))):
            info = pair_info[i]
            print(
                f"        Week {info['week']}, Season {info['season']}: {info['winner']} (晋级) > {info['loser']} (淘汰)"
            )

    # 限制约束数量以加速（随机采样）
    MAX_PAIRS = 200
    if len(elimination_pairs) > MAX_PAIRS:
        idx = np.random.choice(len(elimination_pairs), MAX_PAIRS, replace=False)
        elimination_pairs = elimination_pairs[idx]
        print(f"      Sampled pairs for efficiency: {len(elimination_pairs)}")

    # [6/8] 构建 PyMC 模型
    print("\n[6/8] Building PyMC model...")
    model = build_pymc_model(
        obs_season_idx=obs_season_idx,
        obs_week_idx=obs_week_idx,
        obs_contestant_idx=obs_contestant_idx,
        obs_percentage=obs_percentage,
        X_industry=X_industry,
        X_age=X_age,
        elimination_pairs=elimination_pairs,
        n_seasons=n_seasons,
        n_contestants=n_contestants,
        n_observations=n_observations,
    )
    print("      Model built successfully!")

    # [7/8] 运行 MCMC 采样
    print("\n[7/8] Running MCMC sampling...")
    trace = run_mcmc_sampling(model, mcmc_config)
    print("      Sampling completed!")

    # [8/8] 提取并保存结果
    print("\n[8/8] Extracting and saving results...")
    result_df = extract_results(
        trace=trace,
        df=df,
        obs_season_idx=obs_season_idx,
        obs_week_idx=obs_week_idx,
        obs_contestant_idx=obs_contestant_idx,
        obs_percentage=obs_percentage,
        season_map=season_map,
    )
    save_results(result_df, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("🎉 Bayesian MCMC inference completed!")
    print(f"   Total rows: {len(result_df)}")
    print("=" * 60)

    # 打印示例结果
    print("\n📊 Sample results (first 10 rows):")
    print(
        result_df[
            [
                "celebrity_name",
                "season",
                "week",
                "judge_score_pct",
                "vote_intensity_mean",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
