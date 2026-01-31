import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from configs.config import DATA_DIR, OUTPUT_DIR

# 设置随机种子以确保结果可复现
np.random.seed(42)


def load_data(file_name: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载数据并提取评委打分列

    Args:
        file_name: 数据文件名

    Returns:
        (数据框, 评委打分列名列表)
    """
    try:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path, sep=",", header=0, encoding="utf-8")

        judge_cols = [
            col for col in df.columns if re.match(r"week\d+_judge\d+_score", col)
        ]

        return df, judge_cols

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise


def build_long_judge_table(df: pd.DataFrame, judge_cols: List[str]) -> pd.DataFrame:
    """
    构建长格式的评委打分表（仅包含有效打分 score > 0）

    Args:
        df: 原始数据框
        judge_cols: 评委打分列名列表

    Returns:
        长格式数据框，包含 (season, week, celebrity, judge_score)
    """
    rows = []

    for _, row in df.iterrows():
        for col in judge_cols:
            week_match = re.search(r"week(\d+)_", col)
            if week_match:
                week = int(week_match.group(1))
                score = row[col]

                if pd.notna(score) and score > 0:
                    rows.append(
                        {
                            "season": row["season"],
                            "week": week,
                            "celebrity": row["celebrity_name"],
                            "judge_score": score,
                        }
                    )

    long_df = (
        pd.DataFrame(rows)
        .groupby(["season", "week", "celebrity"])
        .agg(judge_score=("judge_score", "mean"))
        .reset_index()
    )

    return long_df


def get_active_contestants(
    long_df: pd.DataFrame,
) -> Dict[Tuple[int, int], List[str]]:
    """
    获取每个(赛季, 周)下的活跃选手列表

    Args:
        long_df: 长格式数据框

    Returns:
        {(season, week): [celebrity_names]} 字典
    """
    active = {
        (season, week): group["celebrity"].tolist()
        for (season, week), group in long_df.groupby(["season", "week"])
    }

    return active


def find_elimination_events(
    df: pd.DataFrame, judge_cols: List[str]
) -> List[Tuple[int, int, str]]:
    """
    识别淘汰事件 - 淘汰周 = 最后一次有评委打分的周次

    Args:
        df: 原始数据框
        judge_cols: 评委打分列名列表

    Returns:
        [(season, elimination_week, celebrity_name), ...] 列表
    """
    elim_events = []

    judge_total = (
        df.melt(
            id_vars=["season", "celebrity_name"],
            value_vars=judge_cols,
            var_name="col",
            value_name="score",
        )
        .assign(week=lambda x: x["col"].str.extract(r"week(\d+)_").astype(int))
        .groupby(["season", "celebrity_name", "week"])["score"]
        .mean()
        .reset_index(name="judge_total_score")
    )

    for (season, celeb), group in judge_total.groupby(["season", "celebrity_name"]):
        active_weeks = group[group["judge_total_score"] > 0]["week"]

        if len(active_weeks) == 0:
            continue

        last_active = active_weeks.max()

        if last_active < group["week"].max():
            elim_events.append((season, last_active, celeb))

    return elim_events


def initialize_latent_support(
    active: Dict[Tuple[int, int], List[str]],
) -> Dict[Tuple[int, int, str], float]:
    """
    初始化潜在支持度 S 的键值对

    Args:
        active: 活跃选手字典 {(season, week): [celebrity_names]}

    Returns:
        {(season, week, celebrity): 0.0} 字典
    """
    S_keys = [
        (season, week, celeb)
        for (season, week), names in active.items()
        for celeb in names
    ]

    S0 = {key: 0.0 for key in S_keys}

    return S0


def vote_share(
    S: Dict[Tuple[int, int, str], float],
    active: Dict[Tuple[int, int], List[str]],
    season: int,
    week: int,
) -> Dict[str, float]:
    """
    使用 Softmax 计算投票份额

    Args:
        S: 潜在支持度字典
        active: 活跃选手字典
        season: 赛季编号
        week: 周次编号

    Returns:
        {celebrity_name: vote_probability} 字典
    """
    names = active[(season, week)]
    scores = np.array([S[(season, week, name)] for name in names])
    scores -= scores.max()  # 数值稳定性处理
    probabilities = np.exp(scores)
    normalized_probs = probabilities / probabilities.sum()

    return dict(zip(names, normalized_probs))


def log_posterior(
    S: Dict[Tuple[int, int, str], float],
    active: Dict[Tuple[int, int], List[str]],
    elim_events: List[Tuple[int, int, str]],
    sigma: float = 0.4,
    tau: float = 0.3,
) -> float:
    """
    计算对数后验概率
    包含三个部分：基础先验、时间惯性、淘汰似然

    Args:
        S: 潜在支持度字典
        active: 活跃选手字典
        elim_events: 淘汰事件列表
        sigma: 基础先验的标准差
        tau: 时间惯性的标准差

    Returns:
        对数后验概率值
    """
    log_prob = 0.0

    # 基础先验：S ~ N(0, sigma^2)
    for value in S.values():
        log_prob -= 0.5 * (value / sigma) ** 2

    # 时间惯性：S(t) ~ N(S(t-1), tau^2)
    for (season, week, celeb), value in S.items():
        if (season, week - 1, celeb) in S:
            diff = value - S[(season, week - 1, celeb)]
            log_prob -= 0.5 * (diff / tau) ** 2

    # 淘汰似然：被淘汰选手的投票概率应该较低
    for season, week, eliminated in elim_events:
        if (season, week) not in active:
            continue
        probs = vote_share(S, active, season, week)
        p_elim = probs.get(eliminated, 1e-12)
        log_prob += np.log(1.0 - p_elim + 1e-12)

    return log_prob


def single_chain_mcmc(
    chain_id: int,
    S0: Dict[Tuple[int, int, str], float],
    active: Dict[Tuple[int, int], List[str]],
    elim_events: List[Tuple[int, int, str]],
    n_iter: int,
    burnin: int,
    thin: int,
    delta: float,
    sigma: float,
    tau: float,
) -> List[Dict[Tuple[int, int, str], float]]:
    """
    单链MCMC采样（全局函数，用于进程池）

    Args:
        chain_id: 链的ID（用于设置随机种子）
        S0: 初始潜在支持度
        active: 活跃选手字典
        elim_events: 淘汰事件列表
        n_iter: 迭代次数
        burnin: 燃烧期
        thin: 稀疏采样间隔
        delta: 提议分布的标准差
        sigma: 基础先验的标准差
        tau: 时间惯性的标准差

    Returns:
        该链的采样结果
    """
    # 为每条链设置不同的随机种子
    np.random.seed(42 + chain_id)

    S = S0.copy()
    keys = list(S.keys())
    samples = []

    log_prob = log_posterior(S, active, elim_events, sigma, tau)

    for iteration in range(n_iter):
        # 随机选择一个键并提议新值
        key = keys[np.random.randint(len(keys))]
        S_new = S.copy()
        S_new[key] += np.random.normal(0, delta)

        # 计算新的对数后验概率
        log_prob_new = log_posterior(S_new, active, elim_events, sigma, tau)

        # Metropolis-Hastings 接受/拒绝
        if np.log(np.random.rand()) < log_prob_new - log_prob:
            S = S_new
            log_prob = log_prob_new

        # 保存样本（燃烧期后，每隔thin次）
        if iteration >= burnin and iteration % thin == 0:
            samples.append(S.copy())

    return samples


def run_mcmc(
    S0: Dict[Tuple[int, int, str], float],
    active: Dict[Tuple[int, int], List[str]],
    elim_events: List[Tuple[int, int, str]],
    n_iter: int = 30000,
    burnin: int = 8000,
    thin: int = 40,
    delta: float = 0.08,
    sigma: float = 0.4,
    tau: float = 0.3,
    n_chains: int = 4,
) -> List[Dict[Tuple[int, int, str], float]]:
    """
    运行多链 Metropolis-Hastings MCMC 采样

    Args:
        S0: 初始潜在支持度
        active: 活跃选手字典
        elim_events: 淘汰事件列表
        n_iter: 迭代次数
        burnin: 燃烧期（丢弃前N次采样）
        thin: 稀疏采样间隔
        delta: 提议分布的标准差
        sigma: 基础先验的标准差
        tau: 时间惯性的标准差
        n_chains: MCMC链的数量（用于并行化）

    Returns:
        采样结果列表（合并所有链的结果）
    """
    print(f"      Running {n_chains} MCMC chains in parallel...")

    # 使用进程池并行运行多条MCMC链
    with ProcessPoolExecutor(max_workers=min(n_chains, mp.cpu_count())) as executor:
        futures = [
            executor.submit(
                single_chain_mcmc,
                chain_id=i,
                S0=S0,
                active=active,
                elim_events=elim_events,
                n_iter=n_iter,
                burnin=burnin,
                thin=thin,
                delta=delta,
                sigma=sigma,
                tau=tau,
            )
            for i in range(n_chains)
        ]
        chain_results = [future.result() for future in as_completed(futures)]

    # 合并所有链的结果
    all_samples = []
    for chain_samples in chain_results:
        all_samples.extend(chain_samples)

    print(f"      Collected {len(all_samples)} total samples from {n_chains} chains")

    return all_samples


def process_single_sample(
    S: Dict[Tuple[int, int, str], float],
    active: Dict[Tuple[int, int], List[str]],
) -> Dict[Tuple[int, int, str], float]:
    """
    处理单个样本的投票份额计算（全局函数，用于进程池）

    Args:
        S: 单个MCMC样本
        active: 活跃选手字典

    Returns:
        该样本的投票份额字典
    """
    sample_probs = {}
    for (season, week), names in active.items():
        probs = vote_share(S, active, season, week)
        for celeb, prob in probs.items():
            sample_probs[(season, week, celeb)] = prob
    return sample_probs


def collect_vote_share_samples(
    samples: List[Dict[Tuple[int, int, str], float]],
    active: Dict[Tuple[int, int], List[str]],
) -> Dict[Tuple[int, int, str], List[float]]:
    """
    从 MCMC 样本中收集投票份额（并行化版本）

    Args:
        samples: MCMC 采样结果
        active: 活跃选手字典

    Returns:
        {(season, week, celebrity): [vote_shares]} 字典
    """
    print(f"      Processing {len(samples)} samples in parallel...")

    # 使用进程池并行处理样本
    vote_samples = defaultdict(list)

    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
        futures = [executor.submit(process_single_sample, S, active) for S in samples]

        for future in as_completed(futures):
            sample_probs = future.result()
            for key, prob in sample_probs.items():
                vote_samples[key].append(prob)

    print(
        f"      Collected vote shares for {len(vote_samples)} (season, week, celebrity) combinations"
    )

    return vote_samples


def compute_single_stat(key_values: Tuple[Tuple[int, int, str], List[float]]) -> Dict:
    """
    计算单个统计量的函数（全局函数，用于进程池）

    Args:
        key_values: ((season, week, celebrity), values_list) 元组

    Returns:
        统计结果字典
    """
    (season, week, celeb), values = key_values
    values_array = np.array(values)

    return {
        "season": season,
        "week": week,
        "celebrity_name": celeb,
        "vote_mean": values_array.mean(),
        "vote_lower_95": np.percentile(values_array, 2.5),
        "vote_upper_95": np.percentile(values_array, 97.5),
        "vote_sd": values_array.std(),
    }


def compute_posterior_statistics(
    vote_samples: Dict[Tuple[int, int, str], List[float]],
) -> pd.DataFrame:
    """
    计算后验统计量：均值和95%置信区间（并行化版本）

    Args:
        vote_samples: 投票份额样本字典

    Returns:
        包含统计结果的数据框
    """
    print(
        f"      Computing statistics for {len(vote_samples)} combinations in parallel..."
    )

    # 使用进程池并行计算统计量
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), 8)) as executor:
        futures = [
            executor.submit(compute_single_stat, item) for item in vote_samples.items()
        ]
        rows = [future.result() for future in as_completed(futures)]

    result_df = (
        pd.DataFrame(rows)
        .sort_values(["season", "week", "vote_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    return result_df


def save_results(df: pd.DataFrame, output_file: str) -> None:
    """
    保存结果到文件

    Args:
        df: 结果数据框
        output_file: 输出文件名
    """
    output_path = OUTPUT_DIR / "trained" / output_file
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")


def main():
    """主函数：执行完整的 MCMC 推断流程"""

    # 配置参数
    INPUT_FILE = "2026_MCM_Problem_C_Data.csv"
    OUTPUT_FILE = "weekly_audience_vote_share_with_95CI_STABLE.csv"

    # MCMC 参数
    N_CHAINS = min(mp.cpu_count(), 8)  # 根据CPU核心数自动调整链数
    N_ITER = 30000
    BURNIN = 8000
    THIN = 40
    DELTA = 0.08
    SIGMA = 0.4
    TAU = 0.3

    print("=" * 60)
    print("Starting MCMC-based Audience Vote Share Estimation")
    print(f"Using {N_CHAINS} CPU cores for parallel processing")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/8] Loading data...")
    df, judge_cols = load_data(INPUT_FILE)
    print(f"      Loaded {len(df)} records with {len(judge_cols)} judge columns")

    # 2. 构建长格式评委打分表
    print("\n[2/8] Building long-format judge table...")
    long_df = build_long_judge_table(df, judge_cols)
    print(f"      Created {len(long_df)} judge score entries")

    # 3. 获取活跃选手
    print("\n[3/8] Identifying active contestants...")
    active = get_active_contestants(long_df)
    print(f"      Found {len(active)} (season, week) combinations")

    # 4. 识别淘汰事件
    print("\n[4/8] Finding elimination events...")
    elim_events = find_elimination_events(df, judge_cols)
    print(f"      Identified {len(elim_events)} elimination events")

    # 5. 初始化潜在支持度
    print("\n[5/8] Initializing latent support variables...")
    S0 = initialize_latent_support(active)
    print(f"      Initialized {len(S0)} latent variables")

    # 6. 运行 MCMC
    print("\n[6/8] Running MCMC sampling...")
    print(
        f"      Chains: {N_CHAINS}, Iterations per chain: {N_ITER}, Burnin: {BURNIN}, Thin: {THIN}"
    )
    samples = run_mcmc(
        S0,
        active,
        elim_events,
        n_iter=N_ITER,
        burnin=BURNIN,
        thin=THIN,
        delta=DELTA,
        sigma=SIGMA,
        tau=TAU,
        n_chains=N_CHAINS,
    )
    print(f"      Total samples collected: {len(samples)}")

    # 7. 收集投票份额样本
    print("\n[7/8] Collecting vote share samples...")
    vote_samples = collect_vote_share_samples(samples, active)

    # 8. 计算后验统计量并保存
    print("\n[8/8] Computing posterior statistics...")
    result_df = compute_posterior_statistics(vote_samples)
    save_results(result_df, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("MCMC inference completed successfully!")
    print(f"Results saved with {len(result_df)} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
