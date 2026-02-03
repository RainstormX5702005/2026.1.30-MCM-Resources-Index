"""
灵敏度分析：基于置信区间的误差棒图
- 淘汰边缘选手：评委分低但仍然晋级的人（置信区间较窄）
- 稳晋级选手：评委分高、投票也高（置信区间较宽）
- 稳淘汰选手：评委分低、投票也低（置信区间较宽）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from configs.config import DATA_DIR, OUTPUT_DIR

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_name_mapping():
    """加载 contestant_id 到真实姓名和赛季的映射"""
    # 从原始数据加载
    raw_df = pd.read_csv(DATA_DIR / "2026_MCM_Problem_C_Data.csv")

    # 创建 contestant_id（每行一个唯一的选手）
    raw_df = raw_df.reset_index()
    raw_df = raw_df.rename(columns={"index": "contestant_id"})

    # 创建映射字典
    name_mapping = dict(zip(raw_df["contestant_id"], raw_df["celebrity_name"]))
    season_mapping = dict(zip(raw_df["contestant_id"], raw_df["season"]))

    return name_mapping, season_mapping, raw_df


def find_edge_cases_pct(df):
    """
    从 pct 数据中找出三类典型选手：
    1. 淘汰边缘：评委分低（排名数字大）但仍然晋级的人
    2. 稳晋级：评委分高（排名数字小）且最终名次好
    3. 稳淘汰：评委分低（排名数字大）且最终被淘汰
    """
    # 按赛季和周分组，计算评委分的排名
    # ascending=False: 分数高的 rank 小（1,2,3），分数低的 rank 大（10,11,12）
    df = df.copy()
    df["judge_rank_in_week"] = df.groupby(["season", "week_idx"])["judge_score"].rank(
        ascending=False
    )
    df["n_contestants"] = df.groupby(["season", "week_idx"])["judge_score"].transform(
        "count"
    )

    # 计算置信区间宽度
    df["ci_width"] = df["audience_upper_95"] - df["audience_lower_95"]

    # 1. 淘汰边缘选手：评委排名差（数字大，比如10/12），但最终没被淘汰
    # 这种人靠粉丝投票救了回来
    edge_candidates = df[
        (
            df["judge_rank_in_week"] >= df["n_contestants"] * 0.65
        )  # 评委排名后35%（数字大）
        & (df["placement"] < df["n_contestants"])  # 但未被淘汰
    ].copy()

    # 按置信区间宽度排序（越窄说明约束越紧）
    edge_candidates = edge_candidates.sort_values("ci_width", ascending=True)
    # 去重：每个选手只保留一条（置信区间最窄的）
    edge_candidates = edge_candidates.drop_duplicates(
        subset=["celebrity_name"], keep="first"
    )

    # 2. 稳晋级选手：评委排名好（数字小，比如1,2,3）且最终名次好
    safe_candidates = df[
        (df["judge_rank_in_week"] <= df["n_contestants"] * 0.3)  # 评委排名前30%
        & (df["placement"] <= 3)  # 最终名次好
    ].copy()
    safe_candidates = safe_candidates.sort_values("ci_width", ascending=False)
    # 去重：每个选手只保留一条（置信区间最宽的）
    safe_candidates = safe_candidates.drop_duplicates(
        subset=["celebrity_name"], keep="first"
    )

    # 3. 稳淘汰选手：评委排名差（数字大）且被淘汰
    elim_candidates = df[
        (df["judge_rank_in_week"] >= df["n_contestants"] * 0.7)  # 评委排名后30%
        & (df["placement"] == df["n_contestants"])  # 被淘汰
    ].copy()
    elim_candidates = elim_candidates.sort_values("ci_width", ascending=False)
    # 去重：每个选手只保留一条（置信区间最宽的）
    elim_candidates = elim_candidates.drop_duplicates(
        subset=["celebrity_name"], keep="first"
    )

    return edge_candidates, safe_candidates, elim_candidates


def find_edge_cases_rank(df):
    """
    从 rank 数据中找出三类典型选手
    注意：rank 数据中 judge_rank 已经是排名（1最好），数字越大排名越差
    """
    df = df.copy()

    # 计算每周的参赛人数
    df["n_contestants"] = df.groupby(["season_idx", "week_idx"])[
        "contestant_id"
    ].transform("count")

    # 计算置信区间宽度
    df["ci_width"] = df["implied_audience_upper_95"] - df["implied_audience_lower_95"]

    # 1. 淘汰边缘选手：评委排名差（数字大）但未被淘汰
    edge_candidates = df[
        (
            df["judge_rank"] >= df["n_contestants"] * 0.65
        )  # 评委排名后35%（数字大=排名差）
        & (df["placement"] < df["n_contestants"])  # 未被淘汰
    ].copy()
    edge_candidates = edge_candidates.sort_values("ci_width", ascending=True)
    # 去重
    edge_candidates = edge_candidates.drop_duplicates(
        subset=["contestant_id"], keep="first"
    )

    # 2. 稳晋级选手：评委排名好（数字小）且名次好
    safe_candidates = df[
        (df["judge_rank"] <= df["n_contestants"] * 0.3)  # 评委排名前30%
        & (df["placement"] <= 3)
    ].copy()
    safe_candidates = safe_candidates.sort_values("ci_width", ascending=False)
    # 去重
    safe_candidates = safe_candidates.drop_duplicates(
        subset=["contestant_id"], keep="first"
    )

    # 3. 稳淘汰选手：评委排名差（数字大）且被淘汰
    elim_candidates = df[
        (df["judge_rank"] >= df["n_contestants"] * 0.7)  # 评委排名后30%
        & (df["placement"] == df["n_contestants"])
    ].copy()
    elim_candidates = elim_candidates.sort_values("ci_width", ascending=False)
    # 去重
    elim_candidates = elim_candidates.drop_duplicates(
        subset=["contestant_id"], keep="first"
    )

    return edge_candidates, safe_candidates, elim_candidates


def plot_sensitivity_errorbar(
    edge_df, safe_df, elim_df, data_type, output_dir, n_edge=3, n_safe=4, n_elim=4
):
    """
    绘制带误差棒的灵敏度分析图
    """
    # 选择样本
    edge_samples = edge_df.head(n_edge)
    safe_samples = safe_df.head(n_safe)
    elim_samples = elim_df.head(n_elim)

    # 根据数据类型确定列名
    if data_type == "pct":
        score_col = "implied_audience_score"
        lower_col = "audience_lower_95"
        upper_col = "audience_upper_95"
        season_col = "season"
    else:
        score_col = "implied_audience_score"
        lower_col = "implied_audience_lower_95"
        upper_col = "implied_audience_upper_95"
        season_col = "season_idx"

    # 合并数据并添加类别标签
    edge_samples = edge_samples.copy()
    safe_samples = safe_samples.copy()
    elim_samples = elim_samples.copy()

    edge_samples["category"] = "Edge (Low Judge, Survived)"
    safe_samples["category"] = "Safe (High Judge, Top Placement)"
    elim_samples["category"] = "Eliminated (Low Judge, Out)"

    all_samples = pd.concat(
        [edge_samples, safe_samples, elim_samples], ignore_index=True
    )

    # 打乱顺序，使各类选手混合显示
    np.random.seed(42)
    all_samples = all_samples.sample(frac=1).reset_index(drop=True)

    # 创建标签（姓名 + 真实赛季周）
    # 注意：week_idx 是 0-based，week1 对应 week_idx=0，所以加1显示
    all_samples["label"] = all_samples.apply(
        lambda row: f"{row['real_name']}\n(S{int(row['true_season'])}W{int(row['week_idx'])+1})",
        axis=1,
    )

    # 计算误差
    all_samples["yerr_lower"] = all_samples[score_col] - all_samples[lower_col]
    all_samples["yerr_upper"] = all_samples[upper_col] - all_samples[score_col]

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))

    # 颜色映射
    colors = {
        "Edge (Low Judge, Survived)": "#FF6B6B",  # 红色
        "Safe (High Judge, Top Placement)": "#4ECDC4",  # 青色
        "Eliminated (Low Judge, Out)": "#95A5A6",  # 灰色
    }

    # x轴位置
    x_positions = np.arange(len(all_samples))

    # 先绘制置信区间的填充背景（连接上下界）
    upper_values = all_samples[upper_col].values
    lower_values = all_samples[lower_col].values
    ax.fill_between(
        x_positions,
        lower_values,
        upper_values,
        alpha=0.15,
        color="#FF6B6B",
        label="95% Confidence Interval",
    )

    # 绘制上下界连线（淡红色线条）
    ax.plot(
        x_positions,
        upper_values,
        color="#FF6B6B",
        alpha=0.3,
        linewidth=1.5,
        linestyle="--",
    )
    ax.plot(
        x_positions,
        lower_values,
        color="#FF6B6B",
        alpha=0.3,
        linewidth=1.5,
        linestyle="--",
    )

    # 用于收集图例的标签（避免重复）
    legend_added = {}

    # 绘制误差棒
    for i, (idx, row) in enumerate(all_samples.iterrows()):
        color = colors[row["category"]]
        category = row["category"]

        # 只在第一次出现时添加图例标签
        label = category if category not in legend_added else ""
        if label:
            legend_added[category] = True

        ax.errorbar(
            x_positions[i],
            row[score_col],
            yerr=[[row["yerr_lower"]], [row["yerr_upper"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.5,
            capsize=6,
            capthick=2.5,
            markersize=10,
            label=label,
            zorder=5,
        )

    # 设置x轴标签
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_samples["label"], rotation=45, ha="right", fontsize=10)

    # 标题和标签
    method_name = "Percentage-based" if data_type == "pct" else "Rank-based"
    ax.set_title(
        f"Sensitivity Analysis: Audience Score Confidence Intervals\n({method_name} Method)",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Contestant (Season & Week)", fontsize=12)
    ax.set_ylabel("Implied Audience Score (with 95% CI)", fontsize=12)

    ax.grid(True, alpha=0.3, axis="y")

    # 图例（去重）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)

    plt.tight_layout()

    # 保存
    filename = f"sensitivity_errorbar_{data_type}.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    print(f"Saved: {filename}")

    plt.close()

    # 输出统计信息到终端
    print(f"\n[{method_name}] Selected Contestants:")
    print("-" * 80)
    for _, row in all_samples.iterrows():
        ci_width = row[upper_col] - row[lower_col]
        print(
            f"  {row['category'][:15]:15s} | {str(row['real_name'])[:20]:20s} | "
            f"Score: {row[score_col]:6.2f} | CI: [{row[lower_col]:6.2f}, {row[upper_col]:6.2f}] | Width: {ci_width:6.2f}"
        )


def main():
    print("=" * 80)
    print("灵敏度分析：置信区间误差棒图")
    print("=" * 80)

    # 加载名称映射
    print("\n加载名称映射...")
    name_mapping, season_mapping, raw_df = load_name_mapping()
    print(f"映射数量: {len(name_mapping)}")

    # 数据路径
    pct_input = (
        OUTPUT_DIR
        / "question1_res"
        / "draw_await"
        / "percent"
        / "pct_processed_data.csv"
    )
    rank_input = OUTPUT_DIR / "question1_res" / "rank_processed_data.csv"
    output_dir = OUTPUT_DIR / "question5_res" / "stick_draw"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 处理 pct 数据
    print("\n" + "=" * 80)
    print("处理百分比方法数据 (Season 3-27)")
    print("=" * 80)

    pct_df = pd.read_csv(pct_input)
    # 将 celebrity_name（实际是 contestant_id）映射为真实姓名和赛季
    pct_df["real_name"] = pct_df["celebrity_name"].map(name_mapping)
    pct_df["true_season"] = pct_df["celebrity_name"].map(season_mapping)
    pct_df["real_name"] = pct_df["real_name"].fillna(
        pct_df["celebrity_name"].astype(str)
    )
    pct_df["true_season"] = pct_df["true_season"].fillna(pct_df["season"])
    print(f"数据量: {len(pct_df)}")

    edge_pct, safe_pct, elim_pct = find_edge_cases_pct(pct_df)
    print(f"淘汰边缘候选: {len(edge_pct)}")
    print(f"稳晋级候选: {len(safe_pct)}")
    print(f"稳淘汰候选: {len(elim_pct)}")

    plot_sensitivity_errorbar(
        edge_pct, safe_pct, elim_pct, "pct", output_dir, n_edge=6, n_safe=8, n_elim=8
    )

    # 处理 rank 数据
    print("\n" + "=" * 80)
    print("处理排名方法数据 (Season 1-2, 28-33)")
    print("=" * 80)

    rank_df = pd.read_csv(rank_input)
    # 将 contestant_id 映射为真实姓名和赛季
    rank_df["real_name"] = rank_df["contestant_id"].map(name_mapping)
    rank_df["true_season"] = rank_df["contestant_id"].map(season_mapping)
    rank_df["real_name"] = rank_df["real_name"].fillna(
        rank_df["celebrity_name"].astype(str)
    )
    rank_df["true_season"] = rank_df["true_season"].fillna(
        rank_df.get("season", rank_df["season_idx"])
    )
    print(f"数据量: {len(rank_df)}")

    edge_rank, safe_rank, elim_rank = find_edge_cases_rank(rank_df)
    print(f"淘汰边缘候选: {len(edge_rank)}")
    print(f"稳晋级候选: {len(safe_rank)}")
    print(f"稳淘汰候选: {len(elim_rank)}")

    plot_sensitivity_errorbar(
        edge_rank,
        safe_rank,
        elim_rank,
        "rank",
        output_dir,
        n_edge=6,
        n_safe=8,
        n_elim=8,
    )

    print("\n" + "=" * 80)
    print(f"所有图像已保存至: {output_dir}")
    print("完成！")


if __name__ == "__main__":
    main()
