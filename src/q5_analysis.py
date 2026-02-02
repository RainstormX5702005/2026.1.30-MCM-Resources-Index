import pandas as pd
import numpy as np
import re
from configs.config import OUTPUT_DIR, DATA_DIR


def calculate_estimated_votes(row, viewers_df, weeks_count):
    """
    计算估算的观众投票数
    公式: (赛季观看人数 * 20) / 周数 * 每周投票强度
    """
    season = row["season"]
    season_viewers = viewers_df[viewers_df["Season"] == season]

    if len(season_viewers) == 0:
        return None, None, None

    avg_viewers = (
        season_viewers["AvgViewers"].values[0] * 1_000_000
    )  # 转换为实际人数（单位是百万）
    base_votes_per_week = (avg_viewers * 20) / weeks_count

    # 收集每周的投票数据
    weekly_votes = []
    for week in range(1, weeks_count + 1):
        col_name = f"week{week}_audience_votes"
        if col_name in row.index and pd.notna(row[col_name]):
            weekly_vote_pct = row[col_name]  # 这是归一化后的百分比
            estimated_votes = base_votes_per_week * weekly_vote_pct
            weekly_votes.append(estimated_votes)

    if len(weekly_votes) == 0:
        return None, None, None

    total_estimated = sum(weekly_votes)
    min_estimated = total_estimated * 0.8  # 区间下限
    max_estimated = total_estimated * 1.2  # 区间上限

    return total_estimated, min_estimated, max_estimated


def select_low_score_finalists_pct(df, top_n=5):
    """
    基于百分比方法选择低分进决赛的特例（3-27季）
    """
    # 筛选3-27季，进入决赛，且有低分晋级记录的选手
    mask = (
        (df["season"] >= 3)
        & (df["season"] <= 27)
        & (df["is_final_reached"] == True)
        & (df["low_score_advanced_count"] > 0)
    )
    candidates = df[mask].copy()

    # 按low_score_advanced_count降序排列，选取前top_n个
    candidates = candidates.sort_values(
        "low_score_advanced_count", ascending=False
    ).head(top_n)

    return candidates


def select_low_score_finalists_rank(df, top_n=5):
    """
    基于排名方法选择低分进决赛的特例（其他季）
    """
    # 筛选非3-27季，进入决赛，且有低分晋级记录的选手
    mask = (
        ((df["season"] < 3) | (df["season"] > 27))
        & (df["is_final_reached"] == True)
        & (df["low_score_advanced_count"] > 0)
    )
    candidates = df[mask].copy()

    # 按low_score_advanced_count降序排列，选取前top_n个
    candidates = candidates.sort_values(
        "low_score_advanced_count", ascending=False
    ).head(top_n)

    return candidates


def calculate_normalized_votes(featured_df, season, week):
    """
    计算某个赛季某一周所有选手的归一化投票强度
    返回一个字典：{celebrity_name: normalized_intensity}
    """
    vote_col = f"week{week}_audience_votes"

    # 获取该赛季所有选手在该周的数据
    season_data = featured_df[featured_df["season"] == season].copy()

    # 过滤出有效数据（不是NaN且不是-1）
    valid_data = season_data[
        (season_data[vote_col].notna()) & (season_data[vote_col] >= 0)
    ]

    if len(valid_data) == 0:
        return {}

    # 计算总投票强度
    total_intensity = valid_data[vote_col].sum()

    if total_intensity == 0:
        return {}

    # 归一化
    normalized_dict = {}
    for _, row in valid_data.iterrows():
        normalized_dict[row["celebrity_name"]] = row[vote_col] / total_intensity

    return normalized_dict


def analyze_candidate(
    row, featured_df, viewers_df, season_week_normalized, vote_factor=20
):
    """
    分析单个候选人的详细信息
    season_week_normalized: 预先计算好的归一化数据 {(season, week): {celebrity_name: normalized_value}}
    vote_factor: 每个观众投票数（1=下界，20=上界）
    """
    result = {
        "celebrity_name": row["celebrity_name"],
        "season": row["season"],
        "is_final_reached": row["is_final_reached"],
        "low_score_advanced_count": row["low_score_advanced_count"],
    }

    season = row["season"]
    celebrity_name = row["celebrity_name"]
    season_viewers = viewers_df[viewers_df["Season"] == season]

    if len(season_viewers) == 0:
        result["weeks_participated"] = 0
        result["weekly_details"] = []
        return result

    avg_viewers = (
        season_viewers["AvgViewers"].values[0] * 1_000_000
    )  # 转换为实际人数（单位是百万）

    # 统计参与的周数
    weeks_count = 0
    weekly_details = []

    for week in range(1, 12):  # 最多11周
        vote_col = f"week{week}_audience_votes"
        pct_col = f"week{week}_audience_pct"
        rank_col = f"week{week}_audience_rank"

        if vote_col in row.index and pd.notna(row[vote_col]) and row[vote_col] >= 0:
            weeks_count += 1

            # 使用归一化后的投票强度
            normalized_intensity = season_week_normalized.get((season, week), {}).get(
                celebrity_name, 0
            )

            # 计算估算票数：vote_factor * 平均观众数 * 归一化后的投票强度
            estimated_votes_week = vote_factor * avg_viewers * normalized_intensity
            min_votes_week = estimated_votes_week * 0.8
            max_votes_week = estimated_votes_week * 1.2

            week_info = {
                "week": week,
                "audience_votes": row[vote_col],
                "normalized_intensity": normalized_intensity,
                "audience_pct": (
                    row[pct_col]
                    if pct_col in row.index
                    and pd.notna(row[pct_col])
                    and row[pct_col] >= 0
                    else None
                ),
                "audience_rank": (
                    row[rank_col]
                    if rank_col in row.index
                    and pd.notna(row[rank_col])
                    and row[rank_col] >= 0
                    else None
                ),
                "estimated_votes": estimated_votes_week,
                "estimated_votes_min": min_votes_week,
                "estimated_votes_max": max_votes_week,
            }
            weekly_details.append(week_info)

    result["weeks_participated"] = weeks_count
    result["weekly_details"] = weekly_details

    return result


def get_week_all_competitors(
    featured_df, viewers_df, season_week_normalized, season, week, use_pct=True
):
    """
    获取某个赛季某一周的所有选手及其投票数据
    """
    season_data = featured_df[featured_df["season"] == season].copy()
    vote_col = f"week{week}_audience_votes"
    score_col = f"week{week}_score_sum"
    pct_col = f"week{week}_audience_pct"
    rank_col = f"week{week}_audience_rank"
    judge_pct_col = f"week{week}_judge_pct"
    judge_rank_col = f"week{week}_judge_rank"
    combined_pct_col = f"week{week}_combined_pct"
    combined_rank_col = f"week{week}_combined_rank"

    # 过滤有效数据
    valid_data = season_data[
        (season_data[vote_col].notna()) & (season_data[vote_col] >= 0)
    ].copy()

    if len(valid_data) == 0:
        return None

    # 获取观众数
    season_viewers = viewers_df[viewers_df["Season"] == season]
    if len(season_viewers) == 0:
        return None
    avg_viewers = season_viewers["AvgViewers"].values[0] * 1_000_000

    results = []
    for _, row in valid_data.iterrows():
        celebrity_name = row["celebrity_name"]
        normalized_intensity = season_week_normalized.get((season, week), {}).get(
            celebrity_name, 0
        )

        # 计算票数范围（1票~20票）
        votes_min = 1 * avg_viewers * normalized_intensity
        votes_max = 20 * avg_viewers * normalized_intensity

        competitor_info = {
            "celebrity_name": celebrity_name,
            "score": row.get(score_col, None) if score_col in row.index else None,
            "audience_votes": row[vote_col],
            "normalized_intensity": normalized_intensity,
            "estimated_votes_min": votes_min,
            "estimated_votes_max": votes_max,
            "is_final_reached": row.get("is_final_reached", False),
        }

        # 添加评委信息
        if judge_pct_col in row.index:
            competitor_info["judge_pct"] = (
                row[judge_pct_col]
                if pd.notna(row[judge_pct_col]) and row[judge_pct_col] >= 0
                else None
            )
        if judge_rank_col in row.index:
            competitor_info["judge_rank"] = (
                row[judge_rank_col]
                if pd.notna(row[judge_rank_col]) and row[judge_rank_col] >= 0
                else None
            )

        # 添加综合排名信息
        if combined_pct_col in row.index:
            competitor_info["combined_pct"] = (
                row[combined_pct_col]
                if pd.notna(row[combined_pct_col]) and row[combined_pct_col] >= 0
                else None
            )
        if combined_rank_col in row.index:
            competitor_info["combined_rank"] = (
                row[combined_rank_col]
                if pd.notna(row[combined_rank_col]) and row[combined_rank_col] >= 0
                else None
            )

        if use_pct and pct_col in row.index:
            competitor_info["audience_pct"] = (
                row[pct_col] if pd.notna(row[pct_col]) and row[pct_col] >= 0 else None
            )
        if not use_pct and rank_col in row.index:
            competitor_info["audience_rank"] = (
                row[rank_col]
                if pd.notna(row[rank_col]) and row[rank_col] >= 0
                else None
            )

        results.append(competitor_info)

    return results


def evaluate_week_quality(competitors, target_celebrity):
    """
    评估某一周数据的质量
    评分标准：低分但高票的选手（特例）vs 高分低票的选手的对比度
    """
    if not competitors or len(competitors) < 2:
        return 0

    # 找到目标选手
    target = None
    for c in competitors:
        if c["celebrity_name"] == target_celebrity:
            target = c
            break

    if not target or target["score"] is None:
        return 0

    # 计算质量分数
    quality_score = 0

    # 1. 分数排名（越低越好，说明是低分）
    valid_scores = [c for c in competitors if c["score"] is not None]
    valid_scores_sorted = sorted(valid_scores, key=lambda x: x["score"], reverse=True)
    target_score_rank = next(
        (
            i
            for i, c in enumerate(valid_scores_sorted)
            if c["celebrity_name"] == target_celebrity
        ),
        -1,
    )

    if target_score_rank >= len(valid_scores_sorted) / 2:  # 排名在后50%
        quality_score += (target_score_rank / len(valid_scores_sorted)) * 50  # 最多50分

    # 2. 票数排名（越高越好，说明票数高）
    votes_sorted = sorted(
        competitors, key=lambda x: x["estimated_votes_max"], reverse=True
    )
    target_votes_rank = next(
        (
            i
            for i, c in enumerate(votes_sorted)
            if c["celebrity_name"] == target_celebrity
        ),
        -1,
    )

    if target_votes_rank < len(votes_sorted) / 2:  # 票数排名在前50%
        quality_score += (1 - target_votes_rank / len(votes_sorted)) * 50  # 最多50分

    # 3. 对比度：低分高票 vs 高分低票的差异
    target_score = target["score"]
    target_votes = target["estimated_votes_max"]

    contrast = 0
    for c in competitors:
        if c["celebrity_name"] != target_celebrity and c["score"] is not None:
            if c["score"] > target_score and c["estimated_votes_max"] < target_votes:
                # 这个人分数比目标高，但票数比目标低
                contrast += 1

    quality_score += (contrast / max(len(competitors) - 1, 1)) * 50  # 最多50分

    return quality_score


def find_best_examples(
    featured_df, viewers_df, season_week_normalized, candidates, use_pct=True, top_n=2
):
    """
    从候选人中找出最能说明问题的案例
    """
    all_examples = []

    for _, row in candidates.iterrows():
        celebrity_name = row["celebrity_name"]
        season = row["season"]

        # 分析这个选手的每一周
        for week in range(1, 12):
            vote_col = f"week{week}_audience_votes"
            if vote_col in row.index and pd.notna(row[vote_col]) and row[vote_col] >= 0:
                # 获取这一周的所有竞争对手
                competitors = get_week_all_competitors(
                    featured_df,
                    viewers_df,
                    season_week_normalized,
                    season,
                    week,
                    use_pct,
                )

                if competitors:
                    # 评估这一周的质量
                    quality = evaluate_week_quality(competitors, celebrity_name)

                    all_examples.append(
                        {
                            "celebrity_name": celebrity_name,
                            "season": season,
                            "week": week,
                            "quality_score": quality,
                            "competitors": competitors,
                        }
                    )

    # 按质量分数排序，选出top_n
    all_examples.sort(key=lambda x: x["quality_score"], reverse=True)
    return all_examples[:top_n]


def main():
    # 读取数据
    featured_df = pd.read_csv(
        OUTPUT_DIR / "q4_featured_data.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    viewers_df = pd.read_csv(
        OUTPUT_DIR / "question5_res" / "q5_viewers.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )

    print("=" * 80)
    print("选择低分进决赛的特例分析")
    print("=" * 80)

    # 预先计算所有赛季所有周的归一化投票强度
    print("\n正在计算归一化投票强度...")
    season_week_normalized = {}
    unique_seasons = featured_df["season"].unique()
    for season in unique_seasons:
        for week in range(1, 12):
            normalized = calculate_normalized_votes(featured_df, season, week)
            if normalized:
                season_week_normalized[(season, week)] = normalized
    print(f"已完成 {len(season_week_normalized)} 个赛季-周组合的归一化计算")

    # 方法1：基于百分比（3-27季）
    print("\n【方法1：基于百分比指标（3-27季）】")
    pct_candidates = select_low_score_finalists_pct(featured_df, top_n=5)
    print(f"\n找到 {len(pct_candidates)} 个候选人：")

    pct_results = []
    for idx, row in pct_candidates.iterrows():
        # 计算两种情况：vote_factor=1（下界）和vote_factor=20（上界）
        result_min = analyze_candidate(
            row, featured_df, viewers_df, season_week_normalized, vote_factor=1
        )
        result_max = analyze_candidate(
            row, featured_df, viewers_df, season_week_normalized, vote_factor=20
        )
        pct_results.append((result_min, result_max))

        print(
            f"\n{len(pct_results)}. {result_max['celebrity_name']} (Season {result_max['season']})"
        )
        print(f"   - 进入决赛: {result_max['is_final_reached']}")
        print(f"   - 低分晋级次数: {result_max['low_score_advanced_count']}")
        print(f"   - 参与周数: {result_max['weeks_participated']}")

        # 计算总票数
        total_votes_min = sum(
            w.get("estimated_votes", 0) for w in result_min["weekly_details"]
        )
        total_votes_max = sum(
            w.get("estimated_votes", 0) for w in result_max["weekly_details"]
        )
        print(
            f"   - 总估算票数范围: [{total_votes_min:,.0f}, {total_votes_max:,.0f}]（每人1票~20票）"
        )

        print(f"   - 每周独立投票数据（下界：每人1票 ~ 上界：每人20票）:")
        for i, week_info_min in enumerate(result_min["weekly_details"]):
            week_info_max = result_max["weekly_details"][i]
            if week_info_max["audience_pct"] is not None:
                print(
                    f"      Week {week_info_max['week']:2d}: 强度={week_info_max['normalized_intensity']:.4f} ({week_info_max['audience_pct']*100:6.2f}%) | 票数: [{week_info_min['estimated_votes']:>12,.0f}, {week_info_max['estimated_votes']:>12,.0f}]"
                )

    # 方法2：基于排名（其他季）
    print("\n" + "=" * 80)
    print("【方法2：基于排名指标（非3-27季）】")
    rank_candidates = select_low_score_finalists_rank(featured_df, top_n=5)
    print(f"\n找到 {len(rank_candidates)} 个候选人：")

    rank_results = []
    for idx, row in rank_candidates.iterrows():
        # 计算两种情况：vote_factor=1（下界）和vote_factor=20（上界）
        result_min = analyze_candidate(
            row, featured_df, viewers_df, season_week_normalized, vote_factor=1
        )
        result_max = analyze_candidate(
            row, featured_df, viewers_df, season_week_normalized, vote_factor=20
        )
        rank_results.append((result_min, result_max))

        print(
            f"\n{len(rank_results)}. {result_max['celebrity_name']} (Season {result_max['season']})"
        )
        print(f"   - 进入决赛: {result_max['is_final_reached']}")
        print(f"   - 低分晋级次数: {result_max['low_score_advanced_count']}")
        print(f"   - 参与周数: {result_max['weeks_participated']}")

        # 计算总票数
        total_votes_min = sum(
            w.get("estimated_votes", 0) for w in result_min["weekly_details"]
        )
        total_votes_max = sum(
            w.get("estimated_votes", 0) for w in result_max["weekly_details"]
        )
        print(
            f"   - 总估算票数范围: [{total_votes_min:,.0f}, {total_votes_max:,.0f}]（每人1票~20票）"
        )

        print(f"   - 每周独立投票数据（下界：每人1票 ~ 上界：每人20票）:")
        for i, week_info_min in enumerate(result_min["weekly_details"]):
            week_info_max = result_max["weekly_details"][i]
            if week_info_max["audience_rank"] is not None:
                print(
                    f"      Week {week_info_max['week']:2d}: 强度={week_info_max['normalized_intensity']:.4f} (排名={week_info_max['audience_rank']:.0f}) | 票数: [{week_info_min['estimated_votes']:>12,.0f}, {week_info_max['estimated_votes']:>12,.0f}]"
                )

    # 保存结果到CSV
    output_dir = OUTPUT_DIR / "question5_res"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存百分比方法的结果（每周详情）
    pct_weekly_data = []
    for result_min, result_max in pct_results:
        for i, week_info_min in enumerate(result_min["weekly_details"]):
            week_info_max = result_max["weekly_details"][i]
            # 只保存有效数据（投票强度不为负）
            if (
                week_info_max.get("audience_pct") is not None
                and week_info_max.get("audience_pct") >= 0
            ):
                pct_weekly_data.append(
                    {
                        "celebrity_name": result_max["celebrity_name"],
                        "season": result_max["season"],
                        "low_score_advanced_count": result_max[
                            "low_score_advanced_count"
                        ],
                        "week": week_info_max["week"],
                        "audience_pct": week_info_max["audience_pct"],
                        "estimated_votes": week_info_max["estimated_votes"],
                        "estimated_votes_min": week_info_min["estimated_votes"],
                        "estimated_votes_max": week_info_max["estimated_votes"],
                    }
                )

    pct_df = pd.DataFrame(pct_weekly_data)
    pct_df.to_csv(
        output_dir / "low_score_finalists_pct_weekly.csv", index=False, encoding="utf-8"
    )

    # 找出最佳案例（百分比方法）
    print("\n" + "=" * 80)
    print("【寻找最佳示例案例 - 百分比方法】")
    print("=" * 80)
    best_pct_examples = find_best_examples(
        featured_df,
        viewers_df,
        season_week_normalized,
        pct_candidates,
        use_pct=True,
        top_n=2,
    )

    for i, example in enumerate(best_pct_examples, 1):
        print(f"\n最佳案例 #{i}:")
        print(f"  选手: {example['celebrity_name']}")
        print(f"  赛季: {example['season']}, 第 {example['week']} 周")
        print(f"  质量评分: {example['quality_score']:.2f}/150")
        print(f"\n  该周所有竞争对手:")
        print(
            f"  {'选手名':<25} {'分数':<8} {'评委%':<10} {'粉丝%':<10} {'综合%':<10} {'综合排名':<10} {'票数范围':<35} {'决赛'}"
        )
        print(f"  {'-'*115}")

        competitors_sorted = sorted(
            example["competitors"],
            key=lambda x: x["score"] if x["score"] else 0,
            reverse=True,
        )
        for comp in competitors_sorted:
            name = comp["celebrity_name"]
            score = f"{comp['score']:.2f}" if comp["score"] is not None else "N/A"
            judge_pct = (
                f"{comp.get('judge_pct', 0)*100:5.2f}%"
                if comp.get("judge_pct") is not None
                else "N/A"
            )
            audience_pct = (
                f"{comp.get('audience_pct', 0)*100:5.2f}%"
                if comp.get("audience_pct") is not None
                else "N/A"
            )
            combined_pct = (
                f"{comp.get('combined_pct', 0)*100:5.2f}%"
                if comp.get("combined_pct") is not None
                else "N/A"
            )
            combined_rank = (
                f"#{comp.get('combined_rank', 0):.0f}"
                if comp.get("combined_rank") is not None
                else "N/A"
            )
            votes_range = f"[{comp['estimated_votes_min']:>12,.0f},{comp['estimated_votes_max']:>12,.0f}]"
            is_final = "是" if comp["is_final_reached"] else "否"
            marker = " ★" if comp["celebrity_name"] == example["celebrity_name"] else ""
            print(
                f"  {name:<25} {score:<8} {judge_pct:<10} {audience_pct:<10} {combined_pct:<10} {combined_rank:<10} {votes_range:<35} {is_final}{marker}"
            )

    # 保存最佳案例详情
    best_pct_details = []
    for example in best_pct_examples:
        for comp in example["competitors"]:
            best_pct_details.append(
                {
                    "case_celebrity": example["celebrity_name"],
                    "season": example["season"],
                    "week": example["week"],
                    "quality_score": example["quality_score"],
                    "competitor_name": comp["celebrity_name"],
                    "score": comp["score"],
                    "judge_pct": comp.get("judge_pct"),
                    "audience_pct": comp.get("audience_pct"),
                    "combined_pct": comp.get("combined_pct"),
                    "judge_rank": comp.get("judge_rank"),
                    "combined_rank": comp.get("combined_rank"),
                    "normalized_intensity": comp["normalized_intensity"],
                    "estimated_votes_min": comp["estimated_votes_min"],
                    "estimated_votes_max": comp["estimated_votes_max"],
                    "is_final_reached": comp["is_final_reached"],
                }
            )

    pd.DataFrame(best_pct_details).to_csv(
        output_dir / "best_examples_pct.csv", index=False, encoding="utf-8"
    )

    # 保存排名方法的结果（每周详情）
    rank_weekly_data = []
    for result_min, result_max in rank_results:
        for i, week_info_min in enumerate(result_min["weekly_details"]):
            week_info_max = result_max["weekly_details"][i]
            # 只保存有效数据（排名不为负）
            if (
                week_info_max.get("audience_rank") is not None
                and week_info_max.get("audience_rank") >= 0
            ):
                rank_weekly_data.append(
                    {
                        "celebrity_name": result_max["celebrity_name"],
                        "season": result_max["season"],
                        "low_score_advanced_count": result_max[
                            "low_score_advanced_count"
                        ],
                        "week": week_info_max["week"],
                        "audience_rank": week_info_max["audience_rank"],
                        "audience_votes": week_info_max["audience_votes"],
                        "estimated_votes": week_info_max["estimated_votes"],
                        "estimated_votes_min": week_info_min["estimated_votes"],
                        "estimated_votes_max": week_info_max["estimated_votes"],
                    }
                )

    rank_df = pd.DataFrame(rank_weekly_data)
    rank_df.to_csv(
        output_dir / "low_score_finalists_rank_weekly.csv",
        index=False,
        encoding="utf-8",
    )

    # 找出最佳案例（排名方法）
    print("\n" + "=" * 80)
    print("【寻找最佳示例案例 - 排名方法】")
    print("=" * 80)
    best_rank_examples = find_best_examples(
        featured_df,
        viewers_df,
        season_week_normalized,
        rank_candidates,
        use_pct=False,
        top_n=2,
    )

    for i, example in enumerate(best_rank_examples, 1):
        print(f"\n最佳案例 #{i}:")
        print(f"  选手: {example['celebrity_name']}")
        print(f"  赛季: {example['season']}, 第 {example['week']} 周")
        print(f"  质量评分: {example['quality_score']:.2f}/150")
        print(f"\n  该周所有竞争对手:")
        print(
            f"  {'选手名':<25} {'分数':<8} {'评委排名':<10} {'粉丝排名':<10} {'综合排名':<10} {'综合%':<10} {'票数范围':<35} {'决赛'}"
        )
        print(f"  {'-'*115}")

        competitors_sorted = sorted(
            example["competitors"],
            key=lambda x: x["score"] if x["score"] else 0,
            reverse=True,
        )
        for comp in competitors_sorted:
            name = comp["celebrity_name"]
            score = f"{comp['score']:.2f}" if comp["score"] is not None else "N/A"
            judge_rank = (
                f"#{comp.get('judge_rank', 0):.0f}"
                if comp.get("judge_rank") is not None
                else "N/A"
            )
            audience_rank = (
                f"#{comp.get('audience_rank', 0):.0f}"
                if comp.get("audience_rank") is not None
                else "N/A"
            )
            combined_rank = (
                f"#{comp.get('combined_rank', 0):.0f}"
                if comp.get("combined_rank") is not None
                else "N/A"
            )
            combined_pct = (
                f"{comp.get('combined_pct', 0)*100:5.2f}%"
                if comp.get("combined_pct") is not None
                else "N/A"
            )
            votes_range = f"[{comp['estimated_votes_min']:>12,.0f},{comp['estimated_votes_max']:>12,.0f}]"
            is_final = "是" if comp["is_final_reached"] else "否"
            marker = " ★" if comp["celebrity_name"] == example["celebrity_name"] else ""
            print(
                f"  {name:<25} {score:<8} {judge_rank:<10} {audience_rank:<10} {combined_rank:<10} {combined_pct:<10} {votes_range:<35} {is_final}{marker}"
            )

    # 保存最佳案例详情
    best_rank_details = []
    for example in best_rank_examples:
        for comp in example["competitors"]:
            best_rank_details.append(
                {
                    "case_celebrity": example["celebrity_name"],
                    "season": example["season"],
                    "week": example["week"],
                    "quality_score": example["quality_score"],
                    "competitor_name": comp["celebrity_name"],
                    "score": comp["score"],
                    "judge_rank": comp.get("judge_rank"),
                    "audience_rank": comp.get("audience_rank"),
                    "combined_rank": comp.get("combined_rank"),
                    "judge_pct": comp.get("judge_pct"),
                    "combined_pct": comp.get("combined_pct"),
                    "normalized_intensity": comp["normalized_intensity"],
                    "estimated_votes_min": comp["estimated_votes_min"],
                    "estimated_votes_max": comp["estimated_votes_max"],
                    "is_final_reached": comp["is_final_reached"],
                }
            )

    pd.DataFrame(best_rank_details).to_csv(
        output_dir / "best_examples_rank.csv", index=False, encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("分析完成！结果已保存到:")
    print(f"  - {output_dir / 'low_score_finalists_pct_weekly.csv'}")
    print(f"  - {output_dir / 'low_score_finalists_rank_weekly.csv'}")
    print(f"  - {output_dir / 'best_examples_pct.csv'}  (最佳案例详情)")
    print(f"  - {output_dir / 'best_examples_rank.csv'} (最佳案例详情)")
    print("=" * 80)


if __name__ == "__main__":
    main()
