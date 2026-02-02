import pandas as pd
import numpy as np

import re

from configs.config import DATA_DIR, OUTPUT_DIR


def handle_data(df: pd.DataFrame) -> pd.DataFrame:
    """基于原有的数据集做初步特征工程处理"""
    processed_df = df.copy()

    columns_to_drop = [
        "celebrity_homecountry/region",
        "results",
        "low_score_advanced_week",
        "low_score_advanced_weeks",
    ]
    pattern = re.compile(r"week\d+_judge\d+_score")
    weekX_cols = [
        col
        for col in processed_df.columns
        if col in columns_to_drop or pattern.match(col)
    ]
    columns_to_drop.extend(weekX_cols)
    processed_df = processed_df.drop(columns=columns_to_drop)

    obj_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
    ]
    processed_df[obj_cols] = processed_df[obj_cols].astype("string")

    # highlight 把绝对参与数目转化为参与率，方便比较
    processed_df["participated_ratio"] = np.where(
        df["weeks_participated"] != 0,
        df["weeks_participated"] / df["season_total_weeks"],
        0,
    )

    # highlight 配对舞伴的出场次数，次数越多说明这个舞者越厉害
    processed_df["ballroom_partner_count"] = processed_df.groupby("ballroom_partner")[
        "ballroom_partner"
    ].transform("count")

    processed_df["is_legacy_season"] = (processed_df["season"] == 15).astype(bool)

    processed_df["season_total_contestants"] = processed_df.groupby("season")[
        "season"
    ].transform("count")
    # 相对排名：(placement - 1) / (total_contestants - 1), 数值越小越好
    processed_df["relative_placement"] = (processed_df["placement"] - 1) / (
        processed_df["season_total_contestants"] - 1
    )

    columns_to_drop = [
        "weeks_participated",
        "season_total_weeks",
        "placement",
    ]
    processed_df = processed_df.drop(columns=columns_to_drop)

    # 保留 season 列用于后续合并

    bool_cols = ["is_from_usa", "is_final_awarded", "is_final_reached"]
    processed_df[bool_cols] = processed_df[bool_cols].astype(bool)

    return processed_df


def add_feature_sex(df: pd.DataFrame, sex_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = df.merge(sex_df, on="celebrity_name", how="left")
    return merged_df


def add_audience_votes_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 rank_processed_data.csv 和 pct_processed_data.csv 中提取观众投票数据
    - rank: 1-2季, 28-34季
    - pct: 3-27季
    只提取 implied_audience_score，保持宽格式

    参数:
        df: 已处理的数据集 (包含 season 列)

    返回:
        添加了观众投票列的宽格式数据集
    """
    # 读取原始数据以建立映射关系
    org_df = pd.read_csv(
        OUTPUT_DIR / "processed_data.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )

    # 读取 rank 数据 (1-2季, 28-34季)
    rank_df = pd.read_csv(
        OUTPUT_DIR / "preprocessed" / "rank_processed_data.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    rank_df = rank_df[
        [
            "contestant_id",
            "season",
            "placement",
            "weeks_participated",
            "week_idx",
            "implied_audience_score",
        ]
    ].copy()

    # 读取 pct 数据 (3-27季)
    pct_df = pd.read_csv(
        OUTPUT_DIR / "question1_res" / "pct_processed_data.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    pct_df = pct_df[
        [
            "contestant_id",
            "season",
            "placement",
            "weeks_participated",
            "week_idx",
            "implied_audience_score",
        ]
    ].copy()

    # 合并两个数据源
    all_audience_df = pd.concat([rank_df, pct_df], ignore_index=True)

    # 通过 season, placement, weeks_participated 建立到真实姓名的映射
    mapping_cols = ["season", "placement", "weeks_participated"]
    org_mapping = org_df[["celebrity_name"] + mapping_cols].drop_duplicates()

    # 获取每个 contestant_id 的唯一映射信息
    contestant_mapping = all_audience_df[
        ["contestant_id"] + mapping_cols
    ].drop_duplicates()

    # 合并以获得真实姓名
    name_mapping = contestant_mapping.merge(org_mapping, on=mapping_cols, how="left")

    # 将真实姓名添加到 all_audience_df
    all_audience_df = all_audience_df.merge(
        name_mapping[["contestant_id", "season", "celebrity_name"]].drop_duplicates(),
        on=["contestant_id", "season"],
        how="left",
    )

    # 删除没有匹配到姓名的行
    all_audience_df = all_audience_df.dropna(subset=["celebrity_name"])

    # 选择需要的列
    all_audience_df = all_audience_df[
        ["season", "celebrity_name", "week_idx", "implied_audience_score"]
    ].copy()

    all_audience_df["week"] = all_audience_df["week_idx"] + 1
    all_audience_df = all_audience_df.drop(columns=["week_idx"])

    all_audience_df = all_audience_df.groupby(
        ["season", "celebrity_name", "week"], as_index=False
    ).agg({"implied_audience_score": "mean"})

    audience_wide = all_audience_df.pivot(
        index=["season", "celebrity_name"],
        columns="week",
        values="implied_audience_score",
    ).reset_index()

    audience_wide.columns = [
        f"week{col}_audience_votes" if isinstance(col, int) else col
        for col in audience_wide.columns
    ]

    merged_df = df.merge(audience_wide, on=["season", "celebrity_name"], how="left")

    return merged_df


def calculate_ranking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每一周计算排名和百分比特征

    逻辑：
    1. 对于每一周，只对有有效得分的选手（score > 0）进行排名计算
    2. 被淘汰的选手（score=0或NaN）在该周的排名和百分比为NaN
    3. 生成week1_judge_rank, week1_audience_rank, week1_judge_pct, week1_audience_pct等特征
    """

    result_df = df.copy()

    # 对每一周分别计算排名和百分比
    for week in range(1, 12):
        judge_col = f"week{week}_score_sum"
        audience_col = f"week{week}_audience_votes"

        # 初始化排名和百分比列为NaN
        result_df[f"week{week}_judge_rank"] = float("nan")
        result_df[f"week{week}_audience_rank"] = float("nan")
        result_df[f"week{week}_judge_pct"] = float("nan")
        result_df[f"week{week}_audience_pct"] = float("nan")

        # 只对有有效得分的选手计算排名（score > 0 且不为NaN）
        # 评委排名
        valid_judge_mask = (result_df[judge_col].notna()) & (result_df[judge_col] > 0)
        if valid_judge_mask.any():
            result_df.loc[valid_judge_mask, f"week{week}_judge_rank"] = (
                result_df.loc[valid_judge_mask]
                .groupby("season")[judge_col]
                .rank(ascending=False, method="average")
            )
            # 计算百分比：只在有有效得分的选手中计算
            for season in result_df[valid_judge_mask]["season"].unique():
                season_mask = (result_df["season"] == season) & valid_judge_mask
                season_total = result_df.loc[season_mask, judge_col].sum()
                if season_total > 0:
                    result_df.loc[season_mask, f"week{week}_judge_pct"] = (
                        result_df.loc[season_mask, judge_col] / season_total
                    )

        # 观众排名
        valid_audience_mask = (result_df[audience_col].notna()) & (
            result_df[audience_col] > 0
        )
        if valid_audience_mask.any():
            result_df.loc[valid_audience_mask, f"week{week}_audience_rank"] = (
                result_df.loc[valid_audience_mask]
                .groupby("season")[audience_col]
                .rank(ascending=False, method="average")
            )
            # 计算百分比：只在有有效票数的选手中计算
            for season in result_df[valid_audience_mask]["season"].unique():
                season_mask = (result_df["season"] == season) & valid_audience_mask
                season_total = result_df.loc[season_mask, audience_col].sum()
                if season_total > 0:
                    result_df.loc[season_mask, f"week{week}_audience_pct"] = (
                        result_df.loc[season_mask, audience_col] / season_total
                    )

        result_df[f"week{week}_combined_rank"] = (
            result_df[f"week{week}_judge_rank"] + result_df[f"week{week}_audience_rank"]
        )
        result_df[f"week{week}_combined_pct"] = (
            result_df[f"week{week}_judge_pct"] + result_df[f"week{week}_audience_pct"]
        )

    week_audience_cols = [f"week{i}_audience_votes" for i in range(1, 12)]
    result_df["total_audience_votes"] = result_df[week_audience_cols].sum(axis=1)

    return result_df


def main():
    org_df = pd.read_csv(
        OUTPUT_DIR / "processed_data.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    processed_df = handle_data(org_df)

    sex_df = pd.read_csv(
        DATA_DIR / "gender.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    sex_df = sex_df.loc[:, ["celebrity_name", "gender"]]
    processed_df = add_feature_sex(processed_df, sex_df)
    processed_df = add_audience_votes_wide(processed_df)
    audience_cols = [
        col for col in processed_df.columns if col.endswith("_audience_votes")
    ]
    has_audience_data = processed_df[audience_cols].notna().any(axis=1).sum()

    processed_df = calculate_ranking_features(processed_df)
    ranking_cols = [
        col
        for col in processed_df.columns
        if "_rank" in col or "_pct" in col or col == "total_audience_votes"
    ]
    display_cols = [
        "celebrity_name",
        "season",
        "gender",
        "week1_score_sum",
        "week1_audience_votes",
        "week1_judge_rank",
        "week1_audience_rank",
        "week1_combined_rank",
        "week1_judge_pct",
        "week1_combined_pct",
        "total_audience_votes",
    ]

    output_path = OUTPUT_DIR / "q4_featured_data.csv"
    processed_df["celebrity_homestate"] = processed_df["celebrity_homestate"].fillna(
        "N"
    )
    processed_df["gender"] = processed_df["gender"].fillna("1")
    processed_df = processed_df.fillna(-1)
    processed_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved to: {output_path}")

    return processed_df


if __name__ == "__main__":
    main()
