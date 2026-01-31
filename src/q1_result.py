import pandas as pd
import numpy as np

import re

from configs.config import DATA_DIR, OUTPUT_DIR


def transform_pct(org_df: pd.DataFrame) -> pd.DataFrame:
    """基于百分比的粉丝投票效果数据集转换，追踪每个人在一个 season 中每周的粉丝投票情况

    Args:
        org_df (pd.DataFrame): 包含百分比数据的DataFrame

    Returns:
        pd.DataFrame: 转换后的DataFrame，每个选手每周的 fan_votes 作为独立特征
    """
    df = org_df.copy()

    df["season"] = df["season_idx"] + 3
    df["week_num"] = df["week_idx"] + 1

    pivot_df = df.pivot_table(
        index=[
            "contestant_id",
            "season_idx",
            "season",
            "celebrity_name",
            "weeks_participated",
            "placement",
        ],
        columns="week_num",
        values="fan_votes",
        aggfunc="first",
    ).reset_index()

    pivot_df.columns.name = None
    new_columns = {}
    for col in pivot_df.columns:
        if isinstance(col, int):
            new_columns[col] = f"week{col}_fan_votes"
    pivot_df.rename(columns=new_columns, inplace=True)
    pivot_df = pivot_df.fillna(0).drop(["season_idx"], axis=1)

    return pivot_df


def transform_rank(org_df: pd.DataFrame) -> pd.DataFrame:
    """基于排名的粉丝投票效果数据集转换，追踪每个人在一个 season 中每周的粉丝投票情况

    Args:
        org_df (pd.DataFrame): 包含排名数据的DataFrame

    Returns:
        pd.DataFrame: 转换后的DataFrame，每个选手每周的 fan_votes 作为独立特征
    """
    df = org_df.copy()

    df["week_num"] = df["week_idx"] + 1

    pivot_df = df.pivot_table(
        index=[
            "contestant_id",
            "season_idx",
            "season",
            "celebrity_name",
            "weeks_participated",
            "placement",
        ],
        columns="week_num",
        values="fan_votes",
        aggfunc="first",
    ).reset_index()

    pivot_df.columns.name = None
    new_columns = {}
    for col in pivot_df.columns:
        if isinstance(col, int):
            new_columns[col] = f"week{col}_fan_votes"
    pivot_df.rename(columns=new_columns, inplace=True)
    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.drop(["season_idx"], axis=1)

    return pivot_df


def main():
    """基于百分比和排名计算相应的参数"""
    rank_df = pd.read_csv(
        DATA_DIR / "modeling_results_detailed_rank.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    pct_df = pd.read_csv(
        DATA_DIR / "modeling_results_detailed_pc.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )

    rank_df = transform_rank(rank_df)
    pct_df = transform_pct(pct_df)

    rank_path = OUTPUT_DIR / "question1_res" / "transformed_vote_based_on_rank.csv"
    pct_path = OUTPUT_DIR / "question1_res" / "transformed_vote_based_on_pct.csv"
    rank_df.to_csv(rank_path, index=False, encoding="utf-8")
    pct_df.to_csv(pct_path, index=False, encoding="utf-8")

    print(f"  基于排名的数据集保存至: {rank_path}")
    print(f"  基于百分比的数据集保存至: {pct_path}")


if __name__ == "__main__":
    main()
