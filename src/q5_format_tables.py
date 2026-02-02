"""
格式化输出表格：将现有数据转换为参考图片中的表格格式
- 百分比方法（Season 3-27）：显示百分比
- 排名方法（其他赛季）：显示排名
"""

import pandas as pd
from pathlib import Path

from configs.config import OUTPUT_DIR


def format_percentage_table(df, season, week):
    """
    格式化百分比方法的表格
    格式：Contestant | Total Judges Score | Judges Score Percent | Fan Vote* | Fan Percent* | Sum of Percents
    """
    # 筛选特定赛季和周
    df_filtered = df[(df["season"] == season) & (df["week"] == week)].copy()

    if len(df_filtered) == 0:
        print(f"No data for Season {season}, Week {week}")
        return None

    # 创建格式化的表格
    result = pd.DataFrame()
    result["Contestant"] = df_filtered["competitor_name"]
    result["Total Judges Score"] = df_filtered["score"].round(1)
    result["Judges Score Percent"] = (df_filtered["judge_pct"] * 100).round(1)
    result["Fan Vote (estimated)"] = df_filtered["estimated_votes_max"].apply(
        lambda x: f"{x/1e6:.1f}M"
    )
    result["Fan Percent"] = (df_filtered["audience_pct"] * 100).round(1)
    result["Sum of Percents"] = (df_filtered["combined_pct"] * 100).round(1)

    # 排序：按combined_pct降序
    result = result.sort_values("Sum of Percents", ascending=False).reset_index(
        drop=True
    )

    return result


def format_rank_table(df, season, week):
    """
    格式化排名方法的表格
    格式：Contestant | Total Judges Score | Judges Score Rank | Fan Vote* | Fan Rank* | Sum of ranks
    """
    # 筛选特定赛季和周
    df_filtered = df[(df["season"] == season) & (df["week"] == week)].copy()

    if len(df_filtered) == 0:
        print(f"No data for Season {season}, Week {week}")
        return None

    # 创建格式化的表格
    result = pd.DataFrame()
    result["Contestant"] = df_filtered["competitor_name"]
    result["Total Judges Score"] = df_filtered["score"].round(1)
    result["Judges Score Rank"] = df_filtered["judge_rank"].astype(int)
    result["Fan Vote (estimated)"] = df_filtered["estimated_votes_max"].apply(
        lambda x: f"{x/1e6:.1f}M"
    )
    result["Fan Rank"] = df_filtered["audience_rank"].astype(int)
    result["Sum of ranks"] = df_filtered["combined_rank"]

    # 排序：按combined_rank升序（排名越小越好）
    result = result.sort_values("Sum of ranks", ascending=True).reset_index(drop=True)

    return result


def save_formatted_table(df, filename, title, output_dir):
    """保存格式化的表格到CSV和文本文件"""
    if df is None:
        return

    # 保存CSV
    csv_file = output_dir / f"{filename}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV: {csv_file.name}")

    # 保存为格式化的文本文件（更易读）
    txt_file = output_dir / f"{filename}.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"\n{title}\n")
        f.write("=" * 100 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        f.write("* Fan vote/rank are estimated based on the model\n")

    print(f"✓ Saved TXT: {txt_file.name}")

    # 打印到控制台
    print(f"\n{title}")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\n* Fan vote/rank are estimated based on the model\n")


def main():
    print("=" * 100)
    print("格式化表格输出")
    print("=" * 100)

    # 加载数据
    input_dir = OUTPUT_DIR / "question5_res"

    print("\n加载数据...")
    pct_df = pd.read_csv(input_dir / "best_examples_pct.csv")
    rank_df = pd.read_csv(input_dir / "best_examples_rank.csv")

    print(f"✓ 百分比方法数据: {len(pct_df)} rows")
    print(f"✓ 排名方法数据: {len(rank_df)} rows")

    output_dir = OUTPUT_DIR / "question5_res" / "formatted_tables"
    output_dir.mkdir(exist_ok=True, parents=True)

    # ========== 百分比方法示例 ==========
    print("\n" + "=" * 100)
    print("百分比方法示例（Season 3-27）")
    print("=" * 100)

    # 找一些好的示例
    pct_seasons = pct_df["season"].unique()

    # 示例1：选择一个中间赛季
    example_seasons_pct = [5, 10, 15, 20, 24]  # 多个示例

    for i, season in enumerate(example_seasons_pct):
        if season in pct_seasons:
            # 找到该赛季的一个周
            season_data = pct_df[pct_df["season"] == season]
            available_weeks = sorted(season_data["week"].unique())

            # 选择中间的周
            if len(available_weeks) > 0:
                week = available_weeks[len(available_weeks) // 2]

                print(f"\n示例 {i+1}: Season {season}, Week {week}")
                formatted = format_percentage_table(pct_df, season, week)

                if formatted is not None:
                    title = f"Table: Combining Judge and Fan Votes by Percent (Season {season}, Week {week})"
                    filename = f"table_percent_s{season}_w{week}"
                    save_formatted_table(formatted, filename, title, output_dir)

    # ========== 排名方法示例 ==========
    print("\n" + "=" * 100)
    print("排名方法示例（其他赛季）")
    print("=" * 100)

    rank_seasons = rank_df["season"].unique()

    # 示例：选择几个赛季
    example_seasons_rank = [1, 2, 28, 30, 31, 32, 33, 34]

    for i, season in enumerate(example_seasons_rank):
        if season in rank_seasons:
            season_data = rank_df[rank_df["season"] == season]
            available_weeks = sorted(season_data["week"].unique())

            if len(available_weeks) > 0:
                week = available_weeks[len(available_weeks) // 2]

                print(f"\n示例 {i+1}: Season {season}, Week {week}")
                formatted = format_rank_table(rank_df, season, week)

                if formatted is not None:
                    title = f"Table: Combining Judge and Fan Votes by Rank (Season {season}, Week {week})"
                    filename = f"table_rank_s{season}_w{week}"
                    save_formatted_table(formatted, filename, title, output_dir)

    # ========== 生成完整数据集 ==========
    print("\n" + "=" * 100)
    print("生成完整格式化数据集")
    print("=" * 100)

    # 为所有数据生成格式化版本
    all_pct_formatted = []
    for (season, week), group in pct_df.groupby(["season", "week"]):
        formatted = format_percentage_table(pct_df, season, week)
        if formatted is not None:
            formatted["Season"] = season
            formatted["Week"] = week
            all_pct_formatted.append(formatted)

    if all_pct_formatted:
        all_pct_df = pd.concat(all_pct_formatted, ignore_index=True)
        # 重新排列列
        cols = ["Season", "Week"] + [
            col for col in all_pct_df.columns if col not in ["Season", "Week"]
        ]
        all_pct_df = all_pct_df[cols]

        all_pct_file = output_dir / "all_formatted_percent_tables.csv"
        all_pct_df.to_csv(all_pct_file, index=False)
        print(f"\n✓ Saved all percentage tables: {all_pct_file.name}")
        print(f"  Total records: {len(all_pct_df)}")

    all_rank_formatted = []
    for (season, week), group in rank_df.groupby(["season", "week"]):
        formatted = format_rank_table(rank_df, season, week)
        if formatted is not None:
            formatted["Season"] = season
            formatted["Week"] = week
            all_rank_formatted.append(formatted)

    if all_rank_formatted:
        all_rank_df = pd.concat(all_rank_formatted, ignore_index=True)
        cols = ["Season", "Week"] + [
            col for col in all_rank_df.columns if col not in ["Season", "Week"]
        ]
        all_rank_df = all_rank_df[cols]

        all_rank_file = output_dir / "all_formatted_rank_tables.csv"
        all_rank_df.to_csv(all_rank_file, index=False)
        print(f"\n✓ Saved all rank tables: {all_rank_file.name}")
        print(f"  Total records: {len(all_rank_df)}")

    print("\n" + "=" * 100)
    print(f"完成！所有格式化表格已保存至: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
