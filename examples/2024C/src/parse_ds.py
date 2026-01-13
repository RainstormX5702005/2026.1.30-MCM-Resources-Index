from pathlib import Path

import pandas as pd
import numpy as np

from configs.config import data_dir


def main() -> None:

    matches_path = data_dir / f"raw" / "Wimbledon_featured_matches.csv"

    try:
        matches_df = pd.read_csv(matches_path, header=0, sep=",", encoding="utf-8")
        # 后续所有的解析操作在此进行
        # Step 1. TODO: 数据清洗与异常值处理
        # 对缺失的 speed_mph 进行线性插值，alpha = 0.5
        matches_df["speed_mph"] = pd.to_numeric(
            matches_df["speed_mph"], errors="coerce"
        )
        matches_df["speed_mph"] = matches_df["speed_mph"].interpolate(
            method="linear", limit=1, limit_area="inside"
        )

        # 把离散变量转化为数值，方便处理 NAN
        serve_width_dict = {"B": 1, "BC": 2, "BW": 3, "C": 4, "CW": 5}
        serve_depth_dict = {"CTL": 1, "NCTL": 2}
        return_depth = {"D": 1, "ND": 2}
        matches_df["serve_width"] = matches_df["serve_width"].map(serve_width_dict)
        matches_df["serve_depth"] = matches_df["serve_depth"].map(serve_depth_dict)
        matches_df["return_depth"] = matches_df["return_depth"].map(return_depth)

        matches_df = matches_df.fillna(0)

        # Step 2. TODO: 数据多维度分析与统计

        # highlight: Step 3. 数据按照 match_id 分组，并独立保存到不同的 CSV 文件中
        grouped = matches_df.groupby(by="match_id", sort=False)
        for match_id, group in grouped:
            output_path = data_dir / f"parsed" / f"match_{match_id}.csv"
            group.to_csv(output_path, encoding="utf-8")

    except FileNotFoundError:
        print(f"File not found: {matches_path} \n")


if __name__ == "__main__":
    main()
