import pandas as pd
import numpy as np

import re

from configs.config import DATA_DIR, OUTPUT_DIR


def main():
    df = pd.read_excel(
        DATA_DIR / "viewers.xlsx", sheet_name="Sheet1", header=0, usecols="A,E,G"
    )
    df = df.dropna().reset_index(drop=True)
    df["Viewers"] = df["Viewers"].astype("string")
    df["FinalDate"] = df["FinalDate"].astype("string")
    df["Season"] = df["Season"].astype("int64")

    def extract_float(value):
        match = re.search(r"(\d+\.\d+)", str(value))
        if match:
            return float(match.group(1))
        else:
            return np.nan  # 如果没有匹配，返回NaN

    df["Viewers"] = df["Viewers"].apply(extract_float)
    df["FinalDate"] = df["FinalDate"].apply(extract_float)
    df["AvgViewers"] = (df["Viewers"] + df["FinalDate"]) / 2

    output_dir = OUTPUT_DIR / "question5_res"
    df.to_csv(output_dir / "q5_viewers.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
