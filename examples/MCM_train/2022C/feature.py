import pandas as pd
import numpy as np

from configs.config import DATA_DIR, OUTPUT_DIR


def complete_year(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["Date"].tolist()
    completed_dates = []
    for date_str in dates:
        parts = date_str.split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid date format: {date_str}")

        if len(parts[0]) == 4:
            year = int(parts[2]) + 2000
            month = int(parts[0]) - 2000
            day = int(parts[1])
        else:
            month = int(parts[0])
            day = int(parts[1])
            y = int(parts[2])
            year = y + 2000
        completed_dates.append(f"{year}/{month:02d}/{day:02d}")

    df["Date"] = pd.to_datetime(completed_dates, format="%Y/%m/%d")

    return df


def handle_data(file_name: str) -> pd.DataFrame:
    try:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path)
        if "Date" in df.columns:
            df = complete_year(df)
        df["Date"] = pd.to_datetime(df["Date"])

        df = df.sort_values("Date")

        start_date = pd.Timestamp("2016-09-11")
        end_date = pd.Timestamp("2021-09-10")
        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        full_df = pd.DataFrame(index=full_date_range)
        full_df = full_df.merge(
            df.set_index("Date"), left_index=True, right_index=True, how="left"
        )

        # highlight: 标记缺失值，可以避免信息丢失
        if "USD (PM)" in full_df.columns:
            full_df["IsRest"] = full_df["USD (PM)"].isna().astype(int)
        else:
            full_df["IsRest"] = full_df["Value"].isna().astype(int)
        df = df.sort_values("Date").reset_index(drop=True)

        full_df = full_df.reset_index().rename(columns={"index": "Date"})
        full_df = full_df.ffill()

        if "USD (PM)" in df.columns:
            full_df.loc[0, "USD (PM)"] = 1324.6

        # question: 本列为何标准化？这是用于 HDA 阶段探索数据特征所用，后续训练无需该列
        if "USD (PM)" in df.columns:
            full_df["StandardValue"] = (
                full_df["USD (PM)"] - full_df["USD (PM)"].mean()
            ) / full_df["USD (PM)"].std()
        else:
            full_df["StandardValue"] = (
                full_df["Value"] - full_df["Value"].mean()
            ) / full_df["Value"].std()

        return full_df
    except FileNotFoundError as e:
        print(f"Exception Occurs: {e}.")
        return pd.DataFrame()


def set_features(df: pd.DataFrame) -> pd.DataFrame:
    t = float(0.8)

    start_date = pd.Timestamp("2016-09-20")
    end_date = pd.Timestamp("2021-09-10")

    df["PredictVal"] = df["USD (PM)"] if "USD (PM)" in df.columns else df["Value"]

    for i in range(len(df)):
        current_date = df.loc[i, "Date"]
        if current_date >= start_date and current_date <= end_date:
            # 获取前面10天的索引
            start_idx = max(0, i - 10)
            window = df.loc[start_idx : i - 1]
            if len(window) >= 10:
                # 计算权重：越近的权重越大
                weights = np.array([t**j for j in range(1, 11)])
                weights = weights / weights.sum()  # 归一化权重和为1
                values = (
                    window.iloc[-10:]["USD (PM)"].values
                    if "USD (PM)" in df.columns
                    else window.iloc[-10:]["Value"].values
                )
                predicted = np.dot(values, weights)
                df.loc[i, "PredictVal"] = float(predicted)

    return df


def main():
    gold_path = OUTPUT_DIR / "PROCESSED_GOLD.csv"
    mkpru_path = OUTPUT_DIR / "PROCESSED_MKPRU.csv"

    gold_df = handle_data("LBMA-GOLD.csv")
    gold_df = set_features(gold_df)
    gold_df.to_csv(gold_path, index=False)

    mkpru_df = handle_data("BCHAIN-MKPRU.csv")
    mkpru_df = set_features(mkpru_df)
    mkpru_df.to_csv(mkpru_path, index=False)


if __name__ == "__main__":
    main()
