import pandas as pd
import numpy as np

from configs.config import DATA_DIR, OUTPUT_DIR

import re


def handle_data(file_name: str) -> pd.DataFrame:
    try:
        file_path = DATA_DIR / file_name
        df = pd.read_csv(file_path, sep=",", header=0, encoding="utf-8")

        # 把 object 类型的缺失值补充为 "Unknown"，并改为 str 类型
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in obj_cols:
            df[col] = df[col].fillna("Unknown").astype(str)

        df["CryoSleep"] = np.where(df["CryoSleep"] == "True", 1, 0).astype(bool)
        df["HasName"] = np.where(df["Name"] != "Unknown", 1, 0).astype(bool)
        df["VIP"] = np.where(df["VIP"] == "True", 1, 0).astype(bool)

        # highlight: 发现数据集中需要通过字符串提取才可以获得信息，可以使用正则表达式进行任务
        id_catch = r"(\d+)_(\d+)"
        df[["Group", "Id"]] = df["PassengerId"].str.extract(id_catch)
        cabin_catch = r"(\w+)/(\d+)/(\w+)"
        df[["Deck", "Num", "Side"]] = df["Cabin"].str.extract(cabin_catch)

        cabin_cols = ["Deck", "Side"]
        for col in cabin_cols:
            df[col] = df[col].fillna("U").astype(str)
        df["Num"] = pd.to_numeric(df["Num"], errors="coerce").fillna(-1).astype(int)

        cols = ["HomePlanet", "Destination", "Deck", "Side"]
        for col in cols:
            df[col] = df[col].astype("string")
        df["Id"] = df["Id"].astype("Int64")
        df["Group"] = df["Group"].astype("Int64")
        df["GroupSize"] = (
            df.groupby("Group")["PassengerId"].transform("count").astype("Int64")
        )

        columns_to_drop = ["PassengerId", "Name", "Cabin", "Id"]
        df = df.drop(columns=columns_to_drop)

        return df

    except FileNotFoundError as e:
        print(
            f"Error: {e}. \n Please ensure the data file exists at the specified path."
        )
        return pd.DataFrame()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    """本方法在后续进行 KNN 填充值之后补充新的参数 TotalBill"""
    numeric_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    X["TotalBill"] = X[numeric_cols].sum(axis=1)
    return X


def main():
    df = handle_data("train.csv")
    df = add_features(df)
    output_path = OUTPUT_DIR / "processed_train.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path} successfully!")


if __name__ == "__main__":
    main()
