import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from typing import Tuple
import re

from transformers import pipeline
from tqdm.auto import tqdm


def handle_data(
    filepath: str, pattern_or_substr: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """处理 MCM 2020C 数据集的函数，涵盖清洗数据 - 文本分析 - 特征挖掘过程

    Args:
        filepath (str): 数据集文件路径，可为 `csv` 与 `tsv` 格式
        pattern_or_substr (str): 用于匹配标题和评论内容的子字符串或正则表达式模式

    Example:
    ```python
    cleaned_df, suspicious_df = handle_data("path/to/tsv.tsv", r"\b(?:pacifier)\b")
    ```
    """
    # 使用 pandas 读取数据集
    df = pd.read_csv(filepath, sep="\t", header=0, encoding="utf-8")

    # 数据类型转换，确保后续转换为其它格式时不会出现数据错误
    columns = df.columns.tolist()
    num_cols = ["star_rating", "helpful_votes", "total_votes"]
    date_col = ["review_date"]
    id_cols = ["customer_id", "product_parent"]
    txt_cols = list(set(columns) - set(num_cols) - set(date_col) - set(id_cols))

    df.loc[:, num_cols] = df.loc[:, num_cols].apply(pd.to_numeric, errors="coerce")
    df.loc[:, date_col] = df.loc[:, date_col].apply(pd.to_datetime, errors="coerce")
    df.loc[:, txt_cols] = df.loc[:, txt_cols].astype(str)
    df.loc[:, id_cols] = df.loc[:, id_cols].astype("int64")

    # 对 Y/N 特征进行 1/0 编码转换
    df["vine"] = df["vine"].map({"Y": 1, "y": 1, "N": 0, "n": 0}).astype("int8")
    df["verified_purchase"] = (
        df["verified_purchase"].map({"Y": 1, "y": 1, "N": 0, "n": 0}).astype("int8")
    )

    # 去除无用列
    df = df.drop(columns=["marketplace", "product_category"], axis=1)

    # warning: 通过正则表达式去掉转义字符与 HTML 标签，如果不做的话数据转换将会失败
    html_pattern = re.compile(r"<[^>]+>")
    control_pattern = re.compile(r"[\x00-\x1F\x7F-\x9F]")
    clr_pattern = html_pattern.pattern + "|" + control_pattern.pattern
    df["review_body"] = df["review_body"].apply(
        lambda x: re.sub(clr_pattern, " ", str(x))
    )
    df["review_headline"] = df["review_headline"].apply(
        lambda x: re.sub(clr_pattern, " ", str(x))
    )

    # 进行子字符串或是正则表达式匹配删除掉明显不相关的数据行
    mask = (
        df.loc[:, ["review_body", "product_title"]]
        .apply(
            lambda col: col.astype(str).str.contains(
                pattern_or_substr, case=False, na=False, regex=True
            )
        )
        .any(axis=1)
    )

    # 取反以保留未命中行，保留可疑数据集，后续需要可以通过 ML 写回
    cleaned_df = df[mask].copy()
    suspicious_df = df[~mask].copy()

    #

    # 进一步分离出 Vine 计划评论数据，便于单独分析
    vine_mask = cleaned_df["vine"] == 1
    vine_df = cleaned_df[vine_mask].copy()

    return cleaned_df, suspicious_df, vine_df


def redrain_data(suspicious_df: pd.DataFrame) -> pd.DataFrame:
    """使用 ML 方法对可疑数据集进行重新分类，尝试将部分数据写回清洗数据集

    Args:
        suspicious_df (pd.DataFrame): 可疑数据集

    Returns:
        pd.DataFrame: 重新分类后的数据集
    """
    # question: 是否需要使用取决于训练结果，如果需要再实现此函数
    return pd.DataFrame()


def set_features(df: pd.DataFrame) -> pd.DataFrame:
    """从处理完成的数据集中进行特征的选择与构造操作"""
    featured_df = df.copy()

    # TODO1: 根据 helpful_votes 与 total_votes 构造新的特征：effective_ratio
    featured_df["effective_ratio"] = np.where(
        featured_df["total_votes"] == 0,
        0,
        featured_df["helpful_votes"] / featured_df["total_votes"],
    ).astype("float32")
    # TODO2: 根据 review_body 构造文本特征，考虑点在于：评论长度，大写字母占比，标点符号使用情况，情感分析，全局关键词频率等
    # 综合 headline 和 review_body 的长度（加权：headline 0.8, body 0.2）
    featured_df["length"] = (
        featured_df["review_headline"].apply(len) * 0.7
        + featured_df["review_body"].apply(len) * 0.3
    ).astype("int32")

    # 综合大写字母占比（加权：headline 0.7, body 0.3），名称 upcase_ratio
    def calc_upcase_ratio(row):
        headline = row["review_headline"]
        body = row["review_body"]
        total_len = len(headline) + len(body)
        if total_len == 0:
            return 0.0
        upcase_count = (
            sum(1 for c in headline if c.isupper()) * 0.7
            + sum(1 for c in body if c.isupper()) * 0.3
        )
        return upcase_count / total_len

    featured_df["upcase_ratio"] = featured_df.apply(calc_upcase_ratio, axis=1).astype(
        "float32"
    )

    # 标点强度（punctuation_intensity）：只捕捉 ! 和 ?，连续使用时额外加权
    def calc_punctuation_intensity(row):
        text = row["review_headline"] + " " + row["review_body"]
        if len(text) == 0:
            return 0.0
        punct_positions = [i for i, c in enumerate(text) if c in ["!", "?"]]
        weighted_count = 0
        for i in range(len(punct_positions)):
            if i > 0 and punct_positions[i] == punct_positions[i - 1] + 1:
                weighted_count += 5
            else:
                weighted_count += 1
        return weighted_count / len(text)

    featured_df["punctuation_intensity"] = featured_df.apply(
        calc_punctuation_intensity, axis=1
    ).astype("float32")

    # highlight 基于深度学习的 BERT 情感分析模型，输出情感标签
    clf = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0,
    )

    def calc_sentiment(txt: str):
        # pipeline returns a list of dicts; truncate text for safety
        res = clf(str(txt)[:512])
        if isinstance(res, list) and len(res) > 0:
            label = res[0].get("label", "NEGATIVE")
            return 1 if label == "POSITIVE" else 0
        return 0

    tqdm.pandas(desc="Calculating sentiment")
    featured_df["sentiment"] = (
        featured_df["review_body"].progress_apply(calc_sentiment).astype("int8")
    )

    # TODO3: 更多特征待定

    return featured_df


def main():
    data_dir = Path(__file__).parent / "data"

    pacifiers_pattern = r"\b(?:pacifiers?|silicone|rubber|latex|plastic|soft|smooth|flexible|textured)\b"
    attached = r"\b(?:baby|infant|kid|newborn|toddler|soother|diaper|seat|stroller|napkins?|nipple|teether|binky|dummy|little|sons?|daughters?|safety|\d+\s*(?:month|year)s?\s+old|bottle|crib|monitor|wipes|lotion|toy|blanket|carrier|swing|walker)\b"

    file_name = ["hair_dryer.tsv", "microwave.tsv", "pacifier.tsv"]

    dataset_path = data_dir / "pacifier.tsv"
    cleaned_path = data_dir / "xlsx" / "pacifier_cleaned.xlsx"
    suspicious_path = data_dir / "xlsx" / "pacifier_cleared.xlsx"
    vine_path = data_dir / "xlsx" / "pacifier_vine.xlsx"

    cleaned_df, suspicious_df, vine_df = handle_data(
        str(dataset_path), pacifiers_pattern + "|" + attached
    )
    cleaned_df = set_features(cleaned_df)
    # TODO? 如果数据集不够理想，可以使用 ML 方法把 suspicious_df 重新分类回去部分数据

    cleaned_df.to_excel(cleaned_path, index=False)


if __name__ == "__main__":
    main()
