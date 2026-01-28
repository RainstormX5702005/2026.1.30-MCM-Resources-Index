import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from configs.config import DATA_DIR, OUTPUT_DIR

import warnings

warnings.filterwarnings("ignore")


def time_series_cross_validation(ts, order, n_splits=10):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    for train_index, test_index in tscv.split(ts):
        train, test = ts.iloc[train_index], ts.iloc[test_index]

        try:
            model = ARIMA(train, order=order)
            model_fit = model.fit()
            predictions = model_fit.forecast(steps=len(test))
            mse = np.mean((predictions - test.values) ** 2)
            mse_scores.append(mse)
        except Exception as e:
            print(f"Error in fold: {e}")
            mse_scores.append(np.nan)

    return mse_scores


def main():
    file_path = OUTPUT_DIR / "PROCESSED_GOLD.csv"
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    ts = df["USD (PM)"]

    # 建立循环：从2016-09-20到2021-09-10，使用前面10天计算预测值
    start_date = pd.Timestamp('2016-09-20')
    end_date = pd.Timestamp('2021-09-10')
    predictions = []

    for current_date in pd.date_range(start=start_date, end=end_date, freq='D'):
        # 获取前面10天的窗口
        window = df.loc[df.index < current_date].tail(10)
        if len(window) == 10:
            # 用户在这里添加加权计算逻辑
            # 例如：predicted_value = window['USD (PM)'].mean()  # 示例
            predicted_value = None  # 替换为你的计算
            predictions.append((current_date, predicted_value))
        else:
            # 不足10天，跳过或处理
            pass

    # 检查平稳性
    result = adfuller(ts)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("数据不平稳，需要差分")
        ts_diff = ts.diff().dropna()
        d = 1
    else:
        ts_diff = ts
        d = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(ts_diff, ax=ax1)
    plot_pacf(ts_diff, ax=ax2)
    plt.savefig(OUTPUT_DIR / "acf_pacf.png")
    plt.show()

    p = 2
    q = 2
    order = (p, d, q)

    mse_scores = time_series_cross_validation(ts, order, n_splits=10)
    valid_mse = [m for m in mse_scores if not np.isnan(m)]
    if valid_mse:
        avg_mse = np.mean(valid_mse)
        print(f"10 折交叉验证 MSE 分数: {mse_scores}")
        print(f"平均 MSE: {avg_mse}")
    else:
        print("所有折叠都失败了")

    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())

    predictions = model_fit.forecast(steps=len(test))

    mse = np.mean((predictions - test.values) ** 2)
    print(f"MSE: {mse}")

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(test.index, predictions, label="Predictions", color="red")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "arima_predictions.png")
    plt.show()

    future_steps = 30  # 例如预测未来30天
    future_predictions = model_fit.forecast(steps=future_steps)
    future_dates = pd.date_range(
        start=ts.index[-1], periods=future_steps + 1, freq="D"
    )[1:]

    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts, label="Historical")
    plt.plot(
        future_dates, future_predictions, label="Future Predictions", color="orange"
    )
    plt.legend()
    plt.savefig(OUTPUT_DIR / "future_predictions.png")
    plt.show()


if __name__ == "__main__":
    main()
