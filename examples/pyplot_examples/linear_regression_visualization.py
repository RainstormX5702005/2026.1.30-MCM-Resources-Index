import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate

from configs.encode import set_encoding_method, FIG_DIR

set_encoding_method()


def main(
    *, figsize: tuple = (10, 8), dpi: int = 100, rect: tuple = (0.1, 0.1, 0.8, 0.8)
):
    """线性回归拟合测试与绘图测试"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(rect)

    # 生成模拟数据并保留类别标签以便分开绘制
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=4.0, scale=1.0, size=400)
    c2 = rng.normal(loc=12.0, scale=3.0, size=200)
    c3 = rng.normal(loc=18.0, scale=0.4, size=350)
    xr = rng.uniform(low=0.0, high=20.0, size=200)  # 离散噪声

    x_all = np.concatenate([c1, c2, c3, xr])
    labels_all = np.concatenate(
        [
            np.full(c1.shape, 0, dtype=int),
            np.full(c2.shape, 1, dtype=int),
            np.full(c3.shape, 2, dtype=int),
            np.full(xr.shape, 3, dtype=int),
        ]
    )
    mask = (x_all >= 0) & (x_all <= 20)
    x_all = x_all[mask]
    labels_all = labels_all[mask]
    # 线性映射并加入噪声
    y_all = 2.5 * x_all + rng.normal(loc=0.0, scale=3.0, size=x_all.shape[0])

    X = x_all.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.2, random_state=42
    )

    lg = LinearRegression()
    lg.fit(x_train, y_train)
    cv_results = cross_validate(
        LinearRegression(),
        X,
        y_all,
        cv=5,
        scoring=["r2", "neg_mean_squared_error"],
        return_train_score=True,
        return_estimator=True,
    )

    coef = float(lg.coef_[0])
    intercept = float(lg.intercept_)
    cv_r2_mean = np.mean(cv_results["test_r2"])
    cv_r2_std = np.std(cv_results["test_r2"])
    cv_mse_mean = -np.mean(cv_results["test_neg_mean_squared_error"])
    cv_mse_std = np.std(cv_results["test_neg_mean_squared_error"])

    # 单独绘制每类散点
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "D", "v"]
    names = ["cluster1", "cluster2", "cluster3", "noise"]

    for cls in np.unique(labels_all):
        cls_mask = labels_all == cls
        ax.scatter(
            x_all[cls_mask],
            y_all[cls_mask],
            color=colors[int(cls)],
            marker=markers[int(cls)],
            s=18,
            alpha=0.85,
            label=names[int(cls)],
            edgecolors="k",
            linewidths=0.2,
        )

    # 绘制拟合线的过程
    x_line = np.linspace(0, 20, 300)
    y_line = lg.predict(x_line.reshape(-1, 1))
    # 绘制残差带，表示拟合存在不确定的位置
    resid = y_all - lg.predict(X)
    resid_std = float(np.std(resid))
    ax.fill_between(
        x_line,
        y_line - resid_std,
        y_line + resid_std,
        color="#ffdede",
        alpha=0.25,
        zorder=1,
    )

    # 底衬（加粗淡色）
    ax.plot(x_line, y_line, linewidth=6, color="black", alpha=0.12, zorder=2)
    ax.plot(
        x_line, y_line, color="#d62728", linewidth=2.6, zorder=3, solid_capstyle="round"
    )

    # 位于右上角的参数信息气泡配置
    info_txt = (
        f"coef={coef:.4f}\nIntercept={intercept:.4f}\n"
        f"CV R2={cv_r2_mean:.3f}±{cv_r2_std:.3f}\nCV MSE={cv_mse_mean:.3f}±{cv_mse_std:.3f}"
    )
    ax.text(
        0.98,
        0.98,
        info_txt,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.6", facecolor="#33363a", alpha=0.92, edgecolor="#222"
        ),
    )

    # 公式气泡配置
    x_mean = float(np.mean(x_all))  # warning: 由于 scatter 要求类型为 float，故做处理
    y_mean = float(coef * x_mean + intercept)
    ax.scatter(
        [x_mean],
        [y_mean],
        color="white",
        edgecolors="#444",
        marker="X",
        s=110,
        zorder=5,
    )
    eq_txt = f"y = {coef:.3f}x {intercept:+.3f}"
    ax.annotate(
        eq_txt,
        xy=(x_mean, y_mean),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.6", facecolor="#2b2b2b", alpha=0.95, edgecolor="#111"
        ),
        arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=0.7),
        zorder=6,
    )

    # 坐标轴与标题相关配置
    ax.set_xlim(-0.5, 20.5)
    ax.set_xlabel("自变量", fontsize=12)
    ax.set_ylabel("因变量", fontsize=12)
    ax.set_title("线性回归拟合示意图", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.99))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    plt.tight_layout()

    file_path = FIG_DIR / "linear_regression_visualization.png"
    plt.savefig(file_path, dpi=dpi)
    plt.show()


if __name__ == "__main__":
    main(dpi=200)
