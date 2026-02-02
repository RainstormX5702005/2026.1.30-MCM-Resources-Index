"""
静态特征相关性聚类热力图分析
绘制静态特征之间的相关性，并提供聚类功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from configs.config import DATA_DIR, OUTPUT_DIR

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_circle_heatmap(
    corr_matrix, ax=None, cmap="coolwarm", show_upper=True, min_scale=50, max_scale=500
):
    """
    绘制圆形热力图
    """
    if ax is None:
        ax = plt.gca()

    n_rows, n_cols = corr_matrix.shape
    col_names = corr_matrix.columns
    row_names = corr_matrix.index

    # 构建坐标网格
    x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    # 展平
    x_flat = x.flatten()
    y_flat = y.flatten()
    values_flat = corr_matrix.values.flatten()

    # 创建遮罩
    mask_flat = np.zeros_like(values_flat, dtype=bool)
    if show_upper:
        # 只显示上三角 (k=1)
        for i in range(len(values_flat)):
            r, c = divmod(i, n_cols)
            if r > c:  # 下三角部分
                mask_flat[i] = True
            elif r == c:  # 对角线
                mask_flat[i] = True
    else:
        # 显示下三角 (k=-1) 或 全显示
        # 如果是下三角
        for i in range(len(values_flat)):
            r, c = divmod(i, n_cols)
            if r < c:  # 上三角部分
                mask_flat[i] = True
            elif r == c:  # 对角线
                mask_flat[i] = True

    # 过滤被遮罩的数据
    x_plot = x_flat[~mask_flat]
    y_plot = y_flat[~mask_flat]
    values_plot = values_flat[~mask_flat]

    if len(values_plot) == 0:
        return

    # 计算圆的大小（基于相关系数绝对值）
    # 线性插值调整大小
    abs_values = np.abs(values_plot)
    sizes = abs_values * (max_scale - min_scale) + min_scale

    # 绘制散点
    sc = ax.scatter(
        x_plot,
        y_plot,
        s=sizes,
        c=values_plot,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        marker="o",
        edgecolors="none",
    )

    # 设置坐标轴
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_names)

    # 调整坐标轴范围
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # 翻转Y轴

    # 设置纵横比
    ax.set_aspect("equal")

    # 移除边框线
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 移除刻度线但保留标签
    ax.tick_params(axis="both", which="both", length=0)

    return sc


def plot_static_features_correlation(
    data_path: str = None,
    enable_clustering: bool = True,
    show_upper_triangle: bool = True,
    use_circles: bool = False,
    figsize: tuple = (12, 10),
    cmap: str = "coolwarm",
    save_path: str = None,
):
    """
    绘制静态特征相关性热力图

    Args:
        data_path: 特征数据路径，默认使用 OUTPUT_DIR/q4_featured_data.csv
        enable_clustering: 是否启用聚类（层次聚类），默认True
        show_upper_triangle: 是否只显示上三角矩阵（避免重复），默认True
        use_circles: 是否使用圆形而非方块，默认False
        figsize: 图形大小
        cmap: 颜色映射，推荐 'coolwarm', 'RdBu_r', 'vlag'
        save_path: 保存路径，默认保存到 OUTPUT_DIR/figures/

    Returns:
        correlation_matrix: 相关性矩阵DataFrame
    """
    # 加载数据/
    if data_path is None:
        data_path = OUTPUT_DIR / "q4_featured_data.csv"

    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)

    # 定义静态特征列表
    static_features = [
        "ballroom_partner",  # 舞伴
        "celebrity_industry",  # 行业
        "celebrity_homestate",  # 家乡
        "celebrity_age_during_season",  # 年龄
        "gender",  # 性别
        "is_from_usa",  # 是否美国人
        "ballroom_partner_count",  # 舞伴参赛次数
        "is_legacy_season",  # 是否经典赛季
        "season_total_contestants",  # 当季选手总数
    ]

    # 英文标签映射
    feature_labels = {
        "ballroom_partner": "Ballroom Partner",
        "celebrity_industry": "Industry",
        "celebrity_homestate": "Home State",
        "celebrity_age_during_season": "Age",
        "gender": "Gender",
        "is_from_usa": "Is From USA",
        "ballroom_partner_count": "Partner Experience",
        "is_legacy_season": "Legacy Season",
        "season_total_contestants": "Total Contestants",
    }

    print(f"\n分析的静态特征 ({len(static_features)} 个):")
    for feat in static_features:
        print(f"  - {feat}: {feature_labels.get(feat, feat)}")

    # 准备数据：对类别变量进行编码
    df_encoded = df[static_features].copy()

    # 类别型特征需要编码
    categorical_features = [
        "ballroom_partner",
        "celebrity_homestate",
        "celebrity_industry",
    ]

    for col in categorical_features:
        if col in df_encoded.columns:
            if (
                df_encoded[col].dtype == "object"
                or df_encoded[col].dtype.name == "string"
            ):
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].fillna("Unknown"))
                print(f"\n编码特征 '{col}': {len(le.classes_)} 个类别")

    # 计算相关性矩阵
    print("\n计算相关性矩阵...")
    correlation_matrix = df_encoded.corr()

    # 使用标签
    correlation_matrix.index = [
        feature_labels.get(f, f) for f in correlation_matrix.index
    ]
    correlation_matrix.columns = [
        feature_labels.get(f, f) for f in correlation_matrix.columns
    ]

    # 创建遮罩（只显示上三角或下三角）
    if show_upper_triangle:
        mask = np.triu(
            np.ones_like(correlation_matrix, dtype=bool), k=0
        )  # 包含对角线，用于grid search order
        heatmap_mask = np.triu(
            np.ones_like(correlation_matrix, dtype=bool), k=1
        )  # 绘图时去掉对角线
        mask_label = "Upper"
    else:
        mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=0)
        heatmap_mask = np.tril(np.ones_like(correlation_matrix, dtype=bool), k=-1)
        mask_label = "Lower"

    # 绘制热力图
    # 如果启用聚类，先通过clustermap计算行/列顺序
    reordered_corr = correlation_matrix.copy()

    if enable_clustering:
        print("执行层次聚类以确定特征顺序...")
        # 临时的clustermap用于计算顺序
        g_order = sns.clustermap(
            correlation_matrix,
            metric="euclidean",
            method="average",
            figsize=(1, 1),  # 很小，不显示
        )
        row_order = g_order.dendrogram_row.reordered_ind
        col_order = g_order.dendrogram_col.reordered_ind
        plt.close(g_order.fig)

        # 根据聚类结果重新排序矩阵
        reordered_corr = correlation_matrix.iloc[row_order, col_order]

    plt.figure(figsize=figsize)

    if use_circles:
        # 使用自定义圆形绘制
        print(f"绘制圆形热力图（{mask_label}）...")
        sc = plot_circle_heatmap(
            reordered_corr, ax=plt.gca(), cmap=cmap, show_upper=show_upper_triangle
        )
        plt.colorbar(sc, label="Correlation Coefficent", shrink=0.8)

        plt.title(
            f"Static Features Correlation Analysis",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

    else:
        # 标准热力图（无聚类树状图，但顺序可能已经重排）
        # 如果需要显示聚类树状图，就不能用重排后的矩阵配合普通heatmap，而要直接用clustermap

        if enable_clustering and not use_circles:
            # 使用clustermap直接绘图（带树状图）
            print(f"绘制标准聚类热力图（{mask_label}）...")

            # clustermap 的 mask 需要匹配重新排序后的矩阵，但clustermap会自动处理
            # 这里的mask参数是传给内部heatmap的，所以应该是原始矩阵的大小

            # 注意：seaborn的clustermap mask处理比较棘手，通常mask用于原始矩阵
            # 如果mask是对称的（如上三角），重排后可能就不对了。
            # 实际上，clustermap 不太适合做“只显示上三角且聚类”，因为聚类会打乱行列，导致上三角形状不再是上三角
            # 但是，如果“上三角”是指“不重复显示”，那么在聚类混乱后，mask 应该只是 mask 掉 (i,j) 和 (j,i) 中的一个？
            # 不，通常聚类后，就不再是对称矩阵的简单上/下三角展示了。

            # 如果用户坚持要“聚类”且“只显示一半”，这在数学上有点矛盾（除非不画树状图，只按顺序画）
            # 简单的做法是：不用clustermap画图，只用它的顺序，然后用heatmap画（就像上面圆形逻辑一样），配合自定义mask

            # 为了满足“好看”和“去掉分割线”，我们用 heatmap 绘制重排后的矩阵

            # 重建mask：对于重排后的矩阵，我们仍然只希望显示一半吗？
            # 如果行列顺序变了，"上三角"就不再是原来的上三角。
            # 通常的做法是：只显示上三角本身是基于原始顺序的。如果聚类了，通常显示全图，或者mask掉对角线。
            # 但用户强烈要求“只要一半”。

            # 如果我们强行只显示重排后矩阵的上三角，会丢失信息吗？
            # 会。因为重排后，Corr(A, B) 可能在下三角，而 Corr(C, D) 在上三角。
            # 所以，如果启用了聚类，还是显示全图比较合理，或者只 mask 对角线。
            # 但用户说了“要一半”。

            # 妥协方案：如果启用了聚类，我们这里就只 mask 对角线，或者只能显示全矩阵（如果非要聚类）。
            # 或者，我们假设用户说的“一半”是指“不重复”。
            # 在这种情况下，圆形图比较好处理（只画 x>y 的点）。
            # 对于方块图，可以使用 mask=np.tril(...) on the REORDERED matrix.

            # 构建重排后矩阵的mask
            if show_upper_triangle:
                current_mask = np.triu(np.ones_like(reordered_corr, dtype=bool), k=1)
            else:
                current_mask = np.tril(np.ones_like(reordered_corr, dtype=bool), k=-1)

            sns.heatmap(
                reordered_corr,
                mask=current_mask,  # 应用遮罩
                annot=True,
                fmt=".2f",
                cmap=cmap,
                center=0,
                square=True,
                linewidths=0,  # 去掉分割线
                linecolor="gray",
                cbar_kws={"label": "Correlation Coefficent"},
                vmin=-1,
                vmax=1,
            )

            plt.title(
                f"Static Features Correlation",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

        else:
            # 不聚类，标准热力图
            print(f"绘制标准热力图（{mask_label}）...")

            # 这种情况下，mask 是简单的
            if show_upper_triangle:
                current_mask = np.triu(
                    np.ones_like(correlation_matrix, dtype=bool), k=1
                )
            else:
                current_mask = np.tril(
                    np.ones_like(correlation_matrix, dtype=bool), k=-1
                )

            sns.heatmap(
                correlation_matrix,
                mask=current_mask,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                center=0,
                square=True,
                linewidths=0,  # 去掉分割线
                linecolor="gray",
                cbar_kws={"label": "Correlation Coefficent"},
                vmin=-1,
                vmax=1,
            )

            plt.title(
                f"Static Features Correlation",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()

    # 保存图形
    if save_path is None:
        save_dir = OUTPUT_DIR / "question4_res" / "figures"
        save_dir.mkdir(parents=True, exist_ok=True)
        cluster_suffix = "_clustered" if enable_clustering else ""
        circle_suffix = "_circle" if use_circles else ""
        triangle_suffix = "_upper" if show_upper_triangle else "_lower"
        save_path = (
            save_dir
            / f"static_features_correlation{cluster_suffix}{circle_suffix}{triangle_suffix}.png"
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n图形已保存: {save_path}")

    # 显示图形
    plt.show()

    # 打印相关性统计信息
    print("\n=== Summary Statistics ===")

    # 获取上三角矩阵的相关系数（排除对角线）
    # 使用原始矩阵进行统计
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    upper_triangle = correlation_matrix.where(mask)

    # 展平并排序
    correlations = upper_triangle.unstack().dropna()
    correlations_abs = correlations.abs().sort_values(ascending=False)

    print(f"\nStrong Correlations (|r| > 0.5):")
    strong_corr = correlations_abs[correlations_abs > 0.5]
    if len(strong_corr) > 0:
        for (feat1, feat2), corr_val in strong_corr.items():
            actual_corr = correlations.loc[feat1, feat2]
            print(f"  {feat1} ↔ {feat2}: {actual_corr:.3f}")
    else:
        print("  None")

    print(f"\nMedium Correlations (0.3 < |r| ≤ 0.5):")
    medium_corr = correlations_abs[(correlations_abs > 0.3) & (correlations_abs <= 0.5)]
    if len(medium_corr) > 0:
        for (feat1, feat2), corr_val in medium_corr.items():
            actual_corr = correlations.loc[feat1, feat2]
            print(f"  {feat1} ↔ {feat2}: {actual_corr:.3f}")
    else:
        print("  None")

    print(
        f"\nWeak Correlations (|r| ≤ 0.3): {len(correlations_abs[correlations_abs <= 0.3])} pairs"
    )

    print(f"\nMean Absolute Correlation: {correlations_abs.mean():.3f}")
    print(f"Max Correlation: {correlations_abs.max():.3f}")
    print(f"Min Correlation: {correlations_abs.min():.3f}")

    return correlation_matrix


if __name__ == "__main__":
    print("=" * 80)
    print("Static Features Correlation Analysis")
    print("=" * 80)

    # 绘制带聚类的圆形热力图（推荐，上三角）
    print("\n[Option 1] Clustered Circle Heatmap (Upper Triangle)")
    print("-" * 80)
    corr_matrix_clustered = plot_static_features_correlation(
        enable_clustering=True,
        show_upper_triangle=True,
        use_circles=True,  # 使用圆形
        cmap="coolwarm",
    )

    print("\n\n" + "=" * 80)

    # 绘制无聚类的标准热力图（下三角，方块）
    print("\n[Option 2] Standard Heatmap (Lower Triangle, No Lines)")
    print("-" * 80)
    corr_matrix_standard = plot_static_features_correlation(
        enable_clustering=False,
        show_upper_triangle=False,
        use_circles=False,  # 使用方块
        cmap="coolwarm",
    )

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
