import PyComplexHeatmap as pch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def get_matrix(rows=12, cols=10, seed=42) -> pd.DataFrame:
    """生成随机数据矩阵"""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((rows, cols))
    return pd.DataFrame(
        data,
        columns=[f"Feature{i+1}" for i in range(cols)],
        index=[f"Sample{i+1}" for i in range(rows)],
    )


def get_fake_metadata(df: pd.DataFrame, seed=42) -> tuple:
    """生成虚构的注释数据 (模拟真实的业务场景)"""
    rng = np.random.default_rng(seed)

    # 1. 顶部注释数据 (针对 Columns/Features)
    # 模拟每个 Feature 的重要性 (Barplot) 和 类别 (Color Bar)
    col_meta = pd.DataFrame(index=df.columns)
    col_meta["Importance"] = rng.uniform(0, 1, size=len(df.columns))
    col_meta["Group"] = rng.choice(["Group A", "Group B"], size=len(df.columns))

    # 2. 侧边注释数据 (针对 Rows/Samples)
    # 模拟每个 Sample 的分类 (Color Bar) 和 某个指标 (Scatter)
    row_meta = pd.DataFrame(index=df.index)
    row_meta["Cluster"] = rng.choice(["C1", "C2", "C3"], size=len(df.index))
    row_meta["Age"] = rng.integers(20, 80, size=len(df.index))

    return col_meta, row_meta


def draw_heatmap(df: pd.DataFrame, fig_path: Optional[str] = None):
    """
    进阶版 PyComplexHeatmap 绘图演示
    展示：多重注释、手动 Ax 操作、聚类切分
    """
    try:
        # 设置字体和分辨率
        plt.rcParams["figure.dpi"] = 120

        # 1. 获取注释数据
        col_meta, row_meta = get_fake_metadata(df)

        # 2. 定义顶部注释 (Top Annotation)
        # 包含：Group (色块) + Importance (条形图)
        col_ha = pch.HeatmapAnnotation(
            Group=pch.anno_simple(
                col_meta["Group"], cmap="Set2", legend=True, height=4
            ),
            Importance=pch.anno_barplot(
                col_meta["Importance"],
                cmap="Greens",
                height=15,  # 柱子高一点
                legend=False,
            ),
            axis=1,  # 1=Top
            label_side="right",
            verbose=0,  # 不啰嗦
        )

        # 3. 定义右侧注释 (Right Annotation)
        # 包含：Cluster (色块) + Age (文本/简单色块)
        row_ha = pch.HeatmapAnnotation(
            Cluster=pch.anno_simple(row_meta["Cluster"], cmap="tab10", legend=True),
            Age=pch.anno_simple(
                row_meta["Age"], cmap="Reds", legend=True  # 连续值自动渐变
            ),
            axis=0,  # 0=Right
            label_side="top",
            verbose=0,
        )

        # 4. 绘制主图 (ClusterMapPlotter)
        plt.figure(figsize=(10, 8))

        cm = pch.ClusterMapPlotter(
            data=df,
            # 挂载注释
            top_annotation=col_ha,
            right_annotation=row_ha,
            # 核心切分 (Split)
            col_split=col_meta["Group"],  # 按照 Group 列切分 Feature
            row_split=2,  # 强行把 Sample 切成 2 份 (K-Means)
            col_split_gap=1,  # 缝隙宽度
            row_split_gap=1,
            # 视觉控制
            label="Z-Score",  # 图例标题
            cmap="RdBu_r",  # 红蓝配色
            center=0,  # 0为中心
            annot=True,  # 显示数值
            fmt=".1f",  # 保留1位小数
            # 树状图
            col_dendrogram=True,  # 显示列聚类树
            row_dendrogram=True,  # 显示行聚类树
            show_rownames=True,
            show_colnames=True,
        )

        # ==========================================
        # 5. 高级操作：手动修改 Ax (Manual Ax Manipulation)
        # ==========================================

        # 目标：在顶部 Importance 条形图上画一条红色的阈值线 (y=0.8)

        # A. 找到那个 Ax
        # col_ha.annotations 是一个列表，顺序是我们定义的 [Group, Importance]
        # 所以 index 1 是 Importance
        imp_ax = col_ha.annotations[1].ax

        # B. 调用 Matplotlib 原生 API
        # 注意：对于顶部注释，x轴是特征，y轴是数值
        imp_ax.axhline(y=0.8, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
        imp_ax.text(0, 0.85, "Threshold", color="red", fontsize=8, fontweight="bold")

        print("绘图完成，并已手动添加阈值线。")
        plt.show()

    except Exception as e:
        print(f"绘制热力图时出错: {e}")
        import traceback

        traceback.print_exc()


def main():
    df = get_matrix()
    draw_heatmap(df)


if __name__ == "__main__":
    main()
