import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

from configs.config import DATA_DIR, OUTPUT_DIR

# 设置中文字体
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
rcParams["axes.unicode_minus"] = False


def load_all_bottom_data():
    """加载包含两种方法分别标记的数据"""
    df = pd.read_csv(
        OUTPUT_DIR / "question2_res" / "q4" / "controversial_bottom_two_all.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    return df


def load_elimination_data():
    """加载提前淘汰数据（包含is_eliminated_early字段）"""
    df = pd.read_csv(
        OUTPUT_DIR / "question2_res" / "q4" / "controversial_both_methods_bottom.csv",
        sep=",",
        header=0,
        encoding="utf-8",
    )
    # 只保留需要标记X的记录（is_eliminated_early=1）
    df_eliminated = df[df["is_eliminated_early"] == 1][
        ["celebrity_name", "season", "week"]
    ].copy()
    return df_eliminated


def create_dual_lane_electrophoresis_plot(
    df: pd.DataFrame, df_eliminated: pd.DataFrame, output_path
):
    """
    创建双泳道电泳图

    每个选手分成两个子泳道：
    - 左侧子泳道：排名法 (method1) 的垫底情况
    - 右侧子泳道：百分比法 (method2) 的垫底情况

    可以对比两种方法的差异
    """
    # 创建选手标签
    df["contestant_label"] = df["season"].astype(str) + "-" + df["celebrity_name"]

    # 必须展示的4个选手
    required_contestants = [
        "Jerry Rice",
        "Billy Ray Cyrus",
        "Bristol Palin",
        "Bobby Bones",
    ]

    contestant_info = (
        df.groupby("contestant_label")
        .agg(
            {
                "season": "first",
                "celebrity_name": "first",
                "weeks_participated": "first",
                "placement": "first",
                "method1_is_bottom_two": "sum",  # 排名法垫底次数
                "method2_is_bottom_two": "sum",  # 百分比法垫底次数
            }
        )
        .reset_index()
    )
    contestant_info.rename(
        columns={
            "method1_is_bottom_two": "method1_bottom_count",
            "method2_is_bottom_two": "method2_bottom_count",
        },
        inplace=True,
    )

    # 筛选必须展示的选手
    required_info = contestant_info[
        contestant_info["celebrity_name"].isin(required_contestants)
    ].copy()

    # 剩余选手中随机抽取6个
    other_info = contestant_info[
        ~contestant_info["celebrity_name"].isin(required_contestants)
    ].copy()

    if len(other_info) >= 6:
        other_info = other_info.sample(n=6, random_state=42)

    # 合并选手信息
    contestant_info = pd.concat([required_info, other_info], ignore_index=True)

    # 按赛季和名字排序
    contestant_info = contestant_info.sort_values(["season", "celebrity_name"])
    contestant_labels = contestant_info["contestant_label"].tolist()

    # 配色方案
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    lane_bg_colors = [colors[i % 20] for i in range(len(contestant_labels))]
    method1_color = "#2E86AB"  # 蓝色系 - 排名法
    method2_color = "#A23B72"  # 紫红色系 - 百分比法

    fig, ax = plt.subplots(figsize=(26, 14))

    max_weeks = df["weeks_participated"].max()

    # 每个选手占用的宽度（包含两个子泳道）
    lane_width = 1.6  # 总宽度
    sub_lane_width = 0.7  # 每个子泳道宽度
    gap = 0.1  # 子泳道间隙

    for idx, label in enumerate(contestant_labels):
        contestant_data = df[df["contestant_label"] == label]
        weeks_participated = contestant_data["weeks_participated"].iloc[0]
        season = contestant_data["season"].iloc[0]
        celebrity_name = contestant_data["celebrity_name"].iloc[0]

        center_x = idx * lane_width

        left_x = center_x - sub_lane_width / 2 - gap / 2
        right_x = center_x + sub_lane_width / 2 + gap / 2

        # 为整个选手区域添加淡色背景，用于区分不同选手
        contestant_bg_color = lane_bg_colors[idx]

        # 整体泳道背景
        overall_lane_bg = Rectangle(
            (center_x - lane_width / 2 + 0.05, -0.5),
            lane_width - 0.1,
            max_weeks + 2,
            facecolor=contestant_bg_color,
            alpha=0.08,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(overall_lane_bg)

        ax.vlines(
            left_x,
            -0.5,
            max_weeks + 2,
            colors=method1_color,
            linestyles=":",
            alpha=0.3,
            linewidth=1,
            zorder=0,
        )

        ax.vlines(
            right_x,
            -0.5,
            max_weeks + 2,
            colors=method2_color,
            linestyles=":",
            alpha=0.3,
            linewidth=1,
            zorder=0,
        )

        dna_width = 0.02  # DNA条带宽度
        # 共用的条带参数（用于局部 DNA 条带与终点标记）
        band_width = 0.24
        band_height = 0.12
        band_pad = 0.08

        dna_left = FancyBboxPatch(
            (left_x - dna_width / 2, 0),  # x居中: left_x - (宽度/2)
            dna_width,  # width
            weeks_participated,  # height
            boxstyle="round,pad=0.01",
            facecolor=method1_color,
            edgecolor="none",
            alpha=0.12,
            zorder=1,
        )
        ax.add_patch(dna_left)

        dna_right = FancyBboxPatch(
            (right_x - dna_width / 2, 0),  # x居中: right_x - (宽度/2)
            dna_width,  # width
            weeks_participated,
            boxstyle="round,pad=0.01",
            facecolor=method2_color,
            edgecolor="none",
            alpha=0.12,
            zorder=1,
        )
        ax.add_patch(dna_right)

        # ========== 3. 绘制终点标记（灰色） ==========
        terminal_week = weeks_participated

        # 左子泳道终点 - 使用与条带相同的尺寸
        terminal_left = FancyBboxPatch(
            (left_x - band_width / 2, terminal_week - band_height / 2),
            band_width,  # width (match band)
            band_height,  # height (match band)
            boxstyle=f"round,pad={band_pad}",
            facecolor="gray",
            edgecolor="darkgray",
            linewidth=0.8,
            alpha=0.6,
            zorder=6,
        )
        ax.add_patch(terminal_left)

        terminal_right = FancyBboxPatch(
            (right_x - band_width / 2, terminal_week - band_height / 2),
            band_width,
            band_height,
            boxstyle=f"round,pad={band_pad}",
            facecolor="gray",
            edgecolor="darkgray",
            linewidth=0.8,
            alpha=0.6,
            zorder=6,
        )
        ax.add_patch(terminal_right)

        ax.text(
            center_x,
            weeks_participated + 0.9,
            f"Due week: {int(weeks_participated)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="darkgray",
            zorder=10,
        )

        # 获取该选手的提前淘汰周数
        eliminated_weeks = df_eliminated[
            (df_eliminated["season"] == season)
            & (df_eliminated["celebrity_name"] == celebrity_name)
        ]["week"].values

        # ========== 4. 绘制排名法垫底标记（左子泳道） ==========
        method1_bottom_weeks = contestant_data[
            contestant_data["method1_is_bottom_two"] == 1
        ]["week"].values

        # DNA条带参数
        band_width = 0.24
        band_height = 0.12
        band_pad = 0.08

        for week in method1_bottom_weeks:
            # 跳过终点周
            if abs(week - terminal_week) < 0.1:
                continue

            band_x = left_x - band_width / 2

            band = FancyBboxPatch(
                (band_x, week - band_height / 2),  # x, y (起始位置)
                band_width,  # width
                band_height,  # height
                boxstyle=f"round,pad={band_pad}",  # 圆角样式
                facecolor=method1_color,
                edgecolor="none",
                alpha=0.85,  # 稍微透明一点，更有质感
                zorder=5,
            )
            ax.add_patch(band)

            # 周数标记 (移到条带旁边或上方，这里放在条带上方)
            ax.text(
                left_x,
                week + 0.35,
                f"Week {int(week)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=method1_color,
                fontweight="bold",
                zorder=10,
                alpha=0.9,
            )

            # highlight OUT 标记，用于标记选手可能的提前淘汰周
            if week in eliminated_weeks:
                ax.text(
                    left_x,
                    week,
                    "OUT",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="red",
                    fontweight="bold",
                    zorder=11,
                )

        # ========== 5. 绘制百分比法垫底标记（右子泳道） ==========
        method2_bottom_weeks = contestant_data[
            contestant_data["method2_is_bottom_two"] == 1
        ]["week"].values

        for week in method2_bottom_weeks:
            # 跳过终点周
            if abs(week - terminal_week) < 0.1:
                continue

            band_x = right_x - band_width / 2

            band = FancyBboxPatch(
                (band_x, week - band_height / 2),
                band_width,
                band_height,
                boxstyle=f"round,pad={band_pad}",
                facecolor=method2_color,
                edgecolor="none",
                alpha=0.85,
                zorder=5,
            )
            ax.add_patch(band)

            ax.text(
                right_x,
                week + 0.35,
                f"Week {int(week)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=method2_color,
                fontweight="bold",
                zorder=10,
                alpha=0.9,
            )

            if week in eliminated_weeks:
                ax.text(
                    right_x,
                    week,
                    "OUT",
                    ha="center",
                    va="center",
                    fontsize=10,  # 适度调整字号
                    color="red",  # 改为红色更醒目
                    fontweight="bold",
                    zorder=11,
                )

        # ========== 6. 子泳道标签 ==========
        ax.text(
            left_x,
            -0.8,
            "Rank",
            ha="center",
            va="top",
            fontsize=8,
            color=method1_color,
            fontweight="bold",
        )
        ax.text(
            right_x,
            -0.8,
            "Percentage",
            ha="center",
            va="top",
            fontsize=8,
            color=method2_color,
            fontweight="bold",
        )

    # 设置横坐标
    ax.set_xticks([i * lane_width for i in range(len(contestant_labels))])

    short_labels = []
    for label in contestant_labels:
        season, name = label.split("-", 1)
        if len(name) > 12:
            name = name[:10] + ".."
        short_labels.append(f"S{season}\n{name}")

    ax.set_xticklabels(
        short_labels,
        rotation=0,
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

    # 设置纵坐标
    ax.set_ylabel("Weeks", fontsize=14, fontweight="bold")
    ax.set_xlabel("Season-Name Pair", fontsize=14, fontweight="bold")
    ax.set_title(
        "Electrophoresis Image: Voting Results Applying Judge-Based Rating",
        fontsize=16,
        fontweight="bold",
        pad=25,
    )

    ax.set_ylim(-1.5, max_weeks + 2)
    ax.set_xlim(-lane_width, len(contestant_labels) * lane_width - 0.4)

    ax.yaxis.grid(True, linestyle=":", alpha=0.4, color="gray", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(range(0, int(max_weeks) + 2))

    legend_elements = [
        Rectangle(
            (0, 0), 1, 1, fc=method1_color, alpha=0.85, label="Rank Method Bottom"
        ),
        Rectangle(
            (0, 0), 1, 1, fc=method2_color, alpha=0.85, label="Percentage Method Bottom"
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            fc="gray",
            alpha=0.6,
            ec="darkgray",
            linewidth=1,
            label="Final Elimination Week",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="w",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=12,
            markeredgewidth=2,
            label="Early Elimination (OUT)",
            linestyle="None",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=11,
        framealpha=0.9,
        edgecolor="black",
    )

    plt.tight_layout()
    plt.savefig(
        output_path / "electrophoresis_dual_lane.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_path / "electrophoresis_dual_lane.pdf", bbox_inches="tight")
    print(f"[OK] Saved to: {output_path / 'electrophoresis_dual_lane.png'}")

    plt.close()


def main():
    df = load_all_bottom_data()
    df_eliminated = load_elimination_data()

    output_path = OUTPUT_DIR / "question2_res" / "q4"
    output_path.mkdir(parents=True, exist_ok=True)
    create_dual_lane_electrophoresis_plot(df, df_eliminated, output_path)


if __name__ == "__main__":
    main()
