import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_validate,
    KFold,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import joblib
import re

from configs.config import DATA_DIR, OUTPUT_DIR


def rf_train(
    X: pd.DataFrame, y: pd.Series, feature_type: str = "rank", output_dir=None
):
    """使用Pipeline进行随机森林训练

    Args:
        X: 特征数据
        y: 目标变量
        feature_type: 特征类型，"rank" 或 "pct"
        output_dir: 输出目录，如果为None则使用OUTPUT_DIR
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    print(f"\n{'='*70}")
    print(f"训练 {feature_type.upper()} 特征的随机森林模型")
    print(f"{'='*70}")

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\n数据划分:")
    print(f"  训练集: {X_train.shape[0]} 样本 × {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 样本 × {X_test.shape[1]} 特征")

    # 构建Pipeline
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    # 超参数网格搜索空间（基于之前RandomizedSearch的结果优化）
    param_grid = {
        "rf__n_estimators": [200, 300, 500],
        "rf__max_depth": [12, 14, 16],
        "rf__min_samples_split": [2, 3, 4],
        "rf__min_samples_leaf": [3, 5],
        "rf__max_features": ["sqrt", None],
    }

    # 使用GridSearchCV进行网格搜索
    print(f"\n执行网格搜索...")
    print(
        f"参数组合总数: {len(param_grid['rf__n_estimators']) * len(param_grid['rf__max_depth']) * len(param_grid['rf__min_samples_split']) * len(param_grid['rf__min_samples_leaf']) * len(param_grid['rf__max_features'])}"
    )

    inner_cv = KFold(n_splits=5, random_state=42, shuffle=True)
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring="r2",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    print(f"\n最优参数: {search.best_params_}")
    print(f"最优CV得分 (R²): {search.best_score_:.4f}")

    # 显示Top 5参数组合
    print(f"\nTop 5 参数组合:")
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    for idx, row in results_df.head(5).iterrows():
        print(
            f"  Rank {int(row['rank_test_score'])}: R²={row['mean_test_score']:.4f} (±{row['std_test_score']:.4f}), params={row['params']}"
        )

    # 在测试集上评估
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n测试集性能:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²: {r2:.6f}")

    # 10折交叉验证检查过拟合
    print(f"\n执行10折交叉验证...")
    cv_10fold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_validate(
        best_model,
        X,
        y,
        cv=cv_10fold,
        scoring=["r2", "neg_mean_squared_error"],
        n_jobs=-1,
        return_train_score=True,
    )

    train_r2_mean = cv_results["train_r2"].mean()
    train_r2_std = cv_results["train_r2"].std()
    test_r2_mean = cv_results["test_r2"].mean()
    test_r2_std = cv_results["test_r2"].std()
    train_rmse_mean = np.sqrt(-cv_results["train_neg_mean_squared_error"].mean())
    test_rmse_mean = np.sqrt(-cv_results["test_neg_mean_squared_error"].mean())

    print(f"\n10折交叉验证结果:")
    print(f"  训练集 R²: {train_r2_mean:.4f} ± {train_r2_std:.4f}")
    print(f"  验证集 R²: {test_r2_mean:.4f} ± {test_r2_std:.4f}")
    print(f"  训练集 RMSE: {train_rmse_mean:.6f}")
    print(f"  验证集 RMSE: {test_rmse_mean:.6f}")

    # 过拟合检查
    overfit_gap = train_r2_mean - test_r2_mean
    print(f"\n过拟合检查:")
    print(f"  训练集与验证集R²差距: {overfit_gap:.4f}")
    if overfit_gap > 0.1:
        print(f"  ⚠️  警告: 可能存在过拟合 (差距 > 0.1)")
    elif overfit_gap > 0.05:
        print(f"  ⚡ 注意: 轻微过拟合倾向 (差距 > 0.05)")
    else:
        print(f"  ✓ 模型泛化能力良好")

    # 特征重要性分析
    print(f"\n特征重要性分析:")
    rf_model = best_model.named_steps["rf"]
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    print(f"\nTop 20 最重要特征:")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.6f}")

    # 保存特征重要性
    importance_path = output_dir / f"feature_importance_{feature_type}.csv"
    feature_importance.to_csv(importance_path, index=False, encoding="utf-8")
    print(f"\n✓ 完整特征重要性已保存到: {importance_path}")

    # 保存模型
    model_path = output_dir / f"rf_model_{feature_type}.pkl"
    joblib.dump(best_model, model_path)
    print(f"✓ 模型已保存到: {model_path}")

    return best_model, {
        "feature_type": feature_type,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": X.shape[1],
        "best_params": search.best_params_,
        "cv_score": search.best_score_,
        "test_rmse": rmse,
        "test_r2": r2,
        "cv10_train_r2_mean": train_r2_mean,
        "cv10_train_r2_std": train_r2_std,
        "cv10_test_r2_mean": test_r2_mean,
        "cv10_test_r2_std": test_r2_std,
        "cv10_train_rmse": train_rmse_mean,
        "cv10_test_rmse": test_rmse_mean,
        "overfit_gap": overfit_gap,
        "top_10_features": feature_importance.head(10)[
            ["feature", "importance"]
        ].to_dict("records"),
    }


def main():
    """
    Q4 随机森林模型训练：分析舞伴和选手特征对评委评分和粉丝投票的影响

    问题核心：
    - How much do such things (pro dancers, celebrity characteristics) impact
      how well a celebrity will do in the competition?
    - Do they impact judges scores and fan votes in the same way?

    建模思路：
    - 因变量1: 评委评分均值（judge_score_mean）→ 转换为排名/百分比
    - 因变量2: 观众投票均值（audience_votes_mean）→ 转换为排名/百分比
    - 自变量: 静态特征 + 第一周和第二周的排名/百分比表现

    通过加入早期表现特征，分析：
    1. 静态特征在控制表现后的独立贡献
    2. 静态特征重要性是否被表现特征"挤压"
    3. 舞伴和选手特征对评委和粉丝的真实影响
    """

    # 定义输出目录
    output_dir = OUTPUT_DIR / "question4_res"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        OUTPUT_DIR / "q4_featured_data.csv", sep=",", header=0, encoding="utf-8"
    )

    print(f"\n{'='*70}")
    print("Q4: 舞伴和选手特征对评委评分/粉丝投票的影响分析")
    print("（控制早期表现后的静态特征独立贡献）")
    print(f"{'='*70}")
    print(f"原始数据: {df.shape[0]} 样本 × {df.shape[1]} 特征")

    # ==================== 计算因变量 ====================
    print(f"\n计算因变量...")

    # 1. 评委得分均值
    score_cols = [col for col in df.columns if "score_sum" in col]
    df["judge_score_mean"] = df[score_cols].mean(axis=1, skipna=True)
    print(f"  评委得分均值: judge_score_mean (来自 {len(score_cols)} 周数据)")

    # 2. 观众投票均值
    audience_cols = [
        col
        for col in df.columns
        if "audience_votes" in col and col != "total_audience_votes"
    ]
    df["audience_votes_mean"] = df[audience_cols].mean(axis=1, skipna=True)
    print(f"  观众投票均值: audience_votes_mean (来自 {len(audience_cols)} 周数据)")

    # 3. 将均值转换为相对排名（百分比形式，越高越好）
    # 按season分组计算排名百分比（同一赛季内比较）
    df["judge_score_rank_pct"] = df.groupby("season")["judge_score_mean"].rank(pct=True)
    df["audience_votes_rank_pct"] = df.groupby("season")["audience_votes_mean"].rank(
        pct=True
    )

    print(f"  评委得分排名百分比: judge_score_rank_pct")
    print(f"  观众投票排名百分比: audience_votes_rank_pct")

    # ==================== 准备自变量 ====================
    print(f"\n准备自变量...")

    # 静态特征列表
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

    # 第一周和第二周的表现特征
    week_features = [
        # 第一周
        "week1_judge_rank",
        "week1_audience_rank",
        "week1_combined_rank",
        "week1_judge_pct",
        "week1_audience_pct",
        "week1_combined_pct",
        # 第二周
        "week2_judge_rank",
        "week2_audience_rank",
        "week2_combined_rank",
        "week2_judge_pct",
        "week2_audience_pct",
        "week2_combined_pct",
    ]

    # 类别编码
    obj_cols = ["ballroom_partner", "celebrity_homestate", "celebrity_industry"]
    df[obj_cols] = df[obj_cols].astype("string")

    label_encoders = {}
    for col in obj_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].fillna("Unknown"))
            label_encoders[col] = le

    # 构建完整特征矩阵（静态 + 第一周&第二周表现）
    all_features = static_features + week_features
    X = df[all_features].copy()

    # 因变量
    y_judge = df["judge_score_rank_pct"]  # 评委评分排名百分比
    y_audience = df["audience_votes_rank_pct"]  # 观众投票排名百分比

    print(f"\n自变量构成:")
    print(f"  静态特征 ({len(static_features)} 个):")
    for feat in static_features:
        print(f"    - {feat}")
    print(f"  第一周&第二周表现特征 ({len(week_features)} 个):")
    for feat in week_features:
        print(f"    - {feat}")
    print(f"\n因变量:")
    print(f"  - 模型1: judge_score_rank_pct (评委评分排名百分比)")
    print(f"  - 模型2: audience_votes_rank_pct (观众投票排名百分比)")

    # 移除包含NaN的行
    valid_mask = X.notna().all(axis=1) & y_judge.notna() & y_audience.notna()
    X_clean = X[valid_mask]
    y_judge_clean = y_judge[valid_mask]
    y_audience_clean = y_audience[valid_mask]

    print(f"\n数据清洗后: {len(X_clean)} 样本 × {len(all_features)} 特征")
    print(f"  - 静态特征: {len(static_features)} 个")
    print(f"  - 周表现特征: {len(week_features)} 个")

    # 保存标签编码器
    encoders_path = output_dir / "label_encoders.pkl"
    joblib.dump(label_encoders, encoders_path)
    print(f"✓ 标签编码器已保存到: {encoders_path}")

    # ==================== 训练两个模型 ====================
    print(f"\n" + "=" * 70)
    print("训练模型1: 预测评委评分（静态特征+第1-2周表现）")
    judge_model, judge_results = rf_train(
        X_clean, y_judge_clean, "judge_score", output_dir
    )

    print(f"\n" + "=" * 70)
    print("训练模型2: 预测观众投票（静态特征+第1-2周表现）")
    audience_model, audience_results = rf_train(
        X_clean, y_audience_clean, "audience_votes", output_dir
    )

    # ==================== 对比分析 ====================
    print(f"\n{'='*70}")
    print("模型性能对比（静态特征+第1-2周表现）")
    print(f"{'='*70}")

    print(f"\n评委评分模型 ({judge_results['n_features']} features):")
    print(f"  测试集 R²:   {judge_results['test_r2']:.4f}")
    print(
        f"  10折CV R²:   {judge_results['cv10_test_r2_mean']:.4f} ± {judge_results['cv10_test_r2_std']:.4f}"
    )
    print(f"  过拟合差距:  {judge_results['overfit_gap']:.4f}")

    print(f"\n观众投票模型 ({audience_results['n_features']} features):")
    print(f"  测试集 R²:   {audience_results['test_r2']:.4f}")
    print(
        f"  10折CV R²:   {audience_results['cv10_test_r2_mean']:.4f} ± {audience_results['cv10_test_r2_std']:.4f}"
    )
    print(f"  过拟合差距:  {audience_results['overfit_gap']:.4f}")

    # ==================== 特征重要性对比与分析 ====================
    print(f"\n{'='*70}")
    print("特征重要性详细分析")
    print(f"{'='*70}")

    # 读取两个模型的特征重要性
    judge_importance = pd.read_csv(output_dir / "feature_importance_judge_score.csv")
    audience_importance = pd.read_csv(
        output_dir / "feature_importance_audience_votes.csv"
    )

    # 标记特征类型
    judge_importance["feature_type"] = judge_importance["feature"].apply(
        lambda x: "静态特征" if x in static_features else "周表现"
    )
    audience_importance["feature_type"] = audience_importance["feature"].apply(
        lambda x: "静态特征" if x in static_features else "周表现"
    )

    # 计算各类特征的累积重要性
    print(f"\n【评委评分模型】特征重要性分组统计:")
    judge_static_sum = judge_importance[judge_importance["feature_type"] == "静态特征"][
        "importance"
    ].sum()
    judge_week_sum = judge_importance[judge_importance["feature_type"] == "周表现"][
        "importance"
    ].sum()
    print(f"  静态特征累积重要性: {judge_static_sum:.4f} ({judge_static_sum*100:.2f}%)")
    print(f"  周表现特征累积重要性: {judge_week_sum:.4f} ({judge_week_sum*100:.2f}%)")

    print(f"\n  静态特征 Top 5:")
    for _, row in (
        judge_importance[judge_importance["feature_type"] == "静态特征"]
        .head(5)
        .iterrows()
    ):
        print(f"    {row['feature']:<35}: {row['importance']:.4f}")

    print(f"\n  周表现特征 Top 5:")
    for _, row in (
        judge_importance[judge_importance["feature_type"] == "周表现"]
        .head(5)
        .iterrows()
    ):
        print(f"    {row['feature']:<35}: {row['importance']:.4f}")

    print(f"\n【观众投票模型】特征重要性分组统计:")
    audience_static_sum = audience_importance[
        audience_importance["feature_type"] == "静态特征"
    ]["importance"].sum()
    audience_week_sum = audience_importance[
        audience_importance["feature_type"] == "周表现"
    ]["importance"].sum()
    print(
        f"  静态特征累积重要性: {audience_static_sum:.4f} ({audience_static_sum*100:.2f}%)"
    )
    print(
        f"  周表现特征累积重要性: {audience_week_sum:.4f} ({audience_week_sum*100:.2f}%)"
    )

    print(f"\n  静态特征 Top 5:")
    for _, row in (
        audience_importance[audience_importance["feature_type"] == "静态特征"]
        .head(5)
        .iterrows()
    ):
        print(f"    {row['feature']:<35}: {row['importance']:.4f}")

    print(f"\n  周表现特征 Top 5:")
    for _, row in (
        audience_importance[audience_importance["feature_type"] == "周表现"]
        .head(5)
        .iterrows()
    ):
        print(f"    {row['feature']:<35}: {row['importance']:.4f}")

    # 合并对比（只看静态特征）
    print(f"\n{'='*70}")
    print("静态特征对评委/粉丝影响对比（排除周表现影响后）")
    print(f"{'='*70}")

    judge_static = judge_importance[judge_importance["feature_type"] == "静态特征"][
        ["feature", "importance"]
    ].copy()
    judge_static.columns = ["feature", "importance_judge"]

    audience_static = audience_importance[
        audience_importance["feature_type"] == "静态特征"
    ][["feature", "importance"]].copy()
    audience_static.columns = ["feature", "importance_audience"]

    comparison = judge_static.merge(audience_static, on="feature")
    comparison["diff"] = (
        comparison["importance_judge"] - comparison["importance_audience"]
    )
    comparison["abs_diff"] = abs(comparison["diff"])
    comparison = comparison.sort_values("abs_diff", ascending=False)

    print(f"\n静态特征对评委和粉丝影响的差异排序:")
    print(f"{'特征':<35} {'评委重要性':>12} {'粉丝重要性':>12} {'差异':>10}")
    print("-" * 70)
    for _, row in comparison.iterrows():
        direction = "→评委" if row["diff"] > 0 else "→粉丝"
        print(
            f"{row['feature']:<35} {row['importance_judge']:>12.4f} {row['importance_audience']:>12.4f} {row['diff']:>+10.4f} {direction}"
        )

    # 保存对比结果（包含所有特征）
    all_comparison = judge_importance.merge(
        audience_importance, on="feature", suffixes=("_judge", "_audience")
    )
    all_comparison["diff"] = (
        all_comparison["importance_judge"] - all_comparison["importance_audience"]
    )
    all_comparison_path = output_dir / "feature_importance_comparison.csv"
    all_comparison.to_csv(all_comparison_path, index=False, encoding="utf-8")
    print(f"\n✓ 完整特征重要性对比已保存到: {all_comparison_path}")

    # 保存静态特征对比
    static_comparison_path = output_dir / "feature_importance_static_only.csv"
    comparison.to_csv(static_comparison_path, index=False, encoding="utf-8")
    print(f"✓ 静态特征重要性对比已保存到: {static_comparison_path}")

    print(f"\n{'='*70}")
    print("核心结论")
    print(f"{'='*70}")

    # 找出对评委影响更大的静态特征
    judge_dominated = comparison[comparison["diff"] > 0.01]["feature"].tolist()
    audience_dominated = comparison[comparison["diff"] < -0.01]["feature"].tolist()

    print(f"\n1. 模型整体表现:")
    print(f"   评委评分模型 R² = {judge_results['test_r2']:.4f}")
    print(f"   观众投票模型 R² = {audience_results['test_r2']:.4f}")

    print(f"\n2. 特征类型贡献度:")
    print(
        f"   【评委评分】静态特征贡献: {judge_static_sum*100:.2f}%, 周表现贡献: {judge_week_sum*100:.2f}%"
    )
    print(
        f"   【观众投票】静态特征贡献: {audience_static_sum*100:.2f}%, 周表现贡献: {audience_week_sum*100:.2f}%"
    )

    if judge_static_sum < 0.2 and audience_static_sum < 0.2:
        print(f"\n   💡 关键发现: 在控制早期表现后，静态特征（舞伴、年龄、职业等）")
        print(f"      对评委评分和粉丝投票的影响都很小（<20%），说明：")
        print(f"      - 评委主要看舞蹈技巧和表现，不太受选手背景影响")
        print(f"      - 粉丝主要看实际表演，不太受静态身份影响")

    print(f"\n3. 对评委评分影响更大的静态特征:")
    if judge_dominated:
        for feat in judge_dominated:
            print(f"   - {feat}")
    else:
        print(f"   (无显著差异)")

    print(f"\n4. 对粉丝投票影响更大的静态特征:")
    if audience_dominated:
        for feat in audience_dominated:
            print(f"   - {feat}")
    else:
        print(f"   (无显著差异)")

    print(f"\n5. 答题建议:")
    print(f"   - 舞伴和选手特征对比赛结果有影响，但**不是主导因素**")
    print(
        f"   - 实际表现（早期排名）才是主导因素（占{max(judge_week_sum, audience_week_sum)*100:.0f}%+）"
    )
    print(f"   - 评委和粉丝对静态特征的反应模式基本一致")

    # 保存结果摘要
    results_summary = {
        "judge_score_model": judge_results,
        "audience_votes_model": audience_results,
        "static_features_contribution": {
            "judge_model": float(judge_static_sum),
            "audience_model": float(audience_static_sum),
        },
        "week_features_contribution": {
            "judge_model": float(judge_week_sum),
            "audience_model": float(audience_week_sum),
        },
        "static_features_comparison": comparison.to_dict("records"),
    }

    import json

    results_path = output_dir / "rf_training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 训练结果已保存到: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
