# MCM 中学习过的回归算法

与分类仅仅做分析不同，**回归 (Regression)** 注重于将样本中的参数通过特定的数学模型转化为一个可以具体描述的数学公式，并且根据其中自变量与因变量的关系去预测一个自变量在该关系下的因变量。

以最常见的 **线性回归 (Linear Regression)** 作为例子，我们只需要知道若干个点位的关系并且它们满足一定的要求时，就可以使用一条直线去描述这些点的因果联系。

最终可以拟合出 $\hat{y} = \sum_{i=0}^n x_iw_i  + \hat{\omega_0}$ 的形式。

## 常见的回归算法及其应用方法

最常见的就是线性回归模型，如下所示：

- 线性回归 (Linear Regression)：
    - 原理指南：[lg](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)
    - 使用方法：[sk_lg](https://scikit-learn.org/stable/modules/linear_model.html)

请不要觉得线性回归就是差的，有时候可能非线性模型的训练结果一般，反而线性回归的结果又快又不失准确性。

除了线性回归，之前介绍的分类模型也都具有他们的回归形式，请查阅：

- 决策树 Decision Tree: 
    - 知识链接：[dt](https://www.geeksforgeeks.org/machine-learning/decision-tree/)
- ⭐随机森林 Random Forest
    - 知识链接：[rf](https://www.geeksforgeeks.org/machine-learning/random-forest-algorithm-in-machine-learning/)
    - 使用方法：[sk_rf](https://scikit-learn.org/stable/modules/tree.html)
- ⭐⭐⭐梯度上升树 Gradient Boosting Tree: 
    - 知识链接：[gbt](https://www.geeksforgeeks.org/machine-learning/ml-gradient-boosting/)
    - 使用方法：[sk_gbt](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)
- ⭐⭐XGBoost: 
    - 知识链接：[xgb](https://www.geeksforgeeks.org/machine-learning/xgboost/)
    - 使用方法：[xgb_xgb](https://xgboost.readthedocs.io/en/latest/python/sklearn_estimator.html)