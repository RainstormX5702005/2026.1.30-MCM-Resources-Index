# MCM 中学习过的分类算法

所谓 **分类 (Classification)** 指的就是在根据用户提供的数据训练之后程序利用训练好的模型去推断一个新的个体的类别的建模模式，**决策树 (Decision Tree)** 算法就是分类模型的雏形。

## 分类模型原理及用法列举

对于简单的 Yes or No 问题，只需要使用 **逻辑回归 (Logistic Regression)** 即可分类：

- [logr](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)

常见的分类模型如下所示，可通过知识链接查看其原型和初步使用方法：

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

## 简单例子

