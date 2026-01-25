import numpy as np
import pandas as pd

# example1：使用 `pandas` 读取并解析著名的 Iris 数据集


def main() -> None:

    iris = pd.read_csv(
        "D:\\Projects\\Coursework\\math_modeling\\examples\\pd_examples\\data\\Iris.csv",
        header=0,
        sep=",",
        encoding="utf-8",
    )

    # ----------------------- Step1: 数据集基本信息查询 ------------------------

    iris.info()  # info() 方法用于直接在终端输出数据集的简要信息，甚至包括内存占用
    iris_describe = iris.describe()  # describe() 方法用于生成数据集的常见统计信息

    print("\n", iris.isnull().sum())  # 检查数据集中是否存在缺失值

    # ----------------------- Step2: 数据集基本操作 ------------------------

    # df.loc 可以传递行索引和列索引标签选取数据，使用 .loc() 语义更为清晰
    # df.iloc 常用两个参数，第一个是行索引，第二个是列索引
    # 列的筛选均支持切片，列表与布尔数组
    iris_loc = iris.loc[0:15, ["Id", "SepalWidthCm", "PetalLengthCm"]]
    iris_iloc = iris.iloc[0:15, [0, 2, 3]]
    if iris_loc is iris_iloc:
        print("\n same result using loc and iloc at: \n", iris_loc)
    else:
        # warning: loc 与 iloc 在解析行时有区别，loc 包含结束索引，iloc 不包含结束索引
        # highlight: 因此，此处的结果是不同的
        print("\n different result at: \n", iris_loc, "\n", iris_iloc)
    # 使用布尔数组筛选满足条件的行，只要类型支持相应比较运算即可
    iris_versicolor = iris[iris["Species"] == "Iris-versicolor"].copy()
    # 对于颜色为 versicolor 的 iris 个体，我们可以做一个统计行为计算它的花瓣面积
    iris_versicolor["PetalArea"] = (
        iris_versicolor["PetalLengthCm"] * iris_versicolor["PetalWidthCm"]
    )
    print(iris_versicolor.head())


if __name__ == "__main__":
    main()
