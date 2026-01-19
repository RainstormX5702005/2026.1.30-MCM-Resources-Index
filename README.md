# 老年皮划艇组 MCM 2026.1.30 - 2026.2.3 9:00 AM 训练指南

本仓库致力于 **老年皮划艇** 队伍的算法积累以及训练代码的存储。

## 目录

以下是本库的相关目录，请读者自行根据需求查阅：

- **[使用说明](#使用说明)**
- **[核心算法](#核心算法)**
- **[参考代码](#参考代码)**

## 使用说明

如果您需要参考这个库学习笔者当时的学习思路，请按照如下方式部署：

1. 克隆仓库：

如果您使用的是 HTTPS 协议克隆的话，请使用

```powershell
git clone https://github.com/RainstormX5702005/2026.1.30-MCM-Resources-Index.git
```

2. 安装相关依赖：

为了便于代码的管理，建议您使用 **uv** 作为 python 环境的管理工具，在 clone 以后输入

```powershell
cd your\project\directory
uv sync
```

之后，您就需要使用 **uv** 即可在终端执行代码，比如您可以这么运行代码，更多指令请查阅相关文档：

```powershell
uv run .\path\to\your\file.py
```

注意：

- 请确保同时提交 `pyproject.toml` 与 `uv.lock` 
- 请不要把 `.venv` 等本机环境相关的文件上传到本库，否则本库会拒绝您的提交

本人在此介绍各个文件夹的功能，您在提交代码的时候务必把代码装入相应的文件夹中，否则此次提交会被拒绝：

- 📚docs: 用于存放与比赛相关的文档，可以存储任意与比赛模型或是代码相关的内容
- ⚙src: 用于存放本次 2026MCM 的相关代码
    - 🌳template: 用于存放预先编译好的相关模型或是数据可视化代码片段
    - 📷assets: 用于存放最终训练出来的图片
    - 🔧configs: 用于存放代码所需的工作区配置信息
- ⭐examples: 用于存放所有算法相关的训练内容信息，请您根据已有的分类将您的训练代码存放进去，或是建立新的文件夹避免混淆

其余根目录文件非必要请勿删除，防止因环境破坏而导致编译失败！

## 核心算法

算法均按照主题分类，基本采用 Python 的 `pandas`, `numpy` 与 `scikit-learn` 三个库分别进行数据统计分析与预处理与相应 ML 模型的训练，请读者自行根据需求跳转到如下章节阅读：

### 数据预处理算法

数据预处理算法的使用好坏决定了最终模型训练是否准确，好的数据预处理是建立一个合理的预测模型的前提，希望您可以重视这部分内容的学习。

请阅读该文档：[preprocessing.md](docs/preprocessing.md)

### 分类算法

请阅读该文档：[classification.md](docs/classification.md)

### 聚类算法

请阅读该文档：[regression.md](docs/regression.md)

## 参考代码

TODO: 正在施工

## LICENSE

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.