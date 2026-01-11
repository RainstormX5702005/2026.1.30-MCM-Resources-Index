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

之后，您就需要使用 **uv** 即可在终端执行代码，注意：

- 请确保同时提交 `pyproject.toml` 与 `uv.lock` 以保持同步
- 请不要把 `.venv` 等本机环境相关的文件上传到本库

## 核心算法

算法均按照主题分类，基本采用 Python 的 `pandas`, `numpy` 与 `scikit-learn` 三个库分别进行数据统计分析与预处理与相应 ML 模型的训练，请读者自行根据需求跳转到如下章节阅读：

### 分类算法

请阅读该文档：[classification.md](docs\classification.md)

### 聚类算法

请阅读该文档：[regression.md](docs\regression.md)

## 参考代码

TODO: 正在施工