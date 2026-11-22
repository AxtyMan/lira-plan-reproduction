# LIRA-Proto 少标注遥感场景分类

[实验结果与文献对比](实验结果与文献对比.md) · [项目文件、公式、十种方法及指标说明](项目文件与复现说明.md)

在此目录执行 ` .\python-env.cmd -m unittest discover -s tests -v ` 可验证新环境和算法。完整复跑命令见说明文档。矩阵运行使用原始 RGB 图像编码得到的冻结特征和预先登记的配置。

## 从 GitHub 克隆后运行

使用 Python 3.12 创建 `.venv`，按本机设备选择 `requirements-cpu.txt` 或 `requirements-cuda.txt` 安装依赖。公开数据、权重、上游实现的来源与 SHA-256 见 [资源清单](reports/sources.json)。将资源放到清单指定的位置，再按[复现说明](项目文件与复现说明.md)中的 `prepare` 和 `extract` 命令重建本机路径与特征缓存。仓库提供冻结配置、数据划分、结果汇总、审计记录与图表。虚拟环境、离线安装包及全量二进制实验产物由本地工作目录保存。
