# PGDNet

包含模型、训练验证代码、合成数据（58.5 MB）及 seed44 权重（113 MB，验证集选中第85轮）。单种子包不含基线，不能复现论文三种子均值。

安装 Git LFS 后克隆仓库，在本目录使用 IES 环境运行：

```bash
git lfs pull
conda activate IES
pip install -r requirements.txt
python run.py verify
python run.py train --epochs 100
python run.py validate
python run.py evaluate
```

默认使用 CUDA。验证、评估默认加载随包权重，评估覆盖500个合成测试日；输出保存至 `runs/`。

公开数据OPSD：[OPSD 官方下载页](https://data.open-power-system-data.org/time_series/2020-10-06/)。
