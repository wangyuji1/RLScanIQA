# RL-ScanIQA 中文说明

> **论文已被 CVPR 2026 正式接收并收入会议论文集，页码为 37401–37412，共 12 页。**
>
> [CVF 官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RL-ScanIQA_Reinforcement-Learned_Scanpaths_for_Blind_360deg_Image_Quality_Assessment_CVPR_2026_paper.html) · [arXiv](https://arxiv.org/abs/2603.14297)

RL-ScanIQA 将无参考 360° 图像质量评价重新表述为主动感知问题：模型不再单独模仿人类注视轨迹，而是通过 PPO 从质量预测反馈中学习与 IQA 任务相关的视点选择策略。

## 仓库内容

- 360° ERP 图像的候选视口离散与投影；
- GRU 扫描路径策略和价值网络；
- PPO、GAE 与 rollout buffer；
- 逐步探索、扫描路径多样性和任务对齐奖励；
- 弱、中、强失真增强及排序一致性损失；
- 可运行的参考训练脚本、CPU 测试与论文结果汇总。

## 安装

```bash
git clone https://github.com/wangyuji1/RLScanIQA.git
cd RLScanIQA
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## 重要说明

- 仓库不包含 CVIQD、OIQA、JUFE 原始数据、私人审稿文件或个人资料。
- README 和 [RESULTS.md](RESULTS.md) 中的指标来自论文及作者补充材料，并非此代码快照重新跑出的结果。
- 精确复现实验还需要合法获取数据集、作者使用的划分文件、冻结的 DINOv2 设置以及训练 checkpoint。
- 当前尚未选择开源许可证；在许可证确认前，请联系作者获得复用或再分发许可。

详细配置请阅读 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
