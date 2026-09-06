# Datasets

常用数据集的速查与笔记（论文笔记中提到数据时的展开阅读处）。

## LAION-5B

LAION-5B（Schuhmann et al. 2022，NeurIPS Datasets & Benchmarks；[arXiv:2210.08402](https://arxiv.org/abs/2210.08402)）是从 Common Crawl 网页抓取并经自动过滤得到的图文对数据集，共 **5.85B** 对（其中英语 2.32B），是 OpenCLIP、[[Pretrained Model|SD 1.x/2.x]] 等开源多模态模型的训练基座。

要点：

- 网络数据**靠过滤管线而非人工标注**保质量：CLIP 图文相似度、分辨率、NSFW / watermark / toxic 检测、语言识别、近似去重；
- 附带 CLIP embedding 近邻检索索引，便于按相似度采样或构建子集；
- 社区按美学打分进一步切出 **LAION-Aesthetics** 子集（top 10% / 5% / 1% 等），用于高质量微调与过滤。

与 [[SVD]] 的关系：两条可对照的线索——(1) 数据血缘：SVD 的基座继承自 SD 2.1（在 LAION-5B 上训练），数据思想一脉相承；(2) 方法论：SVD 把"美学质量"作为过滤信号并配 Elo 阈值选取，与 LAION-Aesthetics"挑高美学子集"的思路相通。

<!-- 待扩展：WebVid、EgoExo-4D 等其他常用数据集 -->
