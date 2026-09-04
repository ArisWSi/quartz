---
title: Metrics
tags:
  - concept
  - metric
---

## FID

FID（Fréchet Inception Distance，Heusel et al. 2017）是当前生成模型事实标准的评估指标。它把真实与生成图像分别送进预训练 Inception-v3 取出 2048 维特征，假设两组特征各自服从 Gaussian，再计算两个 Gaussian 之间的 Fréchet distance（即 Wasserstein-2）：

$$
d^2
=
\left\|
\mu_r-\mu_g
\right\|^2
+
\mathrm{tr}
\left(
\Sigma_r
+
\Sigma_g
-
2(\Sigma_r\Sigma_g)^{1/2}
\right)
$$

越低代表两组特征越接近。相比只评估生成样本的 IS，FID 引入了真实分布作参照，能反映 mode dropping 等分布偏移，因此被 [[DDPM]] 及后续几乎所有生成论文采用。

局限：特征非 Gaussian 时假设不成立；协方差估计需要大量样本（惯例 10k–50k 张），小样本下估计有偏，跨论文比较时应固定样本数、分辨率与特征预处理。

## KID

KID（Kernel Inception Distance，Bińkowski et al. 2018）解决的是 FID 的小样本偏差问题。同样取 Inception 特征，但改用三次多项式核的 squared MMD 来衡量真实与生成特征分布的距离：

- 是无偏估计，且不假设特征服从 Gaussian；
- 显式给出估计的方差，可以构造显著性检验来比较两个模型；
- 样本量在千级时仍比 FID 可靠（FID 的 plug-in 估计在小样本下偏差明显）。

代价是结果对核与特征的选择更敏感。IS/FID/KID 都以预训练分类网络特征为基础，常并列报告互相印证；而"单张图 vs 参考图"的成对感知指标见下文的 LPIPS。

## IS

IS（Inception Score，Salimans et al. 2016）是最早广泛使用的生成指标。把生成图像送入 ImageNet 上预训练的 Inception-v3，得到类别分布 $p(y\mid x)$，分数定义为：

$$
\exp
\left(
\mathbb E_x
\left[
D_{KL}(p(y\mid x) \| p(y))
\right]
\right)
$$

质量与多样性两项分别起作用：单张图分类越自信、整体类别边缘分布越均匀，得分越高。但它**不接触真实数据**，只能说明"生成的图像像某个类"，无法说明与目标分布贴合；模型只生成少数类别的样本时依然可能拿高分（对 mode dropping 不敏感），对分类器高置信但怪异的图像也会给高分。如今一般只作辅助参考，与 FID/KID 配合使用。

## LPIPS

LPIPS（Learned Perceptual Image Patch Similarity，Zhang et al. 2018）衡量**两张图像**之间的感知距离，是重建/编辑类任务（超分、压缩、img2img）的标准指标。做法：把两图分别送进预训练网络（默认 VGG，也可 AlexNet 等），在多个层上取特征、按通道归一化，再对逐通道加权的 L2 距离求和；权重在一个小型人类成对比较（2AFC）数据集上学出来，因此它刻意模仿人的感知，而不是像 L2 那样惩罚像素差异。值越低越相似（常见取值范围约在 $[0,1]$，但不严格）。

定位与 FID/KID 不同：它需要**配对的参考图**，回答"两张图差多少"，无法判断"生成分布是否贴合目标分布"；结果对特征骨干与预处理敏感。它既可以当评估指标，也可以直接当训练损失（perceptual loss）。

## PSNR / SSIM

经典的重建质量指标，与 LPIPS 一样属于成对（reference-based）评估：

- **PSNR**：基于逐像素 MSE 的峰值信噪比，dB 为单位、越高越好：

$$
\mathrm{PSNR}
=
10\log_{10}\frac{\mathrm{MAX}_I^2}{\mathrm{MSE}}
$$

- **SSIM**（Wang et al. 2004）：在局部窗口上分别比较亮度、对比度与结构三项，取值范围 $\le 1$、越高越好：

$$
\mathrm{SSIM}(x,y)
=
\frac{
(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)
}{
(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)
}
$$

多尺度版本为 MS-SSIM。

两者的共同点是计算便宜、完全可复现，但与人眼感知的相关度有限（例如 PSNR 对"感知上无关紧要"的噪声也一视同仁地惩罚），如今主要留在压缩与率失真分析的语境里——[[LDM]] 的感知压缩率失真实验正是 PSNR/SSIM/LPIPS 三者并列报告；生成分布质量评估不适用（无配对参考）。

## CLIP score

CLIP score 衡量 text-to-image 结果与 prompt 的**语义对齐**：用 CLIP 分别编码图像与文本后取余弦相似度（编码器本身见 [[Pretrained Model]] 的 CLIP 条目）。优点是无人工标注即可批量打分，也因此被不少扩散模型微调方法当作优化目标或 reward。缺点：CLIP 对底层图像质量不敏感（语义对得上不代表好看或构图正确），对措辞变化敏感，抓不住计数、属性绑定等细粒度错误；更接近人类偏好的自动评测（HPS、ImageReward 等）在此基础上用偏好数据训练得到。

## Precision & Recall

FID 这类指标把 fidelity 与 diversity 揉进一个数字，变差时无法区分"质量下降"还是"mode 丢失"。Improved Precision & Recall（Kynkäänniemi et al. 2019）把两者拆开：在特征空间里用每个样本的 $k$-NN 球估计"真实流形"与"生成流形"，

- **precision** = 生成样本中落在真实流形内的比例（fidelity）；
- **recall** = 真实样本中被生成流形覆盖的比例（diversity / mode coverage）。

特征常用 Inception 或 CLIP 的嵌入。局限：$k$-NN 球的流形估计对 $k$ 的取值与离群点敏感，样本量小时两个数都不可靠。

## Density & Coverage

Density & Coverage（Naeem et al. 2020）针对 precision/recall 的流形估计问题做了修正：

- **density**：每个生成样本统计它被多少个真实样本的 $k$-NN 球覆盖（归一化加权、可以大于 1），对离群点比二值判断更鲁棒；
- **coverage**：统计真实样本中有多少比例的 $k$-NN 球内至少含一个生成样本（$[0,1]$），衡量生成分布覆盖了真实流形的多少。

两者结合能更可靠地区分 fidelity 与 coverage 问题；注意 coverage 无法感知 mode 内部的密度——生成分布若在每个 mode 上只落一个点也能拿满分，需要配合 density 一起看。


