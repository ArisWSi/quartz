---
title: WSDT
aliases:
  - Wasserstein Style Distribution Analysis and Transform for Stylized Image Generation
tags:
  - paper
  - diffusion
  - style-transfer
  - 2025
description: 分析注意力特征的风格敏感性，并在去噪时变换部分通道的特征分布。
---

## Overview

[Wasserstein Style Distribution Analysis and Transform for Stylized Image Generation，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_Wasserstein_Style_Distribution_Analysis_and_Transform_for_Stylized_Image_Generation_ICCV_2025_paper.pdf)

已有免训练风格化方法常在 attention 里替换参考图特征，但为什么选那些特征、怎样避免把参考图内容也带过去，解释并不充分。WSDT 先分析特征对风格变化的敏感度，再在生成过程中做分布变换。

## Style Sensitivity

作者分别比较“内容相同、风格改变”和“风格相同、内容类别改变”时，attention 内部特征分布变化有多大。观察的特征包括输入 $F$、注意力图 $M$、value $V$ 和输出 $O$。他们用高斯近似与 Wasserstein 距离度量变化，并比较风格变化与类别变化造成的距离。

结果是 self-attention 中的 $F,V,O$ 对风格更敏感，而注意力图和 cross-attention 特征更多地响应内容变化。这里的“敏感”来自这套对照实验，不等于已把风格与内容彻底分离。

## Wasserstein Style Distribution Transform

参考风格图先经过前向加噪，得到各时间步的参考特征。生成目标图时，从文本 prompt 和随机噪声出发，在每步去噪的 self-attention 中提取目标与参考双方的 $F,V,O$。对选中的通道，把目标特征的均值、标准差对齐到参考特征；在逐通道高斯近似下，这个变换与 AdaIN 的形式相同。

有个容易读反的地方：前一节找的是**风格改变时分布差异大的特征类型**；实际生成时，作者却选**目标图与参考图之间距离小于阈值**的通道来变换。后一个阈值控制注入范围，设得越大，通常风格越强，也越容易带入参考图内容。这两个距离比较的对象不同。

## Experiments & Discussions

论文在 SD 系列模型上做风格生成，展示了文本内容、参考风格与阈值之间的权衡。它不训练新参数，但每步要处理参考图分支并修改模型内部特征，推理成本不会凭空消失。

我对“Wasserstein”这个名字会保留一点警惕：论文在特征通道上采用高斯和对角协方差假设，得到可计算的分布对齐；这并不说明整张生成图像已经完成最优运输。若参考图有很强的物体形状，最好单独测量内容泄漏，而不只看风格相似度。

补充：[论文附录](https://www.openaccess.thecvf.com/content/ICCV2025/supplemental/Yu_Wasserstein_Style_Distribution_ICCV_2025_supplemental.pdf)

相关：[[LDM]] · [[StableFlow]]
