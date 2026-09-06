---
title: Pretrained Model
tags:
  - concept
  - model
  - pretrained
---

## CLIP

CLIP（Radford et al. 2021）从 4 亿图文对（WIT）用对比学习训练双塔：同一 batch 内最大化匹配图文对的余弦相似度、压低不匹配对，最终把图像与文本映射进同一个 embedding 空间。由此直接获得的能力：zero-shot 分类（把类别名拼进 "a photo of a {label}" 模板后做检索）与图文检索/匹配。

在生成模型中的角色有三类：

- **冻结的文本编码器**：为条件生成提供语义。SD 1.x 用 CLIP ViT-L/14 的逐 token 特征喂 cross-attention（上下文仅 77 token）；SD 2.x 换成 OpenCLIP ViT-H/14（LAION-2B 上训练，1024 维）。
- **图像编码器**：IP-Adapter 等图像条件方法的特征来源。
- **评估器**：CLIP score 衡量生成图与文本的语义对齐程度。

SD 3.x 仍保留两个 CLIP 编码器参与条件，FLUX 则只用 CLIP-L 的 pooled 向量提供全局条件，长 prompt 的细节语义交给 T5。

## SD 1.x 2.x

Stable Diffusion 是把 [[LDM]] 放到 [[Datasets|LAION-5B]] 与大规模算力上训练的产物（CompVis / Runway / Stability，2022），开源 text-to-image 自此成为可能。

1.x 的配置几乎就是 [[LDM]] 原样放大：VAE 把图像压到 8× 的 latent，约 860M 参数的 U-Net 在其中去噪（[[Architecture]]），冻结的 CLIP ViT-L/14（77 token）经 cross-attention 注入，配合 CFG（训练时 drop 10% 的文本条件）；按 512×512 训练。1.4/1.5 只是数据与训练细节的迭代（1.5 总参数量约 983M）。

2.x 则是方向性的改动：过滤训练数据（去 NSFW 与低美学质量样本）后从头训练；文本编码器换成 OpenCLIP ViT-H/14；预测目标从 $\epsilon$-prediction 改为 v-prediction；2.1 提供原生 768×768 的 checkpoint；同期附带 depth2img、inpainting、4× upscaler 等配套模型。

对研究而言 SD 的意义在于它是一个**开放、可微调的预训练骨干**：[[ControlNet]]、LoRA/DreamBooth 等可控生成与个性化方法都建立在其权重之上，很多能力差异（例如模型"风格化"倾向）其实由 text encoder 与训练数据的取舍决定。

## SD 3.x

Stability 2024 年的系列，在架构上彻底转向。主网络从 U-Net 换成 **MMDiT**（multimodal diffusion transformer）：patchify 后的图像 token 与文本 token 拼成同一序列做 joint attention，两种模态各用一套 QKV 权重，信息可以双向流动（文本 attend 图像、图像 attend 文本），大幅改善复杂 prompt 理解与拼字能力。训练范式换成 [[RectifiedFlow]]，并把噪声采样时间做 logit-normal 偏置，让训练集中于感知上更相关的中间噪声尺度。

条件侧用三个冻结文本编码器：CLIP ViT-L/14 与 OpenCLIP ViT-bigG（各 77 token，输出沿特征维拼接），加上 T5-XXL（encoder，逐 token 输出，按序列维拼接）；两个 CLIP 的 pooled 向量拼接后作为全局条件。论文同时展示了 loss 随参数量（800M → 8B）的平滑缩放趋势。

发布线：3.0 于 2024.6 开源了 Medium（2B）权重；2024.10 的 3.5 把 8B 的 Large 与蒸馏版 Large Turbo 一并开源，并换上更宽松的 Community License。3.5 架构上引入 MMDiT-X 改进——transformer 前十几层加入额外的 self-attention 块、采用 QK-norm 稳定训练、位置编码空间扩到 384×384 latent 并配合 256→1440 像素的渐进式多分辨率训练，显著改善非方形/多分辨率出图；推理时常配 Skip-Layer Guidance 提升结构一致性。

## FLUX

FLUX.1（Black Forest Labs, 2024.8）出自参与创建 Latent Diffusion / Stable Diffusion 系列的研究者成立的团队，把 SD3 的路线推到 12B 参数的规模：rectified flow transformer，block 采用"多模态（MMDiT 式 joint attention）+ parallel attention"的混合设计，并引入 RoPE 旋转位置编码与 2×2 latent patch packing（token 数再降 4 倍），在质量与硬件效率上都比 SD3 更进一步。

文本条件拆成两路：T5-XXL 提供长 prompt 的逐 token 语义（最长 512 token），CLIP-L 只取 pooled 向量作全局引导，不再逐 token 参与（与 SD3 的用法不同，见上 CLIP 一节）。家族三档：FLUX.1 [pro]（API）、[dev]（开源权重，guidance-distilled，非商用许可）、[schnell]（Apache 2.0，蒸馏模型，官方示例 4 步、无需 CFG）。生态与 SD 类似，围绕它形成了大量 LoRA 与 adapter 微调资源。
