---
title: World Models
tags:
  - synthesis
  - world-model
  - video-generation
description: 世界模型综述：对比视频生成派（Genie、SVD、PlayerOne、WorldJourney）与表示预测派（JEPA、DINO-world），并讨论其价值与局限。
---

最近我在阅读一系列和World Model有关的论文，比如SVD、Genie、DINO-world、JEPA、Worldjourney、Playerone。

其中Genie、SVD、Playerone、Worldjourney算是一派。
他们本身是视频生成模型，只是被发现具有一定的物理智能/世界理解。这四个工作中，Genie采用自回归范式逐帧生成，并且对齐了一个latent action空间，并实验了机器人运动数据，似乎确实建模了一个dynamics model。SVD属于训练策略的优化，证明了训练流程和数据的重要性。Playerone算是裁缝工作，手工设计了合理的模块，并取得了很好的效果。Worldjourney感觉是一个炒作工作，除了开创了所谓perpetual world generation之外，本质上是一个工作流。

JEPA、DINO-world属于一类工作。他们的意图是从latent获得可预测的表示。其中JEPA是encoder-predictor联合训练，不过LeCun说JEPA不是一个生成架构，而是以获得latent repr，所以这里的predictor更多的只是一个提供监督信号的工具；DINO-world 使用预训练的DINOv2作为encoder，转而对predictor做大规模预训练，获得了更好的dense forcast能力。

我认为Genie、JEPA、DINO-world，非常符合LeCun在A Path to Autonomous Machine Learning中对世界模型的假设。但是我觉得比起它们想要证明的价值和意义，它们的范式其实比较简单，即SSL，自监督学习，试图从无标签数据中学习统计规律，也就是表示。

我想这其实并不能代表智能，因为这只能是可能规律的表示，而不能是不可能规律的表示。如果引入一个不存在的场景，这些世界模型还会有语义吗，比如如果在1500年训练一个这样的模型，当输入一个飞机起飞的视频的时候，它会怎么理解？它能建立类比的理解、或者因果的理解吗？另外，这些模型都只是对一个静态世界的建模，换句话说就是高度依赖预训练。

总而言之，所谓世界模型也许只是具身智能炒作的一个概念，用以验证Sim2Real的可行性。所以这些世界模型的意义是什么呢？我想可能也许还是有一点价值，也许是希望通过视觉来获得关于物理体验的一点先验，正如JEPA、DINO-world做的那样，获得一个抽象的表示空间就不必再从RGB的像素再适应到下游任务，并且比视觉特征稳定很多。
