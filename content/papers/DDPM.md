---
title: DDPM
aliases:
  - Denoising Diffusion Probabilistic Models
  - Ho et al. 2020
tags:
  - paper
  - diffusion
  - generative-model
status: draft
---

# DDPM

> [!quote]
> A face... a fleet... a war... a man... a thought... a trick... a trick to break the walls of Troy... and burn it screaming to the ground!
>
> —— *Odyssey, 2026*

## External Resources

- [概率论推导参考](https://zhuanlan.zhihu.com/p/525106459)：这位作者对于其中的概率论推导已经十分详尽。

## Overview

简单来说，DDPM 定义了一组 latent variable $x_{0:T}$，并规定了从

$$
x_0 \rightarrow \cdots \rightarrow x_T \sim \mathcal N(0,\mathbf I)
$$

的马尔可夫过程，希望建模其反过程，从高斯噪声生成图像。

优化目标是最大似然：

$$
p_\theta(x_0)
$$

## Forward Process

> [!abstract] Core Idea
> 前向过程被人为设计。

前向马尔可夫过程：

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
$$

其中：

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}x_{t-1},
\beta_t\mathbf I
\right)
$$

令：

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s
$$

则：

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_t\mathbf I
\right)
$$

由重参数化：

$$
x_t
=
\sqrt{\alpha_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\mathbf I)
$$

递推可以得到：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

因此可以直接从 $x_0$ 采样任意时间步 $x_t$：

$$
q(x_t\mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)\mathbf I
\right)
$$

## Reverse Process

> [!abstract] Core Idea
> 后向过程被建模为参数化的 Gaussian Markov chain。

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t)
$$

其中：

$$
p(x_T)=\mathcal N(0,\mathbf I)
$$

反向转移定义为：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
$$

> [!important] Goal
> 希望学习：
>
> $$
> p_\theta(x_{t-1}\mid x_t)
> \approx
> q(x_{t-1}\mid x_t)
> $$

## Variational Inference

### ELBO

目标是最大化数据的对数似然：

$$
\log p_\theta(x_0)
$$

但模型中包含隐变量 $x_{1:T}$：

$$
p_\theta(x_0)
=
\int p_\theta(x_{0:T})\,dx_{1:T}
$$

直接计算这个积分很困难。由于前向过程 $q(x_{1:T}\mid x_0)$ 已知，因此将其作为变分分布：

$$
\begin{aligned}
\log p_\theta(x_0)
&=
\log
\int
q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\,dx_{1:T}
\\
&=
\log
\mathbb E_q
\left[
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
\\
&\ge
\mathbb E_q
\left[
\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
\end{aligned}
$$

最后一步来自 Jensen 不等式。

定义 negative ELBO：

$$
\mathcal L
=
\mathbb E_q
\left[
-\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right]
$$

于是：

$$
-\log p_\theta(x_0)
\le
\mathcal L
$$

因此，最小化 $\mathcal L$ 相当于最小化 negative log-likelihood 的一个上界，从而间接优化最大似然目标。

### ELBO Decomposition

前向过程：

$$
q(x_{1:T}\mid x_0)
=
\prod_{t=1}^{T}
q(x_t\mid x_{t-1})
$$

反向过程：

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=1}^{T}
p_\theta(x_{t-1}\mid x_t)
$$

代入 negative ELBO：

$$
\mathcal L
=
\mathbb E_q
\left[
-\log p(x_T)
-\sum_{t=1}^{T}
\log p_\theta(x_{t-1}\mid x_t)
+
\sum_{t=1}^{T}
\log q(x_t\mid x_{t-1})
\right]
$$

由于前向过程满足 Markov 性：

$$
q(x_t\mid x_{t-1},x_0)
=
q(x_t\mid x_{t-1})
$$

利用 Bayes 公式：

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}
$$

因此：

$$
q(x_t\mid x_{t-1})
=
\frac{
q(x_{t-1}\mid x_t,x_0)
q(x_t\mid x_0)
}{
q(x_{t-1}\mid x_0)
}
$$

代回后，对不同时间步的 $q(x_t\mid x_0)$ 项进行 telescoping cancellation，可以得到：

$$
\mathcal L
=
\mathbb E_q
\left[
\underbrace{
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\|p(x_T)
\right)
}_{L_T}
+
\sum_{t=2}^{T}
\underbrace{
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\|
p_\theta(x_{t-1}\mid x_t)
\right)
}_{L_{t-1}}
-
\underbrace{
\log p_\theta(x_0\mid x_1)
}_{L_0}
\right]
$$

其中：

- $L_T$：约束最终的 $q(x_T\mid x_0)$ 接近先验 $\mathcal N(0,I)$；
- $L_{t-1}$：学习每一步反向转移；
- $L_0$：从 $x_1$ 重建最终图像 $x_0$。

由于 $\beta_t$ 固定，$q(x_T\mid x_0)$ 不依赖 $\theta$，因此 $L_T$ 对模型训练而言是常数。

> [!important]
> 真正希望建模的是 $q(x_{t-1}\mid x_t)$，但它涉及未知的数据分布，不能直接解析计算。
>
> 训练时 $x_0$ 是已知样本，因此可以解析计算
>
> $$
> q(x_{t-1}\mid x_t,x_0)
> $$
>
> 并用它来训练 $p_\theta(x_{t-1}\mid x_t)$。

### Analytical Posterior

已知一步前向转移：

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{\alpha_t}x_{t-1},
\beta_t I
\right)
$$

以及：

$$
q(x_{t-1}\mid x_0)
=
\mathcal N
\left(
x_{t-1};
\sqrt{\bar\alpha_{t-1}}x_0,
(1-\bar\alpha_{t-1})I
\right)
$$

由 Bayes：

$$
q(x_{t-1}\mid x_t,x_0)
=
\frac{
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0)
}{
q(x_t\mid x_0)
}
$$

因为分母 $q(x_t\mid x_0)$ 与 $x_{t-1}$ 无关，因此：

$$
q(x_{t-1}\mid x_t,x_0)
\propto
q(x_t\mid x_{t-1})
q(x_{t-1}\mid x_0)
$$

将两个 Gaussian 写成指数形式：

$$
q(x_{t-1}\mid x_t,x_0)
\propto
\exp
\left[
-\frac12
\left(
\frac{
\|x_t-\sqrt{\alpha_t}x_{t-1}\|^2
}{\beta_t}
+
\frac{
\|x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0\|^2
}{
1-\bar\alpha_{t-1}
}
\right)
\right]
$$

展开后，只保留与 $x_{t-1}$ 有关的项：

$$
-\frac12
\left[
A\|x_{t-1}\|^2
-
2b^Tx_{t-1}
+
C
\right]
$$

其中：

$$
A
=
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar\alpha_{t-1}}
$$

$$
b
=
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+
\frac{\sqrt{\bar\alpha_{t-1}}}
{1-\bar\alpha_{t-1}}x_0
$$

利用配平方：

$$
A\|x\|^2-2b^Tx
=
A
\left\|
x-A^{-1}b
\right\|^2
+
C
$$

因此该分布仍然是 Gaussian，其协方差为 $A^{-1}I$，均值为 $A^{-1}b$。

由：

$$
\beta_t=1-\alpha_t
$$

以及：

$$
\bar\alpha_t
=
\alpha_t\bar\alpha_{t-1}
$$

可得：

$$
\begin{aligned}
A
&=
\frac{\alpha_t}{\beta_t}
+
\frac{1}{1-\bar\alpha_{t-1}}
\\
&=
\frac{
\alpha_t(1-\bar\alpha_{t-1})+\beta_t
}{
\beta_t(1-\bar\alpha_{t-1})
}
\\
&=
\frac{
1-\bar\alpha_t
}{
\beta_t(1-\bar\alpha_{t-1})
}
\end{aligned}
$$

因此后验方差为：

$$
\boxed{
\tilde\beta_t
=
A^{-1}
=
\frac{
1-\bar\alpha_{t-1}
}{
1-\bar\alpha_t
}
\beta_t
}
$$

后验均值为：

$$
\tilde\mu_t
=
A^{-1}b
$$

即：

$$
\boxed{
\tilde\mu_t(x_t,x_0)
=
\frac{
\sqrt{\bar\alpha_{t-1}}\beta_t
}{
1-\bar\alpha_t
}x_0
+
\frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})
}{
1-\bar\alpha_t
}x_t
}
$$

最终：

$$
\boxed{
q(x_{t-1}\mid x_t,x_0)
=
\mathcal N
\left(
x_{t-1};
\tilde\mu_t(x_t,x_0),
\tilde\beta_tI
\right)
}
$$

> [!summary]
> 前向过程是 linear Gaussian Markov chain，因此给定 $x_0$ 和 $x_t$ 后，一步反向 posterior $q(x_{t-1}\mid x_t,x_0)$ 仍然是可以解析求出的 Gaussian。

### Fixed Variance

反向模型为：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
$$

DDPM 不学习 reverse variance，而将其固定为只与时间步有关的常数：

$$
\Sigma_\theta(x_t,t)
=
\sigma_t^2I
$$

论文考虑两种选择：

$$
\sigma_t^2=\beta_t
\qquad\text{or}\qquad
\sigma_t^2=\tilde\beta_t
$$

因此：

$$
L_{t-1}
=
D_{\mathrm{KL}}
\left(
\mathcal N(\tilde\mu_t,\tilde\beta_tI)
\|
\mathcal N(\mu_\theta,\sigma_t^2I)
\right)
$$

一般 Gaussian 之间的 KL divergence 为：

$$
D_{\mathrm{KL}}
\left(
\mathcal N(\mu_1,\Sigma_1)
\|
\mathcal N(\mu_2,\Sigma_2)
\right)
=
\frac12
\left[
\log\frac{|\Sigma_2|}{|\Sigma_1|}
-d
+
\operatorname{tr}
\left(
\Sigma_2^{-1}\Sigma_1
\right)
+
(\mu_2-\mu_1)^T
\Sigma_2^{-1}
(\mu_2-\mu_1)
\right]
$$

这里：

$$
\Sigma_1=\tilde\beta_tI,
\qquad
\Sigma_2=\sigma_t^2I
$$

代入得到：

$$
L_{t-1}
=
\mathbb E_q
\left[
\frac{1}{2\sigma_t^2}
\left\|
\tilde\mu_t(x_t,x_0)
-
\mu_\theta(x_t,t)
\right\|^2
\right]
+
C
$$

其中 $C$ 只包含 $\tilde\beta_t$ 和 $\sigma_t^2$，均不依赖 $\theta$。

因此固定 reverse variance 后，Gaussian KL 对模型参数的优化只剩下：

$$
\boxed{
\mu_\theta(x_t,t)
\approx
\tilde\mu_t(x_t,x_0)
}
$$

即 **mean matching**。

### $\epsilon$-Parameterization

由前向采样公式：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon
$$

可以反解：

$$
x_0
=
\frac{
x_t-\sqrt{1-\bar\alpha_t}\epsilon
}{
\sqrt{\bar\alpha_t}
}
$$

将其代入后验均值：

$$
\tilde\mu_t
=
\frac{
\sqrt{\bar\alpha_{t-1}}\beta_t
}{
1-\bar\alpha_t
}x_0
+
\frac{
\sqrt{\alpha_t}(1-\bar\alpha_{t-1})
}{
1-\bar\alpha_t
}x_t
$$

整理得到：

$$
\boxed{
\tilde\mu_t
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon
\right)
}
$$

因此可以不直接预测 $\tilde\mu_t$，而将模型参数化为：

$$
\boxed{
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}
{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right)
}
$$

也就是让神经网络预测构造 $x_t$ 时加入的噪声 $\epsilon$。

两者相减：

$$
\tilde\mu_t-\mu_\theta
=
\frac{\beta_t}
{\sqrt{\alpha_t(1-\bar\alpha_t)}}
\left(
\epsilon_\theta-\epsilon
\right)
$$

因此：

$$
\left\|
\tilde\mu_t-\mu_\theta
\right\|^2
=
\frac{\beta_t^2}
{\alpha_t(1-\bar\alpha_t)}
\left\|
\epsilon-\epsilon_\theta
\right\|^2
$$

代回 $L_{t-1}$：

$$
\boxed{
L_{t-1}
=
\mathbb E
\left[
\frac{
\beta_t^2
}{
2\sigma_t^2
\alpha_t(1-\bar\alpha_t)
}
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
\right]
+
C
}
$$

因此原始 ELBO 的中间项最终转化为一个**带时间步权重的 noise prediction MSE**。

### $L_{\mathrm{simple}}$

定义时间步权重：

$$
w_t
=
\frac{
\beta_t^2
}{
2\sigma_t^2
\alpha_t(1-\bar\alpha_t)
}
$$

则前面的训练目标具有：

$$
w_t
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|^2
$$

的形式。

DDPM 进一步去掉 $w_t$，得到实际使用的简化目标 Eq. 14：

$$
\boxed{
L_{\mathrm{simple}}
=
\mathbb E_{t,x_0,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_\theta
\left(
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
t
\right)
\right\|^2
\right]
}
$$

其中：

$$
t\sim\mathrm{Uniform}\{1,\dots,T\},
\qquad
\epsilon\sim\mathcal N(0,I)
$$

去掉 $w_t$ 后，$L_{\mathrm{simple}}$ 不再严格等价于原始 variational bound，而是重新改变了不同时间步 loss 的相对权重。

在相应参数化和尺度变换下，这一目标与不同噪声尺度上的 [[Score Matching|denoising score matching]] 密切对应。

### Discrete Decoder

前面的 $L_{t-1}$ 对应 $t\ge2$ 的反向转移，而最后一步：

$$
L_0
=
-\log p_\theta(x_0\mid x_1)
$$

需要处理实际图像的离散像素值。

> [!note] $L_0$ 与离散像素
> 图像像素首先从 $\{0,\dots,255\}$ 线性缩放到 $[-1,1]$。
>
> 对 $p_\theta(x_0\mid x_1)$，论文使用 discretized Gaussian likelihood：对每个离散像素值所对应的小区间积分 Gaussian density，从而得到该像素值的概率。
>
> 在生成过程中，$t=1$ 的最后一步不再加入随机噪声，最终输出对应于 $\mu_\theta(x_1,1)$。

> [!summary] Training Objective
> DDPM 的训练推导可以概括为：
>
> $$
> \log p_\theta(x_0)
> \rightarrow
> \mathrm{ELBO}
> \rightarrow
> D_{\mathrm{KL}}
> \left(
> q(x_{t-1}\mid x_t,x_0)
> \|
> p_\theta(x_{t-1}\mid x_t)
> \right)
> $$
>
> $$
> \rightarrow
> \text{mean matching}
> \rightarrow
> \text{weighted noise MSE}
> \rightarrow
> L_{\mathrm{simple}}
> $$

## Experiments & Discussions

1. 比较不同参数化方法对生成质量的影响。
2. 通过编码长度，从感知压缩（率失真分析）、自回归生成、[[Score Matching]] 等角度理解 diffusion 这一脱胎于 [[Langevin Dynamics]] 的方法（更 CV）。
3. 插值表现。

## Related Notes

- [[ELBO]]
- [[Score Matching]]
- [[Langevin Dynamics]]
