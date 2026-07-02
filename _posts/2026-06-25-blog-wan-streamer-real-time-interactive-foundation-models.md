---
title: 'Wan-Streamer 深读：端到端实时音视频全双工模型到底解决了什么'
date: 2026-06-25
permalink: /posts/2026-06-25-blog-wan-streamer-real-time-interactive-foundation-models/
tags:
  - wan-streamer
  - multimodal
  - streaming
  - audio-visual
  - video-generation
paperurl: https://arxiv.org/abs/2606.25041
projecturl: https://wan-streamer.com/
citation: 'Wan Team, Alibaba Group. Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models. arXiv:2606.25041, 2026.'
---

# Wan-Streamer 深读：端到端实时音视频全双工模型到底解决了什么

> 论文：[Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models](https://arxiv.org/abs/2606.25041)
> 作者：Wan Team, Alibaba Group；appendix 列出 Core Contributors: Lianghua Huang, Zhifan Wu, Wei Wang, Yupeng Shi, Mengyang Feng, Junjie He, Chenwei Xie, Yu Liu, Jingren Zhou
> 时间 / 版本：arXiv v1, submitted 2026-06-23
> 类别：Computer Vision and Pattern Recognition / Artificial Intelligence / Graphics / Sound
> 链接：[Paper](https://arxiv.org/abs/2606.25041) / [PDF](https://arxiv.org/pdf/2606.25041) / [Project Page](https://wan-streamer.com/)
> 本文基于 arXiv TeX source、PDF、项目页 HTML 和官方 demo poster 阅读；检索日期：2026-06-25。

---

## 开篇点评：这篇论文到底解决了什么问题

Wan-Streamer 要解决的不是“生成一个更像真人的 avatar”这么窄的问题。它瞄准的是实时全双工音视频交互：用户说话、停顿、看向屏幕、打断时，agent 不能只是等 ASR 结束、让 LLM 想完、TTS 说完、再让 avatar renderer 对嘴型。它应该持续看、持续听、边说边感知、被打断时能调整，同时保持视觉存在感。

论文的判断很直接：现有系统大多是级联管线，常见形式是 VAD / ASR / LLM / TTS / audio-driven animation / video rendering。这样的系统可以工程上跑起来，但模块边界会带来等待时间，ASR 错误会传给语言模型，TTS 和 avatar 同步需要事后修补，turn-taking 也常靠规则控制。Wan-Streamer 想把这些拆散的能力放进一个统一因果流里学习。

它的核心主张是：streamability 不是 serving optimization，而是 modeling constraint。也就是说，实时交互不能先做一个离线多模态模型，再靠缓存、并行、后处理补成流式系统；模型结构、数据表示、训练目标和推理调度从一开始就要因果化。

我的判断是，这篇报告最有价值的不是“200 ms”这个数字本身，而是它把实时交互模型应该如何定义讲清楚了：语言、音频、视频都既是输入也是输出，所有新观测要立刻进入历史，所有生成结果也要写回历史。这个方向非常对，但 v0.1 的证据还偏 proof-of-concept：分辨率只有 192p，没有代码、权重、训练细节，也缺少标准化质量评测和 ablation。

![Wan-Streamer framework](/files/blogs_image/260625-wan-streamer-framework.png)

*图：官方 framework 图。Wan-Streamer 把用户侧和 agent 侧的 text/audio/video 都放进同一个 Transformer，使用 block-causal attention 做增量式流式生成。*

## Paper Card

| 项目 | 信息 |
|---|---|
| Paper | [arXiv:2606.25041](https://arxiv.org/abs/2606.25041) |
| Title | Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models |
| Authors | Wan Team, Alibaba Group |
| Date / Version | Submitted 2026-06-23, v1 |
| Category | cs.CV, cs.AI, cs.GR, cs.SD |
| Project | [wan-streamer.com](https://wan-streamer.com/) |
| Code / Model | not released / not specified |
| 主要能力 | text/audio/video input-output、full-duplex interaction、25 FPS video output、约 200 ms model-side latency、约 550 ms total latency with 350 ms network |
| 复现状态 | 只能阅读 paper source 和项目 demo；缺少 code、checkpoint、training data、model size、完整训练配置和评测脚本 |

## Abstract：论文摘要解读

摘要里有三层信息。

第一层是任务定义。Wan-Streamer 是一个 native-streaming、end-to-end interactive foundation model，面向 real-time、low-latency、full-duplex audio-visual interaction。这里的 full-duplex 很关键：不是用户说完模型才答，也不是 agent 回答时停止感知，而是输入和输出可以在同一个时间轴上重叠。

第二层是架构主张。模型把 language、audio、video 同时作为 input 和 output，放进一个 single Transformer。序列中既有 visual/audio/text input tokens，也有 visual/audio/text output tokens；它们通过 block-causal attention 协调，以支持 incremental streaming。

第三层是系统指标。论文说 streaming unit 最短 160 ms，输出为 25 FPS；model-side response latency 约 200 ms；如果加上 350 ms bidirectional network latency，总交互延迟约 550 ms。这个数字覆盖的是一个更完整的 audio-visual response path，而不是只报 avatar renderer FPS 或 speech first-token。

摘要没有说的是同样重要的东西：模型参数规模、VAE latent 形状、音视频 token rate、训练数据规模、硬件型号、solver step 数、蒸馏细节都没有公开。因此这篇更像一份系统方向报告，而不是可直接复现的技术配方。

## Motivation

论文把实时音视频交互拆成一个现有系统很难同时满足的约束集合：

1. 用户正在说话时，agent 仍要表现出可见的 listening behavior；
2. agent 正在回应时，仍要持续感知用户音视频反馈；
3. 输入语音和视频要立即影响输出语音和动作；
4. 生成音频和生成视频要在 latent 层耦合，而不是输出后再对齐；
5. 每个输出单元都要写回历史，保证身份、场景、说话节奏和意图长期一致。

这套约束解释了为什么“把几个强模型接起来”不够。级联系统可以优化每个模块，但模块间的等待、错误和同步问题会保留。比如 ASR 延迟会限制 LLM 开始推理，TTS 的韵律未必和 avatar motion 同步，视觉反馈也可能无法及时影响正在生成的语音。

Wan-Streamer 的设计赌注是：只要从表示、attention、VAE、decoder、训练数据到 serving 都保持 causal，就可以把 response timing、turn management、visible listening、interruption handling 和 audio-visual generation 当作一个行为学习，而不是系统规则。

## 直观效果：先看它能做什么

项目页给了几段预录的 face-to-face interaction demo，人物、声音和场景不同。页面明确说明这些是 unedited model outputs，不是在线实时体验；v0.1 仍是 192p proof of concept。这个限定很重要：demo 可以说明系统形态，但不能替代量化质量评测。

![Wan-Streamer official demo posters](/files/blogs_image/260625-wan-streamer-demo-posters.jpg)

*图：项目页官方 demo poster 拼图。它展示的是论文希望支持的实时面对面交互形态：不同 persona、不同场景、用户侧视频在角落出现。但这不是实验表，也不能证明长期稳定性或同步精度。*

我会把这组 demo 看成“任务边界证据”，而不是“性能证据”。它说明 Wan-Streamer 不是纯 speech-to-speech，也不是单独的 talking-head renderer，而是试图把用户视频感知、语音交互和 agent 视频输出放到同一个 loop 里。至于画质、身份稳定、A/V sync、人类偏好和 interruption 成功率，论文没有给出可复核指标。

## 方法总览：核心思想和系统结构

Wan-Streamer 有一个贯穿全文的 streaming contract：

- every component must operate causally；
- every newly observed unit must be usable immediately；
- every generated unit must be emitted and committed back into interaction history。

为了满足这三条，模型不是把音频、视频、文本当成独立子系统，而是把它们放进同一条因果序列。第 $k$ 个 streaming unit 中，用户观测记作：

$$
u_k=(u_k^{\mathrm{t}},u_k^{\mathrm{a}},u_k^{\mathrm{v}}),
$$

agent 响应记作：

$$
y_k=(y_k^{\mathrm{t}},y_k^{\mathrm{a}},y_k^{\mathrm{v}}).
$$

模型因式分解为：

$$
p_\theta(y_{1:K}\mid u_{1:K})=
\prod_{k=1}^{K}
p_\theta\left(
y_k^{\mathrm{t}},y_k^{\mathrm{a}},y_k^{\mathrm{v}}
\mid
u_{\le k}^{\mathrm{t}},u_{\le k}^{\mathrm{a}},u_{\le k}^{\mathrm{v}},
y_{<k}^{\mathrm{t}},y_{<k}^{\mathrm{a}},y_{<k}^{\mathrm{v}}
\right).
$$

这个公式的含义很朴素：当前 agent 响应可以看见当前及过去用户输入，也可以看见过去 agent 输出，但不能偷看未来。更关键的是，过去 agent 输出包含自己生成过的音频和视频 latent，因此输出也成为之后交互的状态。

架构上，论文列出以下模块：

| 模块 | 作用 | 论文中是否给出细节 |
|---|---|---|
| Single Transformer | 统一处理 input/output 的 text/audio/video tokens 或 latents | 有概念图，但无参数规模 |
| Block-causal attention | 让 multimodal sequence 按 streaming unit 增量因果更新 | 有描述，但未给 attention mask 细节 |
| Causal audio/video VAEs | 流式 audio/video latent coding | 有模块名，无压缩率和 latent shape |
| Causal audio-visual encoders | 编码当前用户音视频观测 | 有模块名，无结构细节 |
| Causal audio/video decoders | 将 clean latents 渲染为输出音视频 | 有模块名，无结构细节 |
| Conditional flow matching | 联合生成当前 response unit 的 audio/video latents | 有核心公式 |
| Thinker-performer serving | 两 GPU overlap 推理，降低 model-side latency | 有官方图和流程描述 |

## 数据全流程：输入、表示、shape 和语义

论文的数据流可以分成训练表示和推理表示两部分。

### 表示层

| 对象 | 含义 | Shape / Dim |
|---|---|---|
| $u_k^{\mathrm{t}}$ | 用户文本输入 | not specified |
| $u_k^{\mathrm{a}}$ | 用户音频观测 | not specified |
| $u_k^{\mathrm{v}}$ | 用户视频观测 | not specified |
| $y_k^{\mathrm{t}}$ | agent 文本响应 token | not specified |
| $z_0^{\mathrm{a}}$ | agent audio clean latent | not specified |
| $z_0^{\mathrm{v}}$ | agent video clean latent | not specified |
| $c_k$ | clean streaming context，包括已到达用户观测和已提交 agent 响应 | not specified |
| KV slice | thinker 发给 performer 的当前历史状态切片 | not specified |

这里的最大缺口是 shape。论文没有公开 audio/video VAE 的时空压缩率、latent channel 数、每个 160 ms unit 里有多少 audio token 和 video token，也没有给 block-causal attention mask 的具体矩阵形式。读者只能理解机制，不能按论文复现。

### Flow matching 生成

语言响应用 next-token prediction。音频和视频响应在连续 latent 空间里联合生成。对 $m\in\{\mathrm{a},\mathrm{v}\}$，clean target latent 是 $z_0^m$，噪声是 $\epsilon^m\sim\mathcal{N}(0,I)$。flow time $\tau$ 下：

$$
z_\tau^m=(1-\tau)z_0^m+\tau\epsilon^m,\qquad
\frac{\partial z_\tau^m}{\partial\tau}=\epsilon^m-z_0^m.
$$

模型学习 velocity field：

$$
\mathcal{L}_{\mathrm{FM}}^m =
\mathbb{E}_{\epsilon^m}
\left\|
f_\theta(z_\tau^{\mathrm{a}}, z_\tau^{\mathrm{v}}, c_k, \tau)
-
\frac{\partial z_\tau^m}{\partial\tau}
\right\|_2^2.
$$

这段公式有一个容易忽略的设计：当前响应的 audio/video 是 noisy variables，但历史 context 是 clean。去噪完成后，估计出的 clean audio/video latents 会直接追加到历史里，成为之后 streaming unit 的上下文。这样模型不是每段都从零生成，而是在自己已经说过、动过、看过的历史上继续交互。

## Training：监督信号、loss 和优化目标

论文把训练分成三阶段。

第一阶段是 independent-task pretraining。作者从 language model 初始化 unified Transformer，训练多模态接口。理解侧包括 image/audio/video understanding、text dialogue、ASR、TTS、audio dialogue、language-audio-visual supervision；生成侧包括 image generation、audio generation、video generation、joint audio-visual generation。目标是让理解、语言推理和 latent generation 先在一个序列模型里对齐。

第二阶段是 end-to-end interaction training。训练数据里用户 text/audio/video inputs 和 agent text/audio/video outputs 被交错在同一条 causal stream 中。这个阶段才真正对齐目标任务：模型要从当前用户观测更新状态，生成同步文本、语音、视频响应，并把 clean latents 写回历史。

第三阶段是 low-latency distillation。teacher 使用 classifier-free guidance 和更多 flow-matching solver steps；student 吸收 CFG 效果并减少 solver steps。论文还提到 rolling distillation 和 self-forcing：让 student 在连续 streaming units 上滚动生成，再用自己的历史训练，以缓解长程 train-test mismatch。

但训练细节大量缺失：

| 项目 | 公开状态 |
|---|---|
| 数据规模 / 来源 / 过滤 | not specified |
| Transformer 参数量 | not specified |
| Audio/video VAE 结构 | partially specified as causal, details not specified |
| optimizer / LR / batch / schedule | not specified |
| solver steps | not specified |
| distillation loss 权重 | not specified |
| hardware / training budget | not specified |

我的判断是，训练路线可信但不可复现。它告诉我们应该怎么组织一个 end-to-end streaming agent 的训练阶段，但没有给出足够细节让社区验证同等效果。

## Inference：thinker-performer 怎样把统一模型跑成实时系统

训练时 Wan-Streamer 是单个 end-to-end 模型。部署时，为了 overlap 和硬件利用率，它被拆成 thinker 和 performer 两部分。

thinker 负责：

- causal audio/video encoders；
- token-causal Transformer path，用于 language prediction 和 state update；
- KV-cache construction；
- causal audio/video decoders，将上一单元 latents 解码成可输出的音视频。

performer 只负责：

- latent generation path；
- flow-matching solver；
- 基于 full-history KV context 生成下一单元 clean audio/video latents。

第 $k$ 个 streaming step 的时序是：

1. thinker 接收当前用户音视频观测 $u_k$；
2. thinker 编码输入并更新 KV cache；
3. thinker 接收 performer 上一步生成的 $y_{k-1}$ latents；
4. thinker 解码 $y_{k-1}$ 并立即输出；
5. thinker 把当前 KV slice 发给 performer；
6. performer 基于 full-history KV 运行 flow-matching solver，生成 $y_k$ latents；
7. $y_k$ 在下一个 unit 返回 thinker 解码。

这不是把系统拆回级联模型。语义上仍然是一个 unified model state，只是把重计算的 flow solver 放到 performer GPU，让 encoding、state update、decoding 和 denoising 互相重叠。

![Thinker-performer overlap](/files/blogs_image/260625-wan-streamer-thinker-performer-overlap.png)

*图：官方 thinker-performer overlap 图。GPU 0 上的 thinker 负责编码、KV 更新和上一响应单元解码；GPU 1 上的 performer 负责下一响应单元的 flow-matching latent generation。*

论文给出的实时条件也很明确：系统能实时运行，只要 performer time 加上 KV/latent communication overhead 小于一个 160 ms streaming unit。model-side signal-to-signal latency 则是 encode、state update、latent generation、decode 的总路径，当前约 200 ms。

## Evaluation：验证集、指标和 baseline 是否公平

论文的实验更像系统可行性报告，而不是标准 benchmark paper。最主要的量化证据是 latency / runtime comparison。

### Response-latency comparison

论文把 speech / omni-modal dialogue 系统放在一张表里。Wan-Streamer 的行是：

- interaction: text/audio/video in/out；
- user-visible response: about 550 ms total including 350 ms network；
- other reported metric: about 200 ms model-side, 25 FPS video output；
- comparison boundary: one end-to-end model，text I/O、speech、synchronized visual response share one causal stream。

这个表的关键不是“谁数字最小”，而是 measurement boundary。作者也提醒：有的系统报 model-internal latency，有的报 first-packet，有的报 API TTFB，有的含 endpointing，有的不含网络。Wan-Streamer 的优势在于报告的是更完整的 audio-visual response path，但严格横向比较并不成立。

### Visual-agent runtime comparison

第二张表比较 visual agents、streaming avatars 和 audio-visual generators。论文把它们分成两类：

- full-loop or interactive digital-human systems，例如 Body of Her、MIDAS、U-Mind、X-Streamer、LPM、MAViD、M.I.O；
- avatar rendering or joint audio-visual generation components，例如 VASA-1、TalkingMachines、StreamAvatar、Avatar Forcing、LiveTalk、Hallo-Live、OmniForcing。

很多系统有很好的 FPS、first-frame 或 renderer latency，但它们通常不包含上游 dialogue、ASR、TTS 或用户视频感知。Wan-Streamer 的 claim 是：它把 text I/O、用户 audio-video perception、response timing、speech generation 和 25 FPS visual expression 放在同一个 end-to-end causal stream 里。

### Naturalness / interruption / proactive speaking

论文还讨论了自然性、打断和主动发言：

- idle 时不冻结 portrait，而是维持 identity、gaze、posture、breathing、subtle facial motion；
- listening 时能产生 gaze shifts、nods、micro-expressions、posture changes；
- speech 和 video latents 由同一 causal context 预测，因此 lip motion、facial dynamics、prosody 是 native synchronized；
- 生成时仍持续消费用户 audio-video，因此可被自然打断，也可以根据视觉事件主动发言。

这些描述很有方向价值，但证据主要是定性，没有人评、同步误差、interruption success rate、identity drift 指标或 ablation。

## 实验与证据：哪些 claim 被支持，哪些还不够

我会把论文 claims 分成三档。

第一档，机制层面支持较强：single Transformer、block-causal streaming、clean history、joint flow matching、thinker-performer overlap 都在 source 中给出明确公式或流程图。这些内容可以被理解和复述。

第二档，系统指标有报告但边界有限：200 ms model-side、550 ms total、25 FPS、160 ms unit 都有文本描述；但硬件型号、batch、solver steps、resolution-dependent scaling 等细节不足。

第三档，能力 claim 还主要是定性：naturalness、interruption、proactive speaking、long-horizon consistency、identity stability 缺少可复核量化指标。项目 demo 能说明形态，但不能替代实验。

最需要警惕的是“192p proof of concept”。论文结论说 scaling to higher resolutions is straightforward and left to future work。这句话是作者判断，不是已证明结果。分辨率提升会影响 video VAE、decoder cost、flow solver cost、network bandwidth、A/V sync 和 latency budget，不能直接外推。

## 复现与工程风险

这篇论文目前复现风险很高。

缺少的关键材料包括：

- official code；
- checkpoints；
- dataset and split；
- token schedule；
- audio/video VAE details；
- block-causal attention mask implementation；
- model size；
- solver step number；
- distillation recipe；
- latency hardware and profiling setup；
- evaluation scripts。

工程上，即使要做一个简化版，也不应直接追完整 Wan-Streamer。更现实的路线是先验证 thinker-performer serving pattern：一个进程/GPU 做 streaming perception + KV update + decode，另一个进程/GPU 做下一单元 latent generation。等 serving loop 跑通，再考虑把语言、音频、视频放到同一 Transformer。

研究上，我最想看到的补充实验是：

1. 去掉 block-causal attention 的 ablation；
2. 去掉 clean latent write-back 的 long-session degradation；
3. 级联 ASR/LLM/TTS/avatar baseline 的同硬件端到端对比；
4. 192p、360p、720p 的 latency-resolution scaling；
5. interruption success rate 和 latency；
6. A/V sync error、identity stability、人类偏好评测。

## 总结

Wan-Streamer v0.1 的价值在于重新定义了实时音视频 agent 的建模边界。它不是一个更快的 talking-head renderer，也不是一个接了摄像头的 speech model，而是把 text/audio/video input-output 都放进同一个因果历史，让模型同时学习感知、推理、说话、可见倾听、打断和视频回应。

它的技术主线很清楚：single Transformer + block-causal attention 负责统一状态，causal audio/video VAEs 和 encoders/decoders 负责流式表示，conditional flow matching 负责生成 audio/video latents，thinker-performer overlap 负责部署时把 160 ms unit 和 200 ms model-side latency 做出来。

但这篇目前更像方向宣言和系统 proof-of-concept。它证明了一个合理架构可以被组织起来，并给出了低延迟目标；还没有证明该架构在高分辨率、长时间、多用户、多场景下稳定，也没有给社区复现所需的训练和模型细节。

我的最终判断：值得持续关注，尤其是 thinker-performer serving、clean latent history 和 full-duplex interleaved data format；但不要把 200 ms / 550 ms 当成可直接复用的工程结论，除非后续开放硬件、模型、代码和更完整评测。
