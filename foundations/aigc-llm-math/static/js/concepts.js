(function () {
  const transformationFocus = new Map([
    ['Scalar 标量', '标量不是空间中的方向，而是调节长度、概率、温度或 loss 强度的单个数；在线性组合里它控制一个方向被放大、缩小或反向。'],
    ['Vector 向量', '把向量看成空间中的点或箭头；模型里的 embedding 是这个点在表示空间中的坐标。'],
    ['Matrix 矩阵', '矩阵不是普通数字表，而是记录基向量落点的线性变换；列向量告诉你空间被拉伸、旋转、剪切或压缩到哪里。'],
    ['转置矩阵', '转置是在输入轴和输出轴之间换观察方向；attention 里的 \\(K^\\top\\) 让 query 可以和每个 key 做成批点积。'],
    ['单位矩阵', '单位矩阵是“不移动空间”的基准变换；残差连接可以理解为在原表示上叠加一个可学习位移。'],
    ['逆矩阵', '逆矩阵描述一条可逆运动的反向路径；不可逆通常意味着某些方向被压扁，信息无法从输出恢复。'],
    ['行列式', '行列式读作体积缩放因子；为 0 表示空间被压到低维，变换失去可逆性。'],
    ['Tensor 张量', '张量是把多个坐标轴叠在一起的数组；先看每个轴代表 batch、时间、head 还是通道，再谈变换。'],
    ['Shape 形状', 'shape 是张量空间的坐标规格；它决定矩阵乘法、广播、mask 和 cache 是否在同一套轴上发生。'],
    ['Dot Product 点积', '点积把一个向量投到另一个方向上，读作“沿这个方向有多少成分”。'],
    ['Matrix Multiplication 矩阵乘法', '矩阵乘法是变换复合：右边的变换先发生，左边的变换再作用。'],
    ['Vector Space 向量空间', '向量空间是允许加法和缩放自由组合的舞台；模型表示只有先落在同一空间里，距离、方向和投影才有意义。'],
    ['Basis 基', '基是一组坐标尺；换基不是换对象，而是换描述同一个对象的坐标语言。'],
    ['Projection 投影', '投影是只保留某个方向或子空间中的影子，丢掉垂直于它的成分。'],
    ['Norm 范数', '范数给空间中的点或变换一个大小尺度；它回答“这一步运动有多强”。'],
    ['Cosine Similarity 余弦相似度', '余弦相似度只看两个向量方向夹角，弱化长度，让相似性更像“朝向是否一致”。'],
    ['矩阵迹', 'trace 把一个方阵压成标量，常用来把多维二次型或矩阵求导写成可优化的标量表达。'],
    ['特征值', '特征值是某条不改方向的运动轨道上的缩放倍数。'],
    ['特征向量', '特征向量是空间变换后方向仍不变的特殊方向。'],
    ['特征分解', '特征分解是在特征向量坐标系里描述同一个变换，让复杂运动变成各方向独立缩放。'],
    ['SVD 分解', 'SVD 把任意矩阵拆成输入空间旋转、沿主轴缩放、输出空间旋转三步运动。'],
    ['矩阵秩', 'rank 是变换后空间还能张开的独立方向数；rank 降低就是空间被压扁。'],
    ['Low-rank Approximation 低秩近似', '低秩近似只保留最重要的运动方向，用较少自由度近似完整变换。'],
    ['正定矩阵', '正定矩阵表示每个非零方向上的二次型都向上，几何上像一个没有下陷方向的碗形曲率。'],
    ['正交矩阵', '正交矩阵像刚体旋转或反射，保持长度和角度不变。'],
    ['PCA 主成分分析', 'PCA 是寻找数据云最主要的运动/变化方向，再把数据投到这些方向上。'],
    ['LDA 线性判别分析', 'LDA 是寻找让类间分开、类内压紧的投影方向。'],
    ['High-dimensional Geometry 高维几何', '高维几何提醒你：空间维度升高后，距离、角度、体积和最近邻直觉都会变形。']
  ]);

  const specialQuestions = new Map([
    ['Scalar 标量', ['loss、temperature、learning rate 都是标量，为什么不能直接比较大小？', '标量出现在日志里时，怎样先确认单位、归一化和统计窗口？']],
    ['Vector 向量', ['为什么 embedding 向量不能把每个维度直接解释成人类语义？', '两个向量看起来相近时，应该先比较点积、范数还是余弦相似度？']],
    ['Matrix 矩阵', ['怎样从矩阵的列向量看出它把空间移动到了哪里？', '为什么很多 Transformer bug 本质上是矩阵输入输出轴写反？']],
    ['Matrix Multiplication 矩阵乘法', ['为什么矩阵乘法要按“右边先作用、左边后作用”来读？', 'FLOPs 相同的矩阵乘法为什么实际延迟可能不同？']],
    ['SVD 分解', ['SVD 和特征分解到底差在哪？', '为什么保留最大奇异值不等于保留任务能力？']],
    ['矩阵秩', ['矩阵秩为什么表示线性变换保留下来的独立方向数？', 'LoRA 的矩阵秩过低时通常会损失什么能力？']],
    ['PCA 主成分分析', ['PCA 为什么保留最大方差方向，而不一定保留分类最有用方向？', '用 PCA 看 embedding 时，低维图能证明什么、不能证明什么？']],
    ['LDA 线性判别分析', ['LDA 和 PCA 的目标为什么不同？', '为什么这里的 LDA 不是主题模型 Latent Dirichlet Allocation？']],
    ['Gradient 梯度', ['梯度为什么只告诉局部最陡方向，而不保证一步到最优？', '梯度消失或爆炸时，应该先看哪些层的梯度范数？']],
    ['概率质量函数', ['概率质量函数 PMF 的单点值为什么就是概率？', 'token 分布里概率质量函数归一化失败通常说明哪里错了？']],
    ['概率密度函数', ['为什么 PDF 的值可以大于 1，而单点概率仍然通常为 0？', '在 diffusion noise 或 latent 连续变量里，什么时候不能套用 token 概率直觉？']],
    ['贝叶斯公式', ['贝叶斯公式里的分母 evidence 为什么经常成为计算瓶颈？', 'VAE 为什么要用近似后验而不是直接算精确后验？']],
    ['KL Divergence KL 散度', ['为什么 \\(D_{KL}(p\\Vert q)\\) 和 \\(D_{KL}(q\\Vert p)\\) 不能互换？', 'RLHF/DPO 里 KL 太大或太小分别会出现什么现象？']],
    ['Perplexity 困惑度', ['perplexity 低为什么不等于回答质量一定高？', '比较两个模型的 PPL 时，为什么 tokenizer 和评测集必须一致？']],
    ['Softmax Gradient', ['为什么 softmax + cross entropy 的梯度会简化成 \\(p-y\\)？', '把多分类 softmax 误当成多个独立 sigmoid 会导致什么错误？']],
    ['AdamW', ['AdamW 为什么要把 weight decay 从梯度更新里解耦？', 'loss spike 出现时，AdamW 的哪些状态需要一起看？']],
    ['Quantization 量化', ['为什么量化省显存不一定降低端到端延迟？', '量化后质量回退时，为什么要检查 calibration 和 outlier channel？']],
    ['FlashAttention', ['FlashAttention 为什么说优化的是 IO 而不只是 FLOPs？', '为什么它能不物化完整 attention matrix 仍保持 exact attention？']],
    ['LayerNorm', ['LayerNorm 为什么按样本内部特征归一化，而不是按 batch 归一化？', 'Pre-LN 和 Post-LN 的差异为什么会影响深层训练稳定性？']],
    ['RMSNorm', ['RMSNorm 去掉均值中心化为什么仍然能稳定尺度？', '从 LayerNorm 换成 RMSNorm 时应重点监控哪些激活尺度？']],
    ['RoPE 旋转位置编码', ['RoPE 为什么能把绝对位置旋转成相对位置信息？', '长上下文外推时为什么高频维度更容易出问题？']],
    ['Attention', ['attention 权重能不能直接当作模型解释？', 'attention 分数异常尖锐时应检查缩放、mask 还是输入范数？']],
    ['Scaled Dot-product Attention', ['为什么 attention logits 要除以 \\(\\sqrt{d_k}\\)？', 'causal mask 加错时会怎样污染 next-token training？']],
    ['KV Cache', ['KV cache 为什么降低 decode 重算，却增加显存和带宽压力？', '长上下文服务里 cache 成本应该按哪些维度估算？']],
    ['LoRA', ['LoRA 为什么用 \\(BA\\) 低秩更新就能适配任务？', 'target module、rank 和 alpha 改动分别影响什么？']],
    ['Diffusion Model 扩散模型', ['diffusion 的训练目标和采样器为什么是两件事？', '采样步数减少后，应该怎样区分速度收益和质量退化？']],
    ['Score', ['score 是密度梯度，为什么不是 reward？', 'score 估计错误会怎样在反向采样中累积？']],
    ['Classifier-free Guidance', ['CFG 为什么会提升条件遵循但牺牲多样性？', 'guidance scale 过大时常见伪影有哪些？']],
    ['Flow Matching', ['Flow Matching 学的是速度场，为什么不等于直接学习样本？', 'NFE 改动时如何判断是 solver 误差还是模型能力问题？']],
    ['Optimal Transport 最优传输', ['最优传输为什么能把“分布差异”解释成搬运成本？', 'OT 路径用于 Flow Matching 时，工程上通常贵在哪里？']],
    ['DPO 直接偏好优化', ['DPO 为什么可以不显式训练 Reward Model？', 'DPO 的 reference model 选错会带来什么漂移风险？']],
    ['PPO', ['PPO 的 clipped objective 到底在限制什么？', 'RLHF 里 PPO 不稳定时应同时看 reward、KL、clip 还是采样？']],
    ['Reward Hacking', ['reward 曲线上升为什么可能代表模型学会了钻空子？', '怎样用人工抽检和分布外任务发现 reward hacking？']],
    ['Confidence Interval 置信区间', ['置信区间重叠时能不能直接说两个模型没有差异？', 'CI 很窄为什么仍然不能排除 benchmark bias？']],
    ['Bootstrap', ['bootstrap 为什么能给复杂指标估计不确定性？', '样本不独立时 bootstrap 区间为什么会过度乐观？']],
    ['P-value', ['p-value 为什么不是“假设为真的概率”？', '大量 benchmark 子集同时比较时为什么 p-value 会膨胀误报？']],
    ['Calibration 校准', ['准确率高的模型为什么仍可能校准很差？', 'LLM judge 的置信度校准应怎么抽查？']]
  ]);

  const mathDetailAdditions = new Map([
    ['Vector Space 向量空间', [
      '形式定义：若 \\(u,v\\in V\\)、\\(a,b\\in\\mathbb{R}\\) 时总有 \\(au+bv\\in V\\)，则 \\(V\\) 是向量空间；这里的封闭性保证线性组合仍留在同一表示空间。'
    ]],
    ['Basis 基', [
      '若 \\(B=[b_1,\\dots,b_d]\\) 线性无关且 \\(\\operatorname{span}(B)=V\\)，任意 \\(x\\in V\\) 都可唯一写成 \\(x=\\sum_i \\alpha_i b_i=B\\alpha\\)。'
    ]],
    ['PCA 主成分分析', [
      '中心化数据 \\(X_c\\in\\mathbb{R}^{n\\times d}\\) 的协方差为 \\(C=X_c^\\top X_c/(n-1)\\)，PCA 求 \\(Cv_i=\\lambda_i v_i\\)，按 \\(\\lambda_i\\) 从大到小保留主方向。',
      '投影坐标写作 \\(z=X_cV_k\\)，其中 \\(V_k\\in\\mathbb{R}^{d\\times k}\\) 是前 \\(k\\) 个主方向。'
    ]],
    ['High-dimensional Geometry 高维几何', [
      '若 \\(x,y\\sim\\mathcal{N}(0,I_d)\\)，则 \\(\\mathbb{E}[x^\\top y]=0\\)，\\(\\|x\\|_2\\approx\\sqrt d\\)；高维随机向量的角度常接近 \\(90^\\circ\\)。'
    ]],
    ['Computational Graph 计算图', [
      '把程序写成节点 \\(v_i=f_i(\\operatorname{pa}(v_i))\\)。反向传播保存伴随量 \\(\\bar v_i=\\partial L/\\partial v_i\\)，并按拓扑逆序累加到父节点。'
    ]],
    ['Backpropagation 反向传播', [
      '对边 \\(u\\to v\\)，反向传播更新 \\(\\bar u\\mathrel{+}=\\bar v\\,\\partial v/\\partial u\\)；这里 \\(\\bar v\\) 是上游梯度，不是参数更新量。'
    ]],
    ['Automatic Differentiation 自动微分', [
      'reverse-mode 计算 \\(v^\\top J\\)，适合标量 loss 对大量参数求梯度；forward-mode 计算 \\(Jv\\)，适合少量输入方向的敏感性分析。'
    ]],
    ['Sampling 采样', [
      '离散采样可写作 \\(x\\sim p(x)\\)，满足 \\(\\sum_x p(x)=1\\)；连续采样写作 \\(x\\sim p(x)\\)，区间概率为 \\(P(a<x<b)=\\int_a^b p(x)dx\\)。'
    ]],
    ['概率密度函数', [
      '连续变量的 PDF 要通过 integral 读作区间面积：\\(P(a<X<b)=\\int_a^b p(x)dx\\)。这里的 \\(p(x)dx\\)，也就是 p(x) dx，才是很小区间上的概率近似，\\(p(x)\\) 本身是密度。'
    ]],
    ['Softmax Gradient', [
      'Jacobian 元素为 \\(\\partial p_i/\\partial z_j=p_i(\\delta_{ij}-p_j)\\)，其中 \\(\\delta_{ij}\\) 是 Kronecker delta；和 one-hot cross entropy 合起来，logits 梯度化简为 \\(p-y\\)。'
    ]],
    ['KL Divergence KL 散度', [
      '离散形式 \\(D_{KL}(p\\Vert q)=\\sum_x p(x)\\log\\frac{p(x)}{q(x)}\\)。若某些 support 上 \\(p(x)>0\\) 但 \\(q(x)=0\\)，KL 会发散；连续情形把求和换成积分。'
    ]],
    ['Loss Function 损失函数', [
      '常见 NLL 写作 \\(L(\\theta)=-\\sum_t\\log p_\\theta(x_t|x_{<t})\\)，MSE 写作 \\(L=\\|y-\\hat y\\|_2^2/n\\)；loss 的归一化单位会影响数值尺度。'
    ]],
    ['Adam', [
      'Adam 状态为 \\(m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t\\)，\\(v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2\\)。',
      'bias correction 使用 \\(\\hat m_t=m_t/(1-\\beta_1^t)\\)、\\(\\hat v_t=v_t/(1-\\beta_2^t)\\)，更新为 \\(\\theta_t=\\theta_{t-1}-\\eta\\hat m_t/(\\sqrt{\\hat v_t}+\\epsilon)\\)。'
    ]],
    ['AdamW', [
      'AdamW 保留 Adam 的 \\(m_t\\)、\\(v_t\\) 和 bias correction，但把权重衰减 decoupled 到参数更新：\\(\\theta_t=\\theta_{t-1}-\\eta\\hat m_t/(\\sqrt{\\hat v_t}+\\epsilon)-\\eta\\lambda\\theta_{t-1}\\)。',
      '这里 \\(\\lambda\\) 是 weight decay 系数；decoupled 的意思是衰减不进入自适应梯度分母。'
    ]],
    ['Learning Rate Schedule 学习率调度', [
      'warmup 可写作 \\(\\eta_t=\\eta_{max}t/T_w\\) for \\(t<T_w\\)；cosine decay 常写作 \\(\\eta_t=\\eta_{min}+\\frac12(\\eta_{max}-\\eta_{min})(1+\\cos(\\pi t/T))\\)。'
    ]],
    ['Weight Decay 权重衰减', [
      'L2 正则目标为 \\(L_{reg}=L+\\frac{\\lambda}{2}\\|\\theta\\|_2^2\\)，SGD 下等价于梯度加 \\(\\lambda\\theta\\)；AdamW 中常用 decoupled update \\(\\theta\\leftarrow(1-\\eta\\lambda)\\theta-\\eta g_{adam}\\)。'
    ]],
    ['Gradient Clipping 梯度裁剪', [
      'global norm clipping 先算 \\(\\|g\\|_2=\\sqrt{\\sum_i\\|g_i\\|_2^2}\\)，若 \\(\\|g\\|_2>c\\)，令 \\(g_i\\leftarrow c g_i/\\|g\\|_2\\)。',
      '这里 \\(c\\) 是阈值；频繁触发说明学习率、loss scale、batch 或 reward 尺度可能异常。'
    ]],
    ['Non-convex Optimization 非凸优化', [
      '非凸表示存在点满足 \\(f(\\alpha x+(1-\\alpha)y)>\\alpha f(x)+(1-\\alpha)f(y)\\)。神经网络 loss 常有鞍点、平坦谷和多个局部 basin。'
    ]],
    ['Floating Point 浮点数', [
      '二进制浮点数通常写作 \\(x=(-1)^s\\times 2^{e-bias}\\times(1.f)\\)，其中 \\(s\\) 是 sign bit，\\(e\\) 是 exponent，\\(f\\) 是 fraction。',
      'exponent 全 0 表示 zero 或 subnormal，exponent 全 1 表示 Inf 或 NaN；训练中的 overflow 常变成 Inf/NaN，underflow 会把小梯度冲到 0 或 subnormal。'
    ]],
    ['FP32 / FP16 / BF16', [
      'FP32: 1 sign bit + 8 exponent bits + 23 fraction bits，bias 127，normal 数值读作 \\(x=(-1)^s2^{e-127}(1.f)\\)。',
      'FP16: 1 sign bit + 5 exponent bits + 10 fraction bits，bias 15，normal 数值读作 \\(x=(-1)^s2^{e-15}(1.f)\\)，范围更小，训练中更容易 overflow/underflow。',
      'BF16: 1 sign bit + 8 exponent bits + 7 fraction bits，bias 127；它保留接近 FP32 的指数范围，但 fraction 更短，所以动态范围强、有效精度低。',
      'exponent 全 0 对应 zero/subnormal，exponent 全 1 且 fraction 为 0 是 Inf，exponent 全 1 且 fraction 非 0 是 NaN。'
    ]],
    ['Mixed Precision 混合精度', [
      '常见 AMP 使用 FP16/BF16 做 forward/backward，同时保留 FP32 master weight、optimizer state 和部分归一化/归约。',
      'loss scaling 把 \\(L\\) 临时变成 \\(S L\\)，反向得到 \\(Sg\\)，再 unscale 为 \\(g\\)；这样减少 FP16 gradient underflow，但要检测 overflow 后跳过或降低 \\(S\\)。'
    ]],
    ['Quantization 量化', [
      '线性量化常写作 \\(q=\\operatorname{clip}(\\operatorname{round}(x/s)+z,q_{min},q_{max})\\)，反量化为 \\(\\hat x=s(q-z)\\)，其中 \\(s\\) 是 scale，\\(z\\) 是 zero-point。',
      'symmetric quantization 令 \\(z=0\\)，asymmetric quantization 允许 \\(z\\ne0\\)；per-channel scale 比 per-tensor scale 更能处理 outlier channel。',
      'calibration 用代表性样本估计 range/scale；校准集偏移会让 INT8/INT4 推理出现系统性质量回退。'
    ]],
    ['Conditioning 条件数', [
      '矩阵条件数 \\(\\kappa(A)=\\sigma_{max}(A)/\\sigma_{min}(A)\\)。\\(\\kappa\\) 越大，输入或梯度中的小扰动越容易被放大。'
    ]],
    ['Initialization 初始化', [
      'Xavier 常用 \\(\\operatorname{Var}(W)=2/(fan_{in}+fan_{out})\\)，Kaiming/ReLU 常用 \\(\\operatorname{Var}(W)=2/fan_{in}\\)。目标是让激活和梯度方差跨层不过快放大或衰减。'
    ]],
    ['Normalization 归一化', [
      '统一形式可写成 \\(y=\\gamma(x-\\mu)/\\sqrt{\\sigma^2+\\epsilon}+\\beta\\)。不同 norm 的区别在于 \\(\\mu,\\sigma^2\\) 沿 batch、token、channel 还是 hidden 维统计。'
    ]],
    ['Tokenizer', [
      'tokenizer 是映射 \\(\\tau: text\\to(t_1,\\dots,t_T)\\)，其中 \\(t_i\\in\\{1,\dots,|V|\\}\\)；模型实际看到的是整数序列而不是原始字符串。'
    ]],
    ['Vocabulary 词表', [
      '词表大小 \\(|V|\\) 决定 embedding table \\(E\\in\\mathbb{R}^{|V|\\times d}\\) 和 LM head logits \\(l_t\\in\\mathbb{R}^{|V|}\\) 的维度。'
    ]],
    ['Positional Encoding 位置编码', [
      '绝对位置可写作 \\(h_t=x_t+p_t\\)；相对/旋转形式让 attention score 显式依赖 \\(i-j\\) 或位置旋转后的 \\(q_i^\\top k_j\\)。'
    ]],
    ['RoPE 旋转位置编码', [
      'RoPE 对每对维度施加旋转 \\(R_{\\theta_m t}\\)，使 \\((R_iq)^\\top(R_jk)=q^\\top R_{j-i}k\\)，从而把相对位置差写进点积。'
    ]],
    ['Attention', [
      '自注意力核心公式为 \\(\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}(QK^\\top/\\sqrt{d_k}+M)V\\)，其中 \\(M\\) 是 mask，\\(Q,K,V\\in\\mathbb{R}^{B\\times H\\times T\\times d_h}\\)。'
    ]],
    ['KV Cache', [
      'KV cache 保存每层历史 \\(K,V\\)。大小近似为 \\(2\\times L\\times B\\times H\\times T\\times d_h\\times bytes\\)。这里 L 是层数，B 是 batch，H 是 heads，T 是上下文长度，d_h 是每个 head 的维度，bytes 是每个元素占用字节数。'
    ]],
    ['Top-k Sampling', [
      '令 \\(S_k\\) 为概率最高的 \\(k\\) 个 token，采样分布变为 \\(p_i^\\prime=p_i/\\sum_{j\\in S_k}p_j\\) for \\(i\\in S_k\\)，其他 token 概率为 0。'
    ]],
    ['Prefix Tuning / Prompt Tuning', [
      'soft prompt 学习 \\(P\\in\\mathbb{R}^{m\\times d}\\) 并与输入 embedding 拼接；prefix tuning 常学习每层 \\(K_p,V_p\\in\\mathbb{R}^{H\\times m\\times d_h}\\)。'
    ]],
    ['Diffusion Model 扩散模型', [
      '前向加噪 \\(q(x_t|x_{t-1})=\\mathcal{N}(\\sqrt{1-\\beta_t}x_{t-1},\\beta_tI)\\)，反向模型学习 \\(p_\\theta(x_{t-1}|x_t,c)\\)。'
    ]],
    ['Reverse SDE 反向 SDE', [
      '若正向 SDE 为 \\(dx=f(x,t)dt+g(t)dw\\)，反向过程的 drift 含 \\(f(x,t)-g(t)^2\\nabla_x\\log p_t(x)\\)，其中 score 控制回到数据密度的方向。'
    ]],
    ['PPO', [
      'PPO clipped objective 常写作 \\(L=\\mathbb{E}[\\min(r_tA_t,\\operatorname{clip}(r_t,1-\\epsilon,1+\\epsilon)A_t)]\\)，其中 \\(r_t=\\pi_\\theta(a_t|s_t)/\\pi_{old}(a_t|s_t)\\)。'
    ]],
    ['RLAIF', [
      'RLAIF 把偏好来源换成 AI judge：数据仍可写成 \\((x,y_w,y_l)\\)，再训练 reward model 或直接优化 \\(\\log\\pi_\\theta(y_w|x)-\\log\\pi_\\theta(y_l|x)\\)。'
    ]],
    ['Reward Hacking', [
      '可写成代理目标 \\(r_{proxy}(x,y)\\) 上升，但真实目标 \\(r_{true}(x,y)\\) 下降或不变；诊断重点是比较 \\(\\Delta r_{proxy}\\) 与人工/分布外指标。'
    ]],
    ['Benchmark', [
      'benchmark 分数是统计量 \\(\\hat\\mu=\\frac1n\\sum_{i=1}^n m_i\\)，其中 \\(m_i\\) 是样本级正确率、得分或胜负变量。'
    ]],
    ['Bootstrap', [
      'bootstrap 从样本 \\(D=\\{x_i\\}_{i=1}^n\\) 有放回抽取 \\(D_b^*\\)，计算 \\(\\hat\\theta_b^*=T(D_b^*)\\)，再用 \\(\\{\\hat\\theta_b^*\\}_{b=1}^B\\) 估计区间。'
    ]],
    ['Confidence Interval 置信区间', [
      '均值的正态近似常写作 \\(\\bar x\\pm1.96\\,s/\\sqrt n\\)，也可读成 \\(mean \\pm 1.96\\times standard\\ error\\)，其中 standard error 是 \\(s/sqrt(n)\\)。'
    ]],
    ['Multiple Comparison 多重比较', [
      'Bonferroni 校正常用阈值 \\(\\alpha/m\\)，其中 \\(m\\) 是检验次数；FDR 控制的是被判显著结果中的期望错误比例。'
    ]],
    ['Calibration 校准', [
      'Expected Calibration Error: \\(ECE=\\sum_b \\frac{|B_b|}{n}|acc(B_b)-conf(B_b)|\\)，其中 \\(B_b\\) 是置信度分桶；acc(B_b) 是分桶准确率，conf(B_b) 是分桶平均置信度。'
    ]],
    ['Bias-Variance Tradeoff 偏差-方差权衡', [
      '平方损失下 \\(\\mathbb{E}[(\\hat f(x)-y)^2]=\\operatorname{Bias}(\\hat f)^2+\\operatorname{Var}(\\hat f)+\\sigma^2\\)，三项分别对应系统误差、估计波动和不可约噪声。'
    ]],
    ['VC Dimension', [
      '若存在 \\(n\\) 个点对任意 \\(2^n\\) 种二分类标记都可由函数类 \\(\\mathcal{F}\\) 实现，则这些点被 shatter；VC dimension 是最大这样的 \\(n\\)。'
    ]]
  ]);

  function c({ group, name, aliases, brief, detail, qa }) {
    return { group, name, aliases: aliases || [], brief, detail, qa };
  }

  function concept(options) {
    return c(options);
  }

  function legacy(group, name, aliases, intuition, math, model, pitfall) {
    return c(expandConcept({ group, name, aliases: aliases || [], intuition, math, model, pitfall }));
  }

  function term(name) {
    return name.replace(/\s+/g, ' ').trim();
  }

  function flatten(value) {
    return Array.isArray(value) ? value.join(' ') : String(value || '');
  }

  function tailText(value) {
    if (Array.isArray(value)) return value.slice(1).join(' ');
    return String(value || '');
  }

  function snippet(value, maxLength) {
    const cleaned = String(value || '')
      .replace(/\\\([^)]*\\\)/g, '公式')
      .replace(/\\\[[\s\S]*?\\\]/g, '公式')
      .replace(/\s+/g, ' ')
      .trim();
    const firstClause = cleaned.split(/[；。]/)[0] || cleaned;
    return firstClause.length > maxLength ? firstClause.slice(0, maxLength - 1) + '…' : firstClause;
  }

  function lineFor(base, fallback) {
    return transformationFocus.get(base.name) || fallback;
  }

  function mathAdditions(base) {
    return mathDetailAdditions.get(base.name) || [];
  }

  function contextSentence(base) {
    const model = snippet(base.model, 80);
    if (base.group === '微积分与自动微分') return '典型落点：' + model + '。读它时关注局部变化量如何沿计算图回传到参数或中间激活。';
    if (base.group === '概率统计与信息论') return '典型落点：' + model + '。先确认随机对象和采样来源，离散 token、连续噪声和样本统计量不能用同一种单点直觉解释。';
    if (base.group === '优化与数值计算') return '典型落点：' + model + '。它影响更新步、数值范围或执行成本，不能只看公式表面是否正确。';
    if (base.group === '深度学习机制') return '典型落点：' + model + '。重点看信号如何穿过很多层，以及表达能力、尺度稳定性和梯度通路是否同时成立。';
    if (base.group === 'Transformer 与 LLM') return '典型落点：' + model + '。它改变 token 表示、上下文混合、logits 分布或推断状态中的一个具体环节。';
    if (base.group === '生成模型') return '典型落点：' + model + '。它对应从噪声、latent、条件或速度场走向样本的某一段机制。';
    if (base.group === '对齐与偏好优化') return '典型落点：' + model + '。它说明偏好信号怎样改变 policy，或 reference/KL 怎样限制这种改变。';
    if (base.group === '评测统计与泛化') return '典型落点：' + model + '。它把模型表现变成可估计、可比较但带不确定性的统计结论。';
    return '典型落点：' + model + '。先把这个落点和公式中的变量对应起来。';
  }

  function mathSentence(base) {
    const pitfall = snippet(base.pitfall, 70);
    if (base.group === '微积分与自动微分') return '读这条式子时要标出谁是输入、谁是输出、梯度沿哪条链路返回；否则很容易出现“' + pitfall + '”。';
    if (base.group === '概率统计与信息论') return '求和、积分、期望或对数项必须跟采样分布配对；混错对象时，最常见的问题就是“' + pitfall + '”。';
    if (base.group === '优化与数值计算') return '公式里的学习率、尺度、精度和更新时机都算语义的一部分；忽略它们会直接触发“' + pitfall + '”。';
    if (base.group === '深度学习机制') return '把公式落到张量轴上看：它作用在 token、hidden 维、通道还是层级状态，决定了“' + pitfall + '”是否会发生。';
    if (base.group === 'Transformer 与 LLM') return '先写出 batch、sequence、head、hidden 或 vocab 轴，再读 softmax、mask、cache 或采样公式，才能避免“' + pitfall + '”。';
    if (base.group === '生成模型') return '要区分训练 loss、分布路径和实际 sampler；三者混在一起时，常会误判为“' + pitfall + '”。';
    if (base.group === '对齐与偏好优化') return 'chosen/rejected、log probability、reward、KL 和 reference 的来源必须写清；否则“' + pitfall + '”会被训练过程放大。';
    if (base.group === '评测统计与泛化') return '样本量、抽样方式、零假设和置信水平决定公式能支持多强的结论；忽略这些就会落入“' + pitfall + '”。';
    return '读公式时把变量、shape、采样来源和适用条件写在旁边。';
  }

  function diagnosticSentence(base) {
    const model = snippet(base.model, 72);
    if (base.group === '微积分与自动微分') return '实验里优先看梯度范数、NaN/Inf、detach 边界和自定义 backward；这些信号能定位 ' + model + ' 的责任链是否断了。';
    if (base.group === '概率统计与信息论') return '排查时看归一化、采样分布、估计方差和单位；' + model + ' 的异常经常来自把概率、密度和统计估计混用。';
    if (base.group === '优化与数值计算') return '日志上要同时看 loss、gradient norm、optimizer state、吞吐和显存；单个曲线很难解释 ' + model + ' 的问题。';
    if (base.group === '深度学习机制') return '如果训练早期就发散或表示塌缩，先看激活均值/方差、残差比例和 norm 前后尺度，再回到 ' + model + '。';
    if (base.group === 'Transformer 与 LLM') return '推断质量、重复、mask 泄漏或长上下文退化出现时，把问题定位到 ' + model + '，再检查对应张量和概率分布。';
    if (base.group === '生成模型') return '伪影、模式崩塌、条件不跟随或采样变慢时，要先判断问题发生在 ' + model + ' 的训练目标、条件输入还是采样器。';
    if (base.group === '对齐与偏好优化') return '当 reward 上升但人工质量下降，或 KL/风格突然漂移，先检查 ' + model + ' 的数据分布、reference 和 judge 偏差。';
    if (base.group === '评测统计与泛化') return '结论看起来过强时，回到 ' + model + ' 的样本来源、置信区间、数据污染和多重比较风险。';
    return '调试时先把现象、变量和指标对应起来，再判断这个概念是否是真正原因。';
  }

  function detailFor(base) {
    if (base.group === '线性代数与张量') {
      return {
        intuition: [
          base.intuition,
          lineFor(base, '把它放回向量空间：输入方向、输出方向、子空间和尺度变化决定了它的几何含义。')
        ],
        math: [
          base.math,
          '和 ' + snippet(base.model, 78) + ' 对齐时，先标输入空间、输出空间、shape 和基底；矩阵列向量优先读作基向量变换后的落点。',
          '如果出现“' + snippet(base.pitfall, 72) + '”，通常要回到变换方向、rank、子空间或矩阵轴顺序重新画一遍。'
        ].concat(mathAdditions(base)),
        model: [
          base.model
        ],
        pitfall: [
          base.pitfall,
          '不要只盯着数字表；这个对象真正描述的是空间被拉伸、旋转、剪切、压缩、投影或换坐标后的结果。',
          '排查时把输入轴、输出轴、基向量落点和可达子空间写清楚，比直接补 transpose 更可靠。'
        ]
      };
    }
    return {
      intuition: [
        base.intuition,
        contextSentence(base)
      ],
      math: [
        base.math,
        mathSentence(base)
      ].concat(mathAdditions(base)),
      model: [
        base.model
      ],
      pitfall: [
        base.pitfall,
        diagnosticSentence(base)
      ]
    };
  }

  function groupQuestion(base) {
    const name = term(base.name);
    const intuition = snippet(base.intuition, 34);
    if (base.group === '线性代数与张量') return name + ' 的几何图像是什么：方向、子空间，还是一次空间变换？';
    if (base.group === '微积分与自动微分') return name + ' 怎样把“' + intuition + '”变成可回传的局部变化量？';
    if (base.group === '概率统计与信息论') return name + ' 的随机对象到底是什么，单点值能不能直接当概率？';
    if (base.group === '优化与数值计算') return name + ' 改动后，loss 曲线、梯度尺度和吞吐分别会怎样变？';
    if (base.group === '深度学习机制') return name + ' 是在增加表达能力，还是在稳住深层信号流？';
    if (base.group === 'Transformer 与 LLM') return name + ' 具体改的是 token 表示、attention 交互、logits，还是 decoding 状态？';
    if (base.group === '生成模型') return name + ' 对应训练目标、概率路径、分布距离还是实际采样器？';
    if (base.group === '对齐与偏好优化') return name + ' 改变的是偏好信号、policy 分布，还是 reference/KL 约束？';
    if (base.group === '评测统计与泛化') return name + ' 支持的是能力估计、显著性判断，还是分布外可信度？';
    return name + ' 的公式和实验现象应该怎样对应起来？';
  }

  function qaFor(base, detail) {
    const name = term(base.name);
    const questions = specialQuestions.get(base.name) || [
      groupQuestion(base),
      name + ' 出现“' + snippet(base.pitfall, 34) + '”时，先看哪条证据？'
    ];
    return [
      {
        question: questions[0],
        answer: [
          flatten(detail.intuition),
          flatten(detail.math),
          base.model
        ]
      },
      {
        question: questions[1],
        answer: [
          base.pitfall,
          tailText(detail.pitfall)
        ]
      }
    ];
  }

  function expandConcept(base) {
    const brief = {
      intuition: base.intuition,
      math: base.math,
      model: base.model,
      pitfall: base.pitfall
    };
    const detail = detailFor(base);
    const qa = qaFor(base, detail);
    return { group: base.group, name: base.name, aliases: base.aliases, brief, detail, qa };
  }

  window.AIGC_LLM_MATH_CONCEPTS = [
    legacy('线性代数与张量', 'Scalar 标量', ['loss', 'learning rate', 'score'], '一个普通数字，通常表示单个强度、概率、温度、loss 或超参数。', '\\(a\\in\\mathbb{R}\\)。标量是 0 维张量。', 'learning rate、loss、attention score、temperature 都是标量。', '标量虽然简单，但单位和尺度很关键；loss 的绝对大小不能跨任务随意比较。'),
    legacy('线性代数与张量', 'Vector 向量', ['embedding', 'feature'], '一串数字，表示对象在多个方向上的坐标或特征。', '\\(x=[x_1,\\dots,x_d]^\\top\\in\\mathbb{R}^d\\)。', 'token embedding、hidden state、query/key/value 都是向量或向量批。', '向量维度没有天然语义；不要把单个维度直接解释成人类概念。'),
    legacy('线性代数与张量', 'Matrix 矩阵', ['linear layer', 'projection'], '二维数字表，既可以装一组向量，也可以表示线性变换。', '\\(A\\in\\mathbb{R}^{m\\times n}\\)，线性层常写作 \\(Y=XW\\)。', 'embedding table、\\(W_Q,W_K,W_V\\)、MLP projection、LoRA 矩阵。', '矩阵乘法维度必须对齐；很多模型 bug 其实是 shape bug。'),
    legacy('线性代数与张量', 'Linear Transformation 线性变换', ['linear map', 'T(x)'], '把空间中的每个点或向量移动到新位置，同时保持直线、原点和线性组合结构。', '\\(T(ax+by)=aT(x)+bT(y)\\)，矩阵形式常写作 \\(y=Ax\\)。', '线性层、attention 投影、MLP 上下投影和 LoRA 增量本质上都是线性变换。', '不要把线性变换理解成“只能拉直线”；它可以旋转、剪切、缩放和压缩，但不能弯曲网格或移动原点。'),
    legacy('线性代数与张量', 'Linear Combination 线性组合', ['linear combo'], '用若干向量乘以系数再相加，表示一个点如何由基向量或方向拼出来。', '\\(v=\\sum_i \\alpha_i v_i\\)。', '矩阵乘法可以看成用输入坐标对矩阵列向量做线性组合；attention 输出也是 value 向量的加权组合。', '系数来源很关键；attention 权重是归一化概率样式的系数，普通线性组合不一定非负或和为 1。'),
    legacy('线性代数与张量', 'Span 张成空间', ['span'], '一组向量所有线性组合能到达的区域，就是它们张成的空间。', '\\(\\operatorname{span}\\{v_i\\}=\\{\\sum_i\\alpha_i v_i\\}\\)。', '列空间、低秩更新的可表达方向、embedding 子空间和 PCA 主方向都可用 span 来描述。', 'span 大小取决于独立方向，不取决于向量数量；重复方向不会增加可达空间。'),
    legacy('线性代数与张量', 'Linear Independence 线性无关', ['independent vectors'], '没有哪个向量能由其他向量线性组合得到，表示这些方向真正提供了新自由度。', '\\(\\sum_i\\alpha_i v_i=0\\Rightarrow \\alpha_i=0\\)。', 'rank、basis、PCA 主轴、LoRA 有效秩和表示坍缩分析都依赖线性无关。', '向量数量多不代表信息维度高；如果方向相关，矩阵仍可能低秩。'),
    legacy('线性代数与张量', 'Subspace 子空间', ['linear subspace'], '空间中对加法和数乘封闭的一块区域，可以理解成模型表示只活动的局部舞台。', '若 \\(u,v\\in S\\)，则 \\(au+bv\\in S\\)。', 'Q/K/V 投影子空间、LoRA 更新子空间、PCA 降维空间和 residual stream 分析。', '子空间必须经过原点并满足封闭性；普通聚类区域或流形不一定是线性子空间。'),
    legacy('线性代数与张量', 'Column Space 列空间', ['image', 'range'], '矩阵所有可能输出构成的空间，也就是列向量张成的区域。', '\\(\\operatorname{Col}(A)=\\{Ax:x\\in\\mathbb{R}^n\\}\\)。', '线性层能把输入送到哪些输出方向、LoRA 更新能覆盖哪些权重变化，都可从列空间看。', '输出维度大不代表列空间大；rank 决定真正可达的独立方向数。'),
    legacy('线性代数与张量', 'Null Space 零空间', ['kernel'], '被矩阵压到 0 的所有输入方向，表示变换完全丢掉的信息。', '\\(\\operatorname{Null}(A)=\\{x:Ax=0\\}\\)。', '分析投影、低秩压缩、不可逆变换、特征退化和表示丢失。', '零空间不是数值小的方向；它严格表示输出为零，实际模型还要区分近似零和数值阈值。'),
    legacy('线性代数与张量', 'Change of Basis 基变换', ['coordinate change'], '同一个向量或变换换一套坐标尺来描述，对象不变，坐标改变。', '若基矩阵为 \\(P\\)，坐标可写作 \\([x]_B=P^{-1}x\\)。', '表示空间对齐、PCA 坐标、RoPE 的二维旋转块、跨模型 embedding 对齐都涉及换基。', '换基不是改变语义对象本身；混淆对象和坐标会导致错误解释。'),
    legacy('线性代数与张量', 'Similarity Transform 相似变换', ['similar matrix'], '同一个线性变换在不同基底下的矩阵表示。', '\\(A_B=P^{-1}AP\\)。', '谱分析、对角化、Hessian/协方差换坐标和表示空间重参数化。', '相似矩阵共享特征值，但矩阵元素会变；不要把元素级差异直接解释成变换本质改变。'),
    legacy('线性代数与张量', 'Diagonal Matrix 对角矩阵', ['diag'], '只沿坐标轴分别缩放，各方向之间没有混合。', '\\(D=\\operatorname{diag}(d_1,\\dots,d_n)\\)，\\((Dx)_i=d_ix_i\\)。', '特征分解、SVD 奇异值、归一化缩放、门控和逐通道 scale。', '对角矩阵的“简单”依赖当前基底；换一个基后同一变换可能不再对角。'),
    legacy('线性代数与张量', 'Rotation / Reflection / Shear 基本线性变换', ['rotation', 'reflection', 'shear'], '旋转、反射和剪切是理解矩阵如何移动空间的基本动作。', '二维旋转 \\(R_\\theta=\\begin{bmatrix}\\cos\\theta&-\\sin\\theta\\\\sin\\theta&\\cos\\theta\\end{bmatrix}\\)，剪切如 \\(\\begin{bmatrix}1&s\\\\0&1\\end{bmatrix}\\)。', 'RoPE 使用二维旋转块；正交初始化、表示对齐和几何可视化常借助这些基本变换。', '旋转保持长度，剪切改变角度，反射翻转方向；不能把所有正交或非正交变换混成同一类。'),
    legacy('线性代数与张量', '转置矩阵', ['transpose', 'A^T'], '把矩阵的行和列互换，是读 attention 与线性代数公式时最常见的形状操作。', '\\((A^\\top)_{ij}=A_{ji}\\)，若 \\(A\\in\\mathbb{R}^{m\\times n}\\)，则 \\(A^\\top\\in\\mathbb{R}^{n\\times m}\\)。', 'attention 中的 \\(QK^\\top\\)、批量矩阵乘法、协方差矩阵构造都依赖转置。', '转置只改变轴顺序，不等于求逆；在代码里还要区分 view、copy 和 memory layout。'),
    legacy('线性代数与张量', '单位矩阵', ['identity matrix', 'I'], '线性变换里的“什么都不做”，像数字乘法中的 1。', '\\(Ix=x\\)，对角线上为 1、其他位置为 0。', '残差连接、正则化项、协方差矩阵、二阶近似和初始化分析中经常出现。', '单位矩阵的尺寸必须和作用对象匹配；\\(I_d\\) 和 \\(I_T\\) 不是同一个对象。'),
    legacy('线性代数与张量', '逆矩阵', ['inverse matrix', 'A^-1'], '如果一个线性变换能被完全还原，它就有逆矩阵。', '\\(AA^{-1}=A^{-1}A=I\\)。', '最小二乘、协方差白化、自然梯度和二阶方法中会出现逆或近似逆。', '深度学习里很少直接求大矩阵逆；通常用分解、迭代解线性方程或近似。'),
    legacy('线性代数与张量', '行列式', ['determinant', 'det'], '衡量线性变换对体积的缩放，以及方向是否翻转。', '\\(\\det(A)\\)；若 \\(\\det(A)=0\\)，矩阵不可逆。', 'Normalizing Flow 的 change-of-variables、Jacobian determinant、可逆模型 likelihood。', '行列式对大矩阵数值很敏感；实际常计算 log-determinant。'),
    legacy('线性代数与张量', 'Tensor 张量', ['multi-dimensional array'], '多维数组；batch、序列、head、通道等维度叠在一起就是张量。', '\\(X\\in\\mathbb{R}^{B\\times T\\times d}\\) 表示 batch、长度和 hidden size。', '深度学习框架中的输入、激活、梯度和缓存几乎都是张量。', '先读 shape 再读公式；不标 shape 的推导很容易误解。'),
    legacy('线性代数与张量', 'Shape 形状', ['dimension', 'axis'], '描述张量每个轴有多长，是调试模型的第一语言。', '例如 \\([B,T,d]\\to[B,H,T,d_h]\\)，其中 \\(d=H d_h\\)。', 'attention、KV cache、MoE routing、batch packing 都依赖 shape。', 'shape 对了不代表语义对；mask、transpose 和 broadcasting 仍可能错。'),
    legacy('线性代数与张量', 'Dot Product 点积', ['inner product'], '衡量两个向量方向是否相近，也能看作加权求和。', '\\(x^\\top y=\\sum_i x_i y_i\\)。', 'attention 用 \\(QK^\\top\\) 计算 token 相关性。', '点积受向量长度影响；需要区分 dot product 与 cosine similarity。'),
    legacy('线性代数与张量', 'Matrix Multiplication 矩阵乘法', ['matmul', 'GEMM'], '批量做点积，或把一批向量同时做线性变换。', '若 \\(A\\in\\mathbb{R}^{m\\times n},B\\in\\mathbb{R}^{n\\times p}\\)，则 \\(AB\\in\\mathbb{R}^{m\\times p}\\)。', 'attention score、MLP、logits、projection 都是矩阵乘法密集区。', 'FLOPs 多不一定慢；真实速度还受 memory bandwidth 和 kernel layout 影响。'),
    legacy('线性代数与张量', 'Vector Space 向量空间', ['representation space'], '模型内部表示所在的高维空间，语义和风格被编码成方向、子空间或流形。', '向量集合对加法和标量乘法封闭，就构成向量空间。', 'embedding space、residual stream、representation probing。', '类比方向只是经验现象，不是所有语义都线性可分。'),
    legacy('线性代数与张量', 'Basis 基', ['coordinate system'], '一组坐标轴；有了基，空间中的向量可以用坐标表示。', '线性无关且能张成整个空间的向量组叫基。', '模型隐空间有隐式坐标系，但通常不可直接解释。', '不要把 embedding 维度当成固定可解释属性。'),
    legacy('线性代数与张量', 'Projection 投影', ['subspace projection'], '看一个向量在某个方向或子空间上有多少成分。', '\\(\\operatorname{proj}_u(x)=(x^\\top u)u\\)，其中 \\(u\\) 为单位向量。', '\\(W_Q,W_K,W_V\\) 可看作把 hidden state 投到不同子空间。', '投影解释依赖方向选择；方向本身若不可靠，解释也不可靠。'),
    legacy('线性代数与张量', 'Norm 范数', ['L1', 'L2', 'gradient norm'], '衡量向量或矩阵大小。', '\\(\\|x\\|_2=\\sqrt{\\sum_i x_i^2}\\)，\\(\\|x\\|_1=\\sum_i |x_i|\\)。', 'gradient clipping、weight decay、embedding norm、normalization。', '范数大不一定坏；要结合训练阶段、层位置和尺度比较。'),
    legacy('线性代数与张量', 'Cosine Similarity 余弦相似度', ['cos sim'], '只看方向相似度，弱化长度影响。', '\\(\\cos\\theta=\\frac{x^\\top y}{\\|x\\|\\|y\\|}\\)。', 'RAG 检索、embedding 相似度、聚类和表示分析。', '高维空间中随机向量常近似正交；小差异也可能有意义。'),
    legacy('线性代数与张量', '矩阵迹', ['trace', 'tr'], '矩阵对角线元素之和，常把矩阵表达式变成标量。', '\\(\\operatorname{tr}(A)=\\sum_i A_{ii}\\)，且 \\(\\operatorname{tr}(ABC)=\\operatorname{tr}(BCA)\\)。', '矩阵求导、二次型、协方差、Hessian 近似和 loss 推导。', 'trace trick 是推导工具；不要把它误解成模型结构。'),
    legacy('线性代数与张量', '特征值', ['eigenvalue', 'spectrum'], '矩阵沿某个特殊方向只拉伸或压缩多少倍，这个倍数就是特征值。', '若 \\(Av=\\lambda v\\)，则 \\(\\lambda\\) 是特征值。', 'Hessian 曲率、权重谱、稳定性、PCA 和表示主方向分析。', '只有满足条件的矩阵才有良好谱解释；神经网络权重谱更多是诊断线索。'),
    legacy('线性代数与张量', '特征向量', ['eigenvector'], '被矩阵作用后方向不变的向量，只是长度按特征值变化。', '\\(Av=\\lambda v\\)，其中 \\(v\\ne0\\)。', 'PCA 主方向、二阶优化、表示空间方向和动力系统稳定性。', '特征向量的符号和尺度通常不唯一；不要过度解释单个坐标。'),
    legacy('线性代数与张量', '特征分解', ['eigendecomposition'], '把矩阵拆成特征向量坐标系和对应特征值。', '可对角化时 \\(A=V\\Lambda V^{-1}\\)。', 'PCA、协方差分析、Hessian 近似和线性动力系统。', '不是所有矩阵都能稳定对角化；非对称矩阵可能出现复特征值或病态分解。'),
    legacy('线性代数与张量', 'SVD 分解', ['奇异值分解', 'singular value decomposition'], '把任意矩阵拆成输入方向、奇异值强度和输出方向。', '\\(A=U\\Sigma V^\\top\\)。', '低秩近似、LoRA 直觉、模型压缩、PCA、embedding 分析。', '保留大奇异值能保留线性能量，但不等于保留任务能力。'),
    legacy('线性代数与张量', '矩阵秩', ['rank', 'matrix rank'], '矩阵中真正独立的信息维度，等价于非零奇异值个数。', '\\(\\operatorname{rank}(BA)\\le r\\)。', 'LoRA 用低秩更新 \\(\\Delta W=BA\\) 限制微调自由度，也用于判断表示是否退化。', 'rank 太低会欠拟合；rank 太高会增加成本和过拟合风险。'),
    legacy('线性代数与张量', 'Low-rank Approximation 低秩近似', ['low rank'], '用少数主要方向近似原矩阵。', '\\(A\\approx U_r\\Sigma_rV_r^\\top\\)。', 'LoRA、adapter、权重压缩、表示降维。', '低秩是工程假设，不是所有层和任务都同样适合。'),
    legacy('线性代数与张量', '正定矩阵', ['positive definite', 'PSD'], '在任意非零方向上二次型都为正，表示“曲率向上”或协方差有效。', '正定：\\(x^\\top Ax>0\\)。半正定：\\(x^\\top Ax\\ge0\\)。', '协方差矩阵、Hessian 曲率、二阶优化、Gaussian 分布参数。', '数值上接近奇异的半正定矩阵会让求逆和 Cholesky 分解不稳定。'),
    legacy('线性代数与张量', '正交矩阵', ['orthogonal matrix'], '列向量彼此垂直且长度为 1 的矩阵，会保持向量长度和角度。', '\\(Q^\\top Q=I\\)。', 'SVD/QR 分解、初始化、旋转位置编码和表示空间变换。', '正交不代表元素稀疏；它保持几何结构，但不保证语义可解释。'),
    legacy('线性代数与张量', 'PCA 主成分分析', ['principal component analysis'], '寻找数据方差最大的正交方向，用少数方向概括主要变化。', '对协方差矩阵做特征分解，或对中心化数据做 SVD。', 'embedding 可视化、表示压缩、激活分析、数据预处理。', 'PCA 保留的是方差，不一定保留任务判别信息；低维图只能作辅助证据。'),
    legacy('线性代数与张量', 'LDA 线性判别分析', ['linear discriminant analysis'], '寻找最能区分类别的线性投影，强调类间远、类内近。', '典型目标是最大化 \\(\\frac{w^\\top S_B w}{w^\\top S_W w}\\)。', '分类特征分析、表示可分性诊断、小样本线性基线。', '这里指 Linear Discriminant Analysis，不是主题模型 Latent Dirichlet Allocation；它依赖类别标签和分布假设。'),
    legacy('线性代数与张量', 'High-dimensional Geometry 高维几何', ['curse of dimensionality'], '高维空间的距离、角度、体积直觉和二维三维不同。', '随机高维向量常接近正交，概率质量常集中在薄壳附近。', 'embedding、nearest neighbor、attention、表示空间分析。', '低维可视化会扭曲高维结构；t-SNE/UMAP 图不能单独当证据。'),

    legacy('微积分与自动微分', 'Derivative 导数', ['slope'], '描述输入变一点，输出变化得多快。', '\\(f\\prime(x)=\\lim_{h\\to0}\\frac{f(x+h)-f(x)}{h}\\)。', '优化时用导数决定参数该往哪里调。', '数值差分不是自动微分；大模型不靠逐参数差分训练。'),
    legacy('微积分与自动微分', 'Gradient 梯度', ['nabla', 'vanishing gradient', '梯度消失', '梯度爆炸'], '多变量函数上升最快的方向；最小化 loss 时沿负梯度走。', '\\(\\nabla_x f=[\\partial f/\\partial x_1,\\dots,\\partial f/\\partial x_d]^\\top\\)。', '训练中每个参数都有对应梯度。', '梯度方向是局部信息；非凸问题里不保证一步到全局最优。'),
    legacy('微积分与自动微分', 'Jacobian 雅可比矩阵', ['J'], '向量函数每个输出对每个输入的敏感度表。', '若 \\(f:\\mathbb{R}^n\\to\\mathbb{R}^m\\)，则 \\(J_{ij}=\\partial f_i/\\partial x_j\\)。', 'softmax 导数、局部敏感性、对抗扰动、JVP/VJP。', '完整 Jacobian 很大，实际训练通常不显式构造。'),
    legacy('微积分与自动微分', 'Hessian 海森矩阵', ['second order'], 'loss 曲面的二阶曲率。', '\\(H_{ij}=\\partial^2 f/(\\partial x_i\\partial x_j)\\)。', 'sharpness、二阶优化、pruning、量化敏感度。', '大模型 Hessian 不能完整存储；通常用近似或向量积。'),
    legacy('微积分与自动微分', 'Chain Rule 链式法则', ['composition derivative'], '复杂函数由简单函数串起来，总导数就是局部导数按路径相乘。', '若 \\(y=f(g(x))\\)，则 \\(dy/dx=(df/dg)(dg/dx)\\)。', '反向传播就是链式法则在计算图上的动态规划。', '链路很深时梯度可能消失或爆炸。'),
    legacy('微积分与自动微分', 'Computational Graph 计算图', ['graph'], '把一次前向计算拆成操作节点和数据边。', '节点保存局部操作，反向时按拓扑逆序传播梯度。', 'PyTorch/JAX/TensorFlow 的 autodiff 都依赖计算图或 trace。', '动态图和静态图影响工程调试，但数学目标相同。'),
    legacy('微积分与自动微分', 'Backpropagation 反向传播', ['backprop'], '把输出 loss 的责任从后往前分配给每个参数。', '本质是链式法则 + 中间量复用。', '训练 Transformer、VAE、Diffusion、Reward Model 都依赖 backprop。', '反向传播不是优化器；它只给梯度，Adam/SGD 才更新参数。'),
    legacy('微积分与自动微分', 'Automatic Differentiation 自动微分', ['autodiff'], '把基本操作的精确导数组合起来，不是符号化推导也不是数值差分。', '常见模式是 reverse-mode 和 forward-mode。', '深度学习训练主要用 reverse-mode，因为标量 loss 对大量参数求梯度。', '自动微分会忠实计算你写的程序；程序语义错，它也不会替你纠正。'),
    legacy('微积分与自动微分', 'VJP 向量-雅可比积', ['vector-Jacobian product'], '把上游梯度乘上局部 Jacobian，是反向模式的核心操作。', '\\(v^\\top J\\)。', 'backprop 实际上传的是 VJP，而不是完整 Jacobian。', '理解 VJP 有助于看懂自定义 backward 和 memory-efficient attention。'),
    legacy('微积分与自动微分', 'JVP 雅可比-向量积', ['Jacobian-vector product'], '看输入沿某个方向变化时输出如何变化。', '\\(Jv\\)，其中 \\(v\\) 是输入空间中的扰动方向。', 'forward-mode autodiff、敏感性分析、隐式层和二阶近似。', '当输出维度小而参数很多时，JVP 不如 VJP 适合训练。'),
    legacy('微积分与自动微分', 'Trace Trick 迹技巧', ['trace'], '用 trace 的循环性质改写矩阵求导，让形状对齐。', '\\(\\operatorname{tr}(ABC)=\\operatorname{tr}(BCA)\\)。', '推导线性层、二次型、attention 相关梯度时常用。', 'trace trick 是推导工具，不是新的训练算法。'),
    legacy('微积分与自动微分', 'Softmax Gradient', ['softmax derivative'], 'softmax 输出彼此耦合，一个 logit 变化会影响所有类别概率。', '\\(\\partial p_i/\\partial z_j=p_i(\\delta_{ij}-p_j)\\)。', 'cross entropy + softmax 会简化成 \\(p-y\\)。', '不要把每个类别当独立 sigmoid；多分类 softmax 有归一化竞争。'),

    legacy('概率统计与信息论', 'Random Variable 随机变量', ['X'], '不是一个固定数，而是一次采样可能得到的结果。', '\\(X:\\Omega\\to\\mathcal{X}\\)。', 'token、图像 latent、reward、benchmark score 都可视作随机变量。', '随机变量和一次观测值不同；评测必须考虑抽样波动。'),
    legacy('概率统计与信息论', 'Probability Distribution 概率分布', ['p(x)', 'q(x)'], '描述不同结果出现的可能性。', '离散时 \\(\\sum_x p(x)=1\\)，连续时 \\(\\int p(x)dx=1\\)。', '语言模型输出 token 分布，Diffusion 定义噪声分布和反向分布。', '连续密度不是单点概率；密度可以大于 1。'),
    legacy('概率统计与信息论', '概率质量函数', ['PMF', 'probability mass function'], '离散随机变量每个取值的概率表。', '\\(p_X(x)=P(X=x)\\)，且 \\(\\sum_x p_X(x)=1\\)。', 'token 分布、分类标签、采样候选集合和离散 latent code。', 'PMF 的单点值就是概率；这和连续变量的 PDF 不同。'),
    legacy('概率统计与信息论', '概率密度函数', ['PDF', 'probability density function'], '连续随机变量的密度函数，区间面积才是概率。', '\\(P(a\\le X\\le b)=\\int_a^b f_X(x)dx\\)，且 \\(\\int f_X(x)dx=1\\)。', 'Gaussian latent、diffusion noise、连续 embedding 扰动、flow likelihood。', 'PDF 值可以大于 1；单点 \\(P(X=x)\\) 通常为 0。'),
    legacy('概率统计与信息论', '累积分布函数', ['CDF', 'cumulative distribution function'], '随机变量落在某个阈值以下的概率。', '\\(F_X(x)=P(X\\le x)\\)。连续情形下 \\(F\\prime(x)=f(x)\\)。', '分位数、阈值选择、校准曲线、统计检验和采样变换。', 'CDF 单调不减；不要把密度峰值和累计概率混为一谈。'),
    legacy('概率统计与信息论', 'Categorical Distribution 类别分布', ['multinomial one draw'], '从有限类别中选一个结果的分布。', '\\(P(X=i)=p_i,\\sum_i p_i=1\\)。', 'next-token sampling 从 vocabulary 上的 categorical distribution 采样。', 'top-k/top-p 改的是采样候选分布，不是模型参数。'),
    legacy('概率统计与信息论', '高斯分布', ['Gaussian distribution', 'normal distribution', '高斯公式'], '由均值和方差控制的钟形连续分布，是生成模型中最常见的可计算基准分布。', '\\(f(x)=\\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)\\)。', 'VAE prior、Diffusion noising、初始化、不确定性估计和 Gaussian transition。', '高斯方便不代表真实数据高斯；很多方法只是把它当可计算 base distribution。'),
    legacy('概率统计与信息论', '高斯积分', ['Gaussian integral'], '高斯密度能归一化的关键积分，也是高斯公式前面的常数来源。', '\\(\\int_{-\\infty}^{\\infty}e^{-x^2}dx=\\sqrt{\\pi}\\)。', '解释 Gaussian PDF 的归一化常数、多维高斯和 diffusion 噪声分布。', '它是数学恒等式，不是模型训练目标；不要把归一化常数和可学习参数混淆。'),
    legacy('概率统计与信息论', 'Conditional Probability 条件概率', ['p(y|x)'], '已知一个条件后，另一个事件的概率。', '\\(p(a|b)=p(a,b)/p(b)\\)。', 'LLM 建模 \\(p(x_t|x_{<t})\\)，条件生成建模 \\(p(x|c)\\)。', '条件改变会改变分布；不要把无条件指标直接套到条件任务。'),
    legacy('概率统计与信息论', '全概率公式', ['law of total probability'], '把一个事件按互斥完备条件拆开求概率。', '\\(p(x)=\\sum_z p(x|z)p(z)\\)；连续时为 \\(p(x)=\\int p(x|z)p(z)dz\\)。', 'latent variable model、VAE evidence、mixture model 和边缘似然。', '隐变量很多或连续时求和/积分常不可算，需要采样或变分近似。'),
    legacy('概率统计与信息论', '贝叶斯公式', ['Bayes rule', 'Bayes theorem'], '用先验和似然反推出后验。', '\\(p(z|x)=\\frac{p(x|z)p(z)}{p(x)}\\)。', 'VAE 后验推断、MAP、Bayesian view of learning。', '分母 evidence 常不可算，才需要变分推断等近似。'),
    legacy('概率统计与信息论', '先验分布', ['prior'], '看到数据之前对变量或参数的假设分布。', '\\(p(z)\\) 或 \\(p(\\theta)\\)。', 'VAE latent prior、Bayesian regularization、MAP、扩散模型的噪声起点。', '先验不是装饰项；它会真实改变后验和生成空间。'),
    legacy('概率统计与信息论', '似然函数', ['likelihood'], '在参数或隐变量给定时，观测数据出现的可能性。', '\\(L(\\theta)=p(D|\\theta)\\)。', 'MLE、VAE decoder likelihood、flow likelihood、分类 cross entropy。', '似然是关于参数的函数，不是参数本身的概率分布。'),
    legacy('概率统计与信息论', '后验分布', ['posterior'], '看到数据之后对隐变量或参数的更新认识。', '\\(p(z|x)\\propto p(x|z)p(z)\\)。', 'VAE encoder 近似后验、Bayesian inference、uncertainty estimation。', '后验通常不可精确计算；近似后验的选择会带来 posterior gap。'),
    legacy('概率统计与信息论', 'Expectation 期望', ['expected value'], '随机变量长期平均值。', '\\(\\mathbb{E}_{x\\sim p}[f(x)]=\\int f(x)p(x)dx\\)。', '训练 loss 通常是对数据和噪声的期望，用 minibatch 估计。', '写期望必须写采样源；否则不知道优化目标来自哪里。'),
    legacy('概率统计与信息论', 'Variance 方差', ['var'], '随机变量围绕均值波动的程度。', '\\(\\operatorname{Var}(X)=\\mathbb{E}(X-\\mathbb{E}X)^2\\)。', '梯度噪声、评测分数波动、采样随机性。', '平均分相同但方差不同，工程风险可能完全不同。'),
    legacy('概率统计与信息论', 'Covariance 协方差', ['cov'], '两个随机变量共同变化的程度。', '\\(\\operatorname{Cov}(X,Y)=\\mathbb{E}[(X-\\mu_X)(Y-\\mu_Y)]\\)。', '表示分析、PCA、特征相关、梯度相关。', '相关不等于因果；协方差受尺度影响。'),
    legacy('概率统计与信息论', 'Maximum Likelihood Estimation 最大似然估计', ['MLE'], '选择参数，让观测数据在模型下尽可能可能。', '\\(\\theta^*=\\arg\\max_\\theta\\sum_i\\log p_\\theta(x_i)\\)。', 'next-token training、Flow exact likelihood、分类 CE 都可从 MLE 看。', 'MLE 优化的是数据概率，不直接等于人类偏好或真实可用性。'),
    legacy('概率统计与信息论', 'MAP 最大后验估计', ['maximum a posteriori'], '在 MLE 上加入参数先验。', '\\(\\theta^*=\\arg\\max_\\theta \\log p(D|\\theta)+\\log p(\\theta)\\)。', '正则化和 weight decay 可从先验视角理解。', '先验选择会影响解；不是客观真理。'),
    legacy('概率统计与信息论', 'Monte Carlo 蒙特卡洛', ['MC'], '用随机样本平均近似难算的期望或积分。', '\\(\\mathbb{E}[f(X)]\\approx \\frac1N\\sum_i f(x_i)\\)。', 'minibatch training、采样评测、ELBO、policy gradient。', '样本量不足会带来高方差；一次采样结论不稳。'),
    legacy('概率统计与信息论', 'Sampling 采样', ['decode', 'draw'], '从一个分布中抽取具体样本。', '离散分布按概率抽 token，连续分布可用重参数化或 MCMC 等。', 'LLM decoding、Diffusion reverse process、VAE latent sampling。', '采样策略会影响输出质量，但不等同于训练目标。'),
    legacy('概率统计与信息论', 'Entropy 熵', ['uncertainty'], '分布的不确定性或平均编码长度。', '\\(H(p)=-\\mathbb{E}_{p}\\log p(x)\\)。', '语言模型分布尖锐程度、探索、多样性。', '高熵不一定好；可能表示多样，也可能表示不确定或混乱。'),
    legacy('概率统计与信息论', 'Cross-Entropy 交叉熵', ['CE'], '用模型分布编码真实样本的平均代价。', '\\(H(p,q)=-\\mathbb{E}_{x\\sim p}\\log q(x)\\)。', '分类训练和 next-token prediction 的核心 loss。', 'CE 低通常有用，但不能单独保证事实性和偏好质量。'),
    legacy('概率统计与信息论', 'KL Divergence KL 散度', ['KL', 'relative entropy'], '衡量用 q 近似 p 多付出的编码代价，方向敏感。', '\\(D_{KL}(p\\Vert q)=\\mathbb{E}_p\\log\\frac{p(x)}{q(x)}\\)。', 'MLE、VAE regularization、distillation、RLHF/DPO KL 约束。', 'KL 不对称；\\(D_{KL}(p\\Vert q)\\) 和 \\(D_{KL}(q\\Vert p)\\) 行为不同。'),
    legacy('概率统计与信息论', 'Mutual Information 互信息', ['MI'], '一个变量告诉你另一个变量多少信息。', '\\(I(X;Y)=D_{KL}(p(x,y)\\Vert p(x)p(y))\\)。', '表示学习、信息瓶颈、多模态对齐。', '互信息难估计；神经估计器可能偏差很大。'),
    legacy('概率统计与信息论', 'Perplexity 困惑度', ['PPL'], 'cross entropy 的指数形式，表示平均每步有效候选数。', '\\(\\operatorname{PPL}=\\exp(H)\\)。', '语言模型预训练和验证集指标。', 'PPL 低不保证对话体验好；评测任务和分布很重要。'),
    legacy('概率统计与信息论', 'Bits 与 Nats', ['log base'], '信息量单位，取决于 log 的底。', '自然对数单位是 nats，\\(\\log_2\\) 单位是 bits。', '论文中 CE/KL 可能用不同单位。', '比较数值前先确认 log base，否则差一个 \\(\\ln2\\) 因子。'),

    legacy('优化与数值计算', 'Objective Function 目标函数', ['objective'], '真正想优化的数学目标。', '\\(\\min_\\theta J(\\theta)\\) 或 \\(\\max_\\theta J(\\theta)\\)。', 'MLE、ELBO、reward maximization、DPO objective。', '工程 loss 可能只是目标的 surrogate，不一定严格等价。'),
    legacy('优化与数值计算', 'Loss Function 损失函数', ['loss'], '训练时实际计算和反向传播的标量。', '常见形式包括 NLL、MSE、hinge、policy loss。', '所有模型训练最终都要把监督信号变成可微 loss。', 'loss 降低不一定表示最终任务指标提升。'),
    legacy('优化与数值计算', 'SGD 随机梯度下降', ['stochastic gradient descent'], '用小批量估计梯度并更新参数。', '\\(\\theta_{t+1}=\\theta_t-\\eta g_t\\)。', '深度学习优化的基本形式。', '学习率、batch size 和梯度噪声共同决定稳定性。'),
    legacy('优化与数值计算', 'Mini-batch', ['batch'], '每次训练用一小批样本估计总体梯度。', '\\(g_t=\\frac1B\\sum_{i=1}^B\\nabla_\\theta \\ell_i\\)。', '预训练、SFT、Diffusion training 都用 minibatch。', 'batch 变大通常要调学习率；吞吐提升不等于样本效率提升。'),
    legacy('优化与数值计算', 'Momentum 动量', ['heavy ball'], '把过去梯度方向累积起来，减少抖动。', '\\(v_t=\\mu v_{t-1}+g_t\\)。', 'SGD with momentum 和 Adam 的一阶矩都含动量思想。', '动量过强可能越过窄谷或放大不稳定。'),
    legacy('优化与数值计算', 'Adam', ['adaptive moment'], '用一阶矩和二阶矩自适应缩放梯度。', '\\(m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t\\)，\\(v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2\\)。', '大模型预训练和微调常用优化器基础。', 'Adam 的 L2 regularization 和 AdamW 的 weight decay 不等价。'),
    legacy('优化与数值计算', 'AdamW', ['decoupled weight decay'], '把 weight decay 从梯度自适应缩放中解耦。', '\\(\\theta\\leftarrow \\theta-\\eta \\hat m/(\\sqrt{\\hat v}+\\epsilon)-\\eta\\lambda\\theta\\)。', 'Transformer 预训练、SFT、LoRA 微调的常见选择。', 'weight decay 不是越大越好；embedding、norm 参数常需特殊处理。'),
    legacy('优化与数值计算', 'Learning Rate Schedule 学习率调度', ['warmup', 'cosine'], '控制每一步更新幅度如何随训练变化。', '常见 warmup + cosine decay。', '大模型训练前期 warmup 避免不稳定，后期 decay 改善收敛。', '换 batch size、数据或 optimizer 后原 schedule 不一定适用。'),
    legacy('优化与数值计算', 'Weight Decay 权重衰减', ['regularization'], '惩罚权重过大，或每步把权重向 0 拉一点。', 'AdamW 中是 decoupled decay。', '控制模型复杂度和泛化，常用于 transformer 权重。', '对 bias、LayerNorm/RMSNorm 参数常不做 decay。'),
    legacy('优化与数值计算', 'Gradient Clipping 梯度裁剪', ['clip grad norm'], '梯度过大时按阈值缩放，防止一步更新炸掉。', '若 \\(\\|g\\|>c\\)，令 \\(g\\leftarrow c g/\\|g\\|\\)。', 'LLM 训练、RLHF、Diffusion 中常用稳定手段。', '裁剪会改变真实梯度方向/尺度；频繁触发说明训练设置可能有问题。'),
    legacy('优化与数值计算', 'Non-convex Optimization 非凸优化', ['loss landscape'], 'loss 曲面有很多鞍点、局部谷和平坦区域。', '神经网络目标通常非凸。', '解释为什么初始化、学习率、归一化和数据顺序影响训练。', '非凸不意味着无法优化；过参数化模型常仍可有效训练。'),
    legacy('优化与数值计算', 'Regularization 正则化', ['regularizer'], '约束模型不要只记训练集，提高泛化。', '目标常写 \\(\\mathcal{L}+\\lambda R(\\theta)\\)。', 'weight decay、dropout、data augmentation、KL penalty。', '正则太强会欠拟合；正则选择要看数据和任务。'),
    legacy('优化与数值计算', 'Constrained Optimization 约束优化', ['constraint'], '在满足约束条件下优化目标。', '\\(\\max f(\\theta)\\;\\text{s.t.}\\;g(\\theta)\\le c\\)。', 'RLHF 中常见 KL-constrained reward maximization。', '约束常通过 penalty 或 dual variable 近似实现，需监控是否真的满足。'),
    legacy('优化与数值计算', 'Duality 对偶', ['Lagrangian'], '把约束问题转成带乘子的无约束或对偶问题。', '\\(\\mathcal{L}(x,\\lambda)=f(x)+\\lambda g(x)\\)。', 'KL 约束 RL、最大熵模型、SVM 等都可用对偶视角。', '强对偶需要条件；非凸问题里对偶解释要谨慎。'),
    legacy('优化与数值计算', 'Floating Point 浮点数', ['float'], '计算机用有限位数近似实数。', 'FP32/FP16/BF16/FP8 有不同指数和尾数位。', '训练、推断、量化和 kernel 选择都受浮点格式影响。', '低精度会带来 overflow、underflow 和舍入误差。'),
    legacy('优化与数值计算', 'FP32 / FP16 / BF16', ['precision'], '不同浮点格式在范围和精度之间取舍。', 'BF16 指数范围接近 FP32，尾数更短；FP16 尾数稍多但范围小。', '大模型训练常用 BF16，推断可能用 FP16/FP8/INT4。', '硬件支持决定收益；格式选择不能只看位宽。'),
    legacy('优化与数值计算', 'Mixed Precision 混合精度', ['AMP'], '用低精度加速大部分计算，同时保留关键状态的稳定性。', '常见做法是低精度 forward/backward，FP32 optimizer state。', '大模型训练吞吐和显存优化。', '需要处理 loss scaling、归一化和 softmax 稳定性。'),
    legacy('优化与数值计算', 'Quantization 量化', ['INT8', 'INT4'], '用更少 bit 表示权重或激活。', '\\(x\\approx s(q-z)\\)，其中 \\(q\\) 是整数码。', 'LLM 推断、端侧部署、KV cache 压缩。', '量化不一定加速；要看 kernel、硬件、batch 和 dequant 是否融合。'),
    legacy('优化与数值计算', 'Conditioning 条件数', ['ill-conditioned'], '输入小变化导致输出大变化的程度。', '矩阵条件数常为最大奇异值和最小奇异值之比。', '优化稳定性、二阶曲率、归一化和数值误差分析。', '病态问题会让梯度方向不可靠或收敛很慢。'),
    legacy('优化与数值计算', 'FLOPs', ['floating point operations'], '浮点运算次数，衡量计算量。', '矩阵乘法约 \\(2mnp\\) FLOPs。', '模型训练成本、推断理论成本、scaling 估算。', 'FLOPs 不是延迟；memory bandwidth、通信和 kernel overhead 也重要。'),
    legacy('优化与数值计算', 'Memory Bandwidth 显存带宽', ['HBM bandwidth'], '单位时间能从显存搬多少数据。', 'memory time \\(\\approx\\) bytes moved / bandwidth。', 'LLM decode、KV cache、attention kernel 经常受带宽限制。', '参数少不一定快；如果访存不规则仍可能慢。'),
    legacy('优化与数值计算', 'FlashAttention', ['IO-aware attention'], '不近似 attention，而是减少中间矩阵在 HBM 的读写。', '分块计算 softmax attention，保持 exact output。', '长序列 prefill、训练 attention、memory-efficient transformer。', 'FlashAttention 解决的是 IO，不直接降低 \\(O(T^2)\\) 的数学交互结构。'),

    legacy('深度学习机制', 'Neuron 神经元', ['unit'], '线性加权求和后接非线性。', '\\(y=\\phi(w^\\top x+b)\\)。', 'MLP、FFN、分类头的基本构件。', '现代网络更应按层和张量理解，而不是单个神经元神话化。'),
    legacy('深度学习机制', 'MLP 多层感知机', ['FFN'], '多层线性变换和非线性组成的函数近似器。', '\\(\\operatorname{MLP}(x)=W_2\\phi(W_1x)\\)。', 'Transformer block 中的 FFN/MLP 负责逐 token 特征变换。', 'attention 混合 token，MLP 通常在每个 token 位置独立作用。'),
    legacy('深度学习机制', 'Activation Function 激活函数', ['ReLU', 'GELU', 'SiLU'], '引入非线性，否则多层线性仍等价于一层线性。', '常见 \\(\\operatorname{GELU}(x)\\)、\\(\\operatorname{SiLU}(x)=x\\sigma(x)\\)。', 'Transformer MLP、SwiGLU、视觉模型都依赖激活函数。', '激活选择影响梯度和数值范围，不能只看表达式。'),
    legacy('深度学习机制', 'Initialization 初始化', ['init'], '训练开始时如何设置参数。', '常按 fan-in/fan-out 控制方差。', '深层网络稳定训练依赖合适初始化和归一化。', '坏初始化会导致激活或梯度爆炸/消失。'),
    legacy('深度学习机制', 'Residual Connection 残差连接', ['skip connection'], '让层学习增量，而不是每层重写全部表示。', '\\(y=x+F(x)\\)。', 'Transformer residual stream、ResNet、Diffusion U-Net。', '残差路径会累积信息，也会让 scale 管理更重要。'),
    legacy('深度学习机制', 'Normalization 归一化', ['norm'], '控制激活尺度，改善优化稳定性。', '常见 LayerNorm、RMSNorm、BatchNorm。', 'Transformer 几乎每层都有 norm。', 'norm 放在 attention/MLP 前后会影响训练动态。'),
    legacy('深度学习机制', 'LayerNorm', ['LN'], '对单个样本的 hidden dimension 做均值方差归一化。', '\\(\\operatorname{LN}(x)=\\gamma\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}+\\beta\\)。', 'Transformer block 的稳定训练核心组件。', 'LayerNorm 不依赖 batch 统计，和 BatchNorm 不同。'),
    legacy('深度学习机制', 'RMSNorm', ['root mean square norm'], '只用均方根缩放，不减均值。', '\\(\\operatorname{RMSNorm}(x)=g\\,x/\\sqrt{\\frac1d\\sum_i x_i^2+\\epsilon}\\)。', '许多现代 LLM 使用 RMSNorm 简化计算。', 'RMSNorm 改变的是尺度控制方式，不是注意力机制。'),
    legacy('深度学习机制', 'Dropout', ['random mask'], '训练时随机丢弃部分激活，减少共适应。', '\\(\\tilde{x}=m\\odot x/(1-p)\\)。', '小模型和某些训练设置中用于正则化。', '大规模 LLM 预训练未必大量使用 dropout；要看数据规模和过拟合风险。'),

    legacy('Transformer 与 LLM', 'Token', ['subword'], '模型处理的离散单位，可以是词片段、字节、图像 patch code 或音频 code。', '序列 \\(x_1,\\dots,x_T\\) 中每个 \\(x_t\\) 是 token id。', 'LLM 的输入输出都先被 tokenizer 离散化。', 'token 不是自然语言单词；中英文、代码和数字会被不同切分。'),
    legacy('Transformer 与 LLM', 'Tokenizer', ['BPE', 'SentencePiece'], '把原始文本变成 token id，并能把 id 还原成文本。', '学习或定义一个字符串到整数序列的映射。', '决定 vocabulary、上下文长度利用率、数字/多语言处理。', 'tokenizer 改了，模型 embedding 和训练分布也变了。'),
    legacy('Transformer 与 LLM', 'Vocabulary 词表', ['vocab'], '所有可能 token 的集合。', 'logits 维度通常等于 vocab size。', 'LM head 输出每个 token 的 logit。', '词表越大不是必然越好；会影响 embedding 参数和稀有 token。'),
    legacy('Transformer 与 LLM', 'Embedding', ['token embedding'], '把离散 token id 映射到连续向量。', '\\(e_t=E[x_t]\\)，\\(E\\in\\mathbb{R}^{|V|\\times d}\\)。', '模型输入表示、输出权重 tied embedding。', 'embedding 相似不等于语义完全相同；上下文会改变 token 表示。'),
    legacy('Transformer 与 LLM', 'Positional Encoding 位置编码', ['position embedding'], '给模型注入 token 顺序信息。', '可学习绝对位置、相对位置或旋转位置编码。', '没有位置，self-attention 本身对顺序不敏感。', '位置编码决定长度外推能力和长上下文行为。'),
    legacy('Transformer 与 LLM', 'RoPE 旋转位置编码', ['Rotary Position Embedding'], '用旋转把位置信息注入 query/key，使相对位置影响点积。', '对成对维度施加角度随位置变化的旋转。', '现代 LLM 常用 RoPE 支持相对位置建模。', 'RoPE scaling 会影响长上下文稳定性，不是简单把长度改大。'),
    legacy('Transformer 与 LLM', 'Attention', ['self-attention'], '让每个 token 根据相关性从其他 token 聚合信息。', '\\(\\operatorname{softmax}(QK^\\top/\\sqrt{d_k})V\\)。', 'Transformer 的核心上下文混合机制。', 'attention weight 不是完整解释；value 和后续层同样重要。'),
    legacy('Transformer 与 LLM', 'Scaled Dot-product Attention', ['scaled attention'], '点积 attention 加 \\(\\sqrt{d_k}\\) 缩放，避免 logits 方差过大。', '\\(A=\\operatorname{softmax}(QK^\\top/\\sqrt{d_k})\\)。', '标准 Transformer attention。', '缩放项是稳定训练细节，不是可随意删除的常数。'),
    legacy('Transformer 与 LLM', 'Causal Mask', ['autoregressive mask'], '禁止当前位置看到未来 token。', '对未来位置加 \\(-\\infty\\) 后再 softmax。', 'decoder-only LLM 的自回归训练与推断。', 'mask 错会造成数据泄漏，让训练指标虚高。'),
    legacy('Transformer 与 LLM', 'Multi-head Attention 多头注意力', ['MHA'], '把表示拆成多个 head，从不同子空间计算 attention。', '\\(\\operatorname{Concat}(head_1,\\dots,head_H)W_O\\)。', 'Transformer block 的基础模块。', 'head 多不一定更好；GQA/MQA 会减少 KV 成本。'),
    legacy('Transformer 与 LLM', 'Residual Stream', ['stream'], 'Transformer 中层与层之间传递的主表示通道。', '每个子层通常输出 \\(x+F(\\operatorname{Norm}(x))\\)。', 'mechanistic interpretability 常分析 residual stream。', '不同层的信息叠加在同一空间里，解释要考虑后续读出。'),
    legacy('Transformer 与 LLM', 'MLP / FFN', ['feed-forward network'], '对每个 token 的 hidden state 做非线性变换。', '常见 \\(W_{down}\\phi(W_{up}x)\\)。', 'Transformer block 中 attention 后的容量主要来自 FFN。', 'FFN 不直接跨 token 混合；跨 token 主要靠 attention。'),
    legacy('Transformer 与 LLM', 'SwiGLU', ['gated MLP'], '用门控激活提高 MLP 表达能力。', '\\(\\operatorname{SwiGLU}(x)=\\operatorname{SiLU}(xW_g)\\odot xW_u\\)。', 'LLaMA 等模型常用的 FFN 变体。', '参数量和 hidden expansion 会变化，比较模型时要看总计算。'),
    legacy('Transformer 与 LLM', 'Next-token Prediction', ['language modeling'], '给定前缀预测下一个 token。', '\\(\\mathcal{L}=-\\sum_t\\log p_\\theta(x_t|x_{<t})\\)。', 'LLM 预训练核心目标。', '它不是“只会接龙”的贬义词；足够数据和模型下可学到复杂结构。'),
    legacy('Transformer 与 LLM', 'Logits', ['pre-softmax scores'], 'softmax 前的未归一化分数。', '\\(p_i=\\exp(l_i)/\\sum_j\\exp(l_j)\\)。', 'LM head 输出 logits，decoding 再转成采样分布。', 'logit 大小受 temperature、bias、校准影响；不是概率本身。'),
    legacy('Transformer 与 LLM', 'Temperature', ['sampling temperature'], '缩放 logits 控制采样分布尖锐程度。', '\\(p_i(T)=\\operatorname{softmax}(l_i/T)\\)。', '推断时控制多样性和稳定性。', 'temperature 不更新模型；它只是 decoding 策略。'),
    legacy('Transformer 与 LLM', 'Top-k Sampling', ['top k'], '只保留概率最高的 k 个 token 再采样。', '令非 top-k token 概率为 0 后重新归一化。', '控制 LLM 输出的尾部风险。', 'k 太小会单调重复，太大可能引入低质量 token。'),
    legacy('Transformer 与 LLM', 'Top-p / Nucleus Sampling', ['nucleus'], '保留累计概率达到 p 的最小 token 集。', '\\(S=\\min\\{S:\\sum_{i\\in S}p_i\\ge p\\}\\)。', '比 top-k 更自适应于分布尖锐程度。', 'top-p 和 temperature 共同作用，不能孤立调参。'),
    legacy('Transformer 与 LLM', 'KV Cache', ['key value cache'], '推断时保存历史 key/value，避免每生成一个 token 都重算历史。', '每层约 \\(2LTHd_h\\) 个数。', 'LLM decode 加速和长上下文 serving 的核心状态。', 'KV cache 会成为显存和带宽瓶颈。'),
    legacy('Transformer 与 LLM', 'LoRA', ['low-rank adaptation'], '冻结原权重，只训练低秩增量。', '\\(W\\prime=W+BA\\)，\\(B\\in\\mathbb{R}^{d\\times r},A\\in\\mathbb{R}^{r\\times k}\\)。', '参数高效微调、风格/任务适配。', 'LoRA rank、target modules 和 alpha 都会影响容量与稳定性。'),
    legacy('Transformer 与 LLM', 'Adapter', ['adapter tuning'], '在模型层中插入小模块，只训练这些模块。', '常见 bottleneck \\(x+W_2\\phi(W_1x)\\)。', '参数高效迁移学习。', 'adapter 会增加推断路径；和 LoRA 的部署合并方式不同。'),
    legacy('Transformer 与 LLM', 'Prefix Tuning / Prompt Tuning', ['soft prompt'], '训练一组连续向量作为额外上下文。', '学习 prefix key/value 或 soft prompt embedding。', '低成本任务适配，不改主模型参数。', '容量有限，对复杂任务可能不如 LoRA/SFT。'),
    legacy('Transformer 与 LLM', 'MoE Mixture of Experts', ['sparse experts'], '用路由器为每个 token 选择少数专家网络。', '输出常为 \\(\\sum_{e\\in topk}g_e(x)E_e(x)\\)。', '用更高总参数量换每 token 较低激活计算。', '瓶颈常在负载均衡、通信和专家容量。'),
    legacy('Transformer 与 LLM', 'Scaling Laws', ['scaling'], '模型大小、数据量、计算量和 loss 之间的经验规律。', '常见幂律关系 \\(L(C)\\approx aC^{-b}+c\\)。', '预训练预算分配和模型规模选择。', 'scaling law 是特定分布和设定下的经验外推，不是物理定律。'),

    legacy('生成模型', 'Latent Variable 潜变量', ['z'], '观测数据背后未直接观测的隐含因素。', '\\(p(x)=\\int p(x|z)p(z)dz\\)。', 'VAE、latent diffusion、style/content 分解。', '潜变量可解释性不是自动得到的，需要约束或分析。'),
    legacy('生成模型', 'Autoencoder 自编码器', ['AE'], '把输入压缩到表示再重构输入。', '\\(z=E(x),\\hat x=D(z)\\)。', 'VAE、VQ-VAE、latent diffusion 的编码器/解码器基础。', '普通 autoencoder 不是完整生成模型，除非定义 latent 采样方式。'),
    legacy('生成模型', 'ELBO 证据下界', ['evidence lower bound'], 'VAE 用可计算下界替代不可算 likelihood。', '\\(\\log p(x)\\ge E_{q(z|x)}\\log p(x|z)-D_{KL}(q(z|x)\\Vert p(z))\\)。', 'VAE 训练目标。', 'ELBO gap 会影响 likelihood 与生成质量。'),
    legacy('生成模型', 'Variational Inference 变分推断', ['VI'], '用可训练近似分布逼近难算后验。', '最小化 \\(D_{KL}(q_\\phi(z|x)\\Vert p_\\theta(z|x))\\)。', 'VAE encoder、Bayesian approximate inference。', '近似族太弱会带来 posterior gap。'),
    legacy('生成模型', 'Reparameterization Trick 重参数化技巧', ['reparam'], '把随机采样改写成确定函数加外部噪声，让梯度能传到分布参数。', '\\(z=\\mu+\\sigma\\odot\\epsilon,\\epsilon\\sim\\mathcal N(0,I)\\)。', 'VAE encoder 到 latent sample 的训练路径，用来让 reconstruction loss 能回传到 encoder 参数。', '离散变量不能直接用同样形式，需 Gumbel-softmax 或其他估计。'),
    legacy('生成模型', 'Generator 生成器', ['G'], '把噪声或条件映射成样本。', '\\(x=G_\\theta(z,c)\\)。', 'GAN、某些隐式生成模型。', '生成器本身不提供显式 likelihood。'),
    legacy('生成模型', 'Discriminator 判别器', ['D'], '判断样本来自真实数据还是生成器。', '\\(D(x)\\in(0,1)\\)。', 'GAN 中提供密度比相关训练信号。', '判别器太强或太弱都可能让生成器训练困难。'),
    legacy('生成模型', 'Minimax Game 极小极大博弈', ['GAN objective'], '生成器和判别器互相竞争。', '\\(\\min_G\\max_D V(G,D)\\)。', 'GAN 训练理论基础。', '实际常用 non-saturating、hinge、WGAN 等变体改善梯度。'),
    legacy('生成模型', 'Jensen-Shannon Divergence', ['JS divergence'], '两个分布相对混合分布的 KL 平均。', '\\(D_{JS}(p\\Vert q)=\\frac12D_{KL}(p\\Vert m)+\\frac12D_{KL}(q\\Vert m)\\)。', '原始 GAN 最优判别器下与 JS divergence 有关。', '分布支撑不重叠时梯度可能不理想。'),
    legacy('生成模型', 'Wasserstein Distance', ['earth mover'], '把一个分布搬成另一个分布的最小运输成本。', '\\(W(p,q)=\\inf_{\\gamma}\\mathbb{E}_{(x,y)\\sim\\gamma}\\|x-y\\|\\)。', 'WGAN、Optimal Transport、Flow Matching 直觉。', '计算和约束 Lipschitz 条件并不简单。'),
    legacy('生成模型', 'Mode Collapse 模式崩塌', ['collapse'], '生成模型只覆盖少数模式，样本多样性不足。', '表现为 \\(p_\\theta\\) 支撑小于 \\(p_{data}\\) 的多样性。', 'GAN、过强 guidance、偏好优化都可能出现。', '只看少量好样本会掩盖 collapse，需要覆盖率指标。'),
    legacy('生成模型', 'Diffusion Model 扩散模型', ['DDPM'], '训练去噪模型，把噪声逐步还原成数据。', 'forward noising + reverse denoising chain。', '图像、视频、音频生成主流框架。', '训练目标、采样器和 scheduler 是三件不同的事。'),
    legacy('生成模型', 'Markov Chain 马尔可夫链', ['Markov'], '下一步只依赖当前状态，不依赖更早历史。', '\\(q(x_t|x_{t-1},...,x_0)=q(x_t|x_{t-1})\\)。', 'DDPM forward/reverse process。', 'Markov 假设是建模结构，不代表真实数据生成过程一定 Markov。'),
    legacy('生成模型', 'Gaussian Transition 高斯转移', ['normal transition'], '每一步加一点高斯噪声。', '\\(q(x_t|x_{t-1})=\\mathcal N(\\sqrt{1-\\beta_t}x_{t-1},\\beta_t I)\\)。', 'DDPM forward process。', '噪声 schedule 会影响训练信号和采样质量。'),
    legacy('生成模型', 'Closed Form of Noising 前向闭式加噪', ['x_t formula'], '不用逐步加噪，可以直接从 \\(x_0\\) 采样任意噪声等级。', '\\(x_t=\\sqrt{\\bar\\alpha_t}x_0+\\sqrt{1-\\bar\\alpha_t}\\epsilon\\)。', 'Diffusion 训练高效采样任意 timestep。', '闭式是 forward process 的性质，不是 reverse sampling。'),
    legacy('生成模型', 'Denoising Objective 去噪目标', ['noise prediction'], '让模型预测加入的噪声或干净样本。', '\\(\\mathbb{E}\\|\\epsilon-\\epsilon_\\theta(x_t,t,c)\\|^2\\)。', 'DDPM/Latent Diffusion 常见训练 loss。', 'simple MSE 是加权 ELBO/score 相关 surrogate，不等同于完整采样器。'),
    legacy('生成模型', 'Score', ['score function'], 'log density 对样本的梯度，指向密度上升方向。', '\\(s(x)=\\nabla_x\\log p(x)\\)。', 'score-based diffusion 和 denoising score matching。', 'score 不是 reward；它是数据空间密度梯度。'),
    legacy('生成模型', 'Score Matching', ['DSM'], '不用显式密度，直接学习 score。', '\\(\\mathbb{E}\\|s_\\theta(x)-\\nabla_x\\log p(x)\\|^2\\)。', 'Diffusion/score-based generative modeling。', '真实 score 不可直接观测，通常通过加噪条件构造监督。'),
    legacy('生成模型', 'SDE 随机微分方程', ['stochastic differential equation'], '连续时间随机过程，包含确定 drift 和随机 noise。', '\\(dx=f(x,t)dt+g(t)dw\\)。', 'score-based diffusion 的连续时间形式。', 'SDE 采样涉及数值离散误差和 solver 选择。'),
    legacy('生成模型', 'Reverse SDE 反向 SDE', ['reverse process'], '从噪声分布反向走回数据分布的随机过程。', '反向 drift 包含 score 项。', 'score-based generation。', 'score 估计误差会在反向采样中累积。'),
    legacy('生成模型', 'ODE 常微分方程', ['probability flow ODE'], '没有随机项的连续动力系统。', '\\(dx/dt=v_\\theta(x,t)\\)。', 'Flow Matching、probability flow ODE、deterministic samplers。', 'ODE solver 步数少会快，但误差可能增大。'),
    legacy('生成模型', 'Classifier-free Guidance', ['CFG'], '用条件和无条件预测差值加强条件遵循。', '\\(\\hat\\epsilon=(1+w)\\epsilon_\\theta(x,c)-w\\epsilon_\\theta(x,\\varnothing)\\)。', '文本到图像/视频生成。', 'guidance 太强会降低多样性、造成过饱和或伪影。'),
    legacy('生成模型', 'Latent Space 潜空间', ['latent'], '压缩后的连续或离散表示空间。', '\\(z=E(x)\\)，生成在 \\(z\\) 空间进行再解码。', 'Latent Diffusion、VAE、VQGAN。', 'latent 压缩会丢信息，影响细节和文字渲染。'),
    legacy('生成模型', 'Latent Diffusion', ['LDM'], '在 VAE latent 而不是像素空间做 diffusion。', '训练 \\(p_\\theta(z_{t-1}|z_t,c)\\)，最后 decoder 还原图像。', 'Stable Diffusion 类图像/视频模型。', 'VAE 质量限制生成上限；latent 与 pixel 指标会错位。'),
    legacy('生成模型', 'Cross-attention', ['condition attention'], '让生成状态查询条件 token。', '\\(\\operatorname{softmax}(Q_xK_c^\\top/\\sqrt d)V_c\\)。', '文本条件图像生成、多模态模型。', 'cross-attention map 只是条件使用线索，不是完整因果解释。'),
    legacy('生成模型', 'Conditional Generation 条件生成', ['p(x|c)'], '给定 prompt、类别、图像、音频等条件生成样本。', '目标是建模 \\(p_\\theta(x|c)\\)。', 'text-to-image、instruction following、audio-driven video。', '条件分布改变后，无条件质量指标不足以评估。'),
    legacy('生成模型', 'Information Bottleneck 信息瓶颈', ['bottleneck'], '压缩输入中无关信息，保留任务相关信息。', '常写成保留 \\(I(Z;Y)\\)，压缩 \\(I(Z;X)\\)。', 'latent compression、representation learning、VAE。', '瓶颈太强会损失细节，太弱又不利于抽象和压缩。'),
    legacy('生成模型', 'Continuous Normalizing Flow 连续归一化流', ['CNF'], '用 ODE 定义可逆连续变换。', '\\(\\frac{d\\log p_t(x_t)}{dt}=-\\nabla\\cdot v_t(x_t)\\)。', 'Flow Matching、Neural ODE 生成模型。', 'likelihood 计算需要 divergence，可能有额外成本。'),
    legacy('生成模型', 'Vector Field 向量场', ['velocity field'], '给空间中每个点分配一个运动方向和速度。', '\\(v_t(x)\\)。', 'Flow Matching 学习从噪声到数据的速度场。', '训练的条件速度和采样的边缘速度要区分。'),
    legacy('生成模型', 'Probability Path 概率路径', ['path'], '连接 base distribution 和 data distribution 的连续分布序列。', '\\(p_0\\to p_t\\to p_1\\)。', 'Flow Matching 先指定路径，再学习速度。', '路径选择影响训练难度和采样轨迹。'),
    legacy('生成模型', 'Continuity Equation 连续性方程', ['mass conservation'], '密度随速度场流动时满足的质量守恒方程。', '\\(\\partial_t p_t+\\nabla\\cdot(p_tv_t)=0\\)。', 'Flow Matching 的理论基础。', '它描述分布演化，不是直接的神经网络 loss。'),
    legacy('生成模型', 'Optimal Transport 最优传输', ['OT'], '寻找把一个分布搬到另一个分布的低成本耦合。', '\\(\\inf_\\gamma \\mathbb{E}_{\\gamma}c(x,y)\\)。', 'Flow Matching 路径设计、Wasserstein 距离。', 'OT 计算可很贵，工程中常用近似或特殊路径。'),

    legacy('对齐与偏好优化', 'MDP 马尔可夫决策过程', ['Markov Decision Process'], '强化学习中状态、动作、奖励和转移的数学框架。', '\\((S,A,P,R,\\gamma)\\)。', '把 LLM 生成过程看作逐 token policy rollout。', 'LLM 对话不总是干净 MDP；状态和奖励定义常有工程近似。'),
    legacy('对齐与偏好优化', 'Policy 策略', ['pi'], '给定状态选择动作的分布。', '\\(\\pi(a|s)\\)。', 'LLM 中 policy 是 \\(\\pi_\\theta(y_t|x,y_{<t})\\)。', 'policy 与 reward model 是不同对象。'),
    legacy('对齐与偏好优化', 'Reward 奖励', ['r'], '对行为或输出质量的标量评价。', '\\(r(x,y)\\)。', 'RLHF、RLAIF、GRPO、DDPO。', 'reward 会被优化器利用漏洞，需要防 reward hacking。'),
    legacy('对齐与偏好优化', 'Value Function 价值函数', ['V'], '从某状态开始未来期望回报。', '\\(V^\\pi(s)=\\mathbb{E}_\\pi[\\sum_t\\gamma^t r_t|s]\\)。', 'PPO critic、advantage estimation。', '价值函数误差会影响 policy gradient 稳定性。'),
    legacy('对齐与偏好优化', 'Advantage 优势函数', ['A'], '某动作比当前平均策略好多少。', '\\(A(s,a)=Q(s,a)-V(s)\\)。', 'PPO、GRPO 中用于降低方差。', 'advantage 归一化改变尺度，要配合 clip/KL 监控。'),
    legacy('对齐与偏好优化', 'Policy Gradient 策略梯度', ['REINFORCE'], '直接对策略期望奖励求梯度。', '\\(\\nabla J=\\mathbb{E}[R\\nabla_\\theta\\log\\pi_\\theta(a|s)]\\)。', 'RLHF/PPO、DDPO。', '方差高，需要 baseline、advantage 或 clipping 稳定。'),
    legacy('对齐与偏好优化', 'PPO', ['proximal policy optimization'], '限制新旧策略变化幅度的策略优化方法。', '使用 clipped ratio objective。', '经典 RLHF pipeline 的 policy optimization 阶段。', 'PPO 工程复杂，reward、KL、clip、采样都可能出问题。'),
    legacy('对齐与偏好优化', 'KL-constrained RL', ['KL penalty'], '最大化奖励同时约束 policy 不要偏离 reference。', '\\(\\max_\\pi E[r]-\\beta D_{KL}(\\pi\\Vert\\pi_{ref})\\)。', 'RLHF、DPO 理论推导。', 'KL 太小学不动，太大可能漂移或 reward hacking。'),
    legacy('对齐与偏好优化', 'Preference Data 偏好数据', ['chosen rejected'], '同一输入下哪个输出更好的人类或模型偏好标注。', '样本形如 \\((x,y_w,y_l)\\)。', 'Reward model、DPO、RLAIF。', '偏好数据质量决定上限；标注偏差会被模型放大。'),
    legacy('对齐与偏好优化', 'Bradley-Terry Model', ['BT model'], '用两个候选的 reward 差建模胜出概率。', '\\(P(y_w\\succ y_l|x)=\\sigma(r(x,y_w)-r(x,y_l))\\)。', 'Reward model 和 DPO 推导的基础。', '它是假设模型；复杂偏好不一定只由一个标量差解释。'),
    legacy('对齐与偏好优化', 'Reward Model 奖励模型', ['RM'], '把 prompt 和输出映射成质量分数。', '\\(r_\\phi(x,y)\\)。', 'RLHF 中先训练 RM，再优化 policy。', 'RM 不是人类偏好的完美代理，会被 policy exploit。'),
    legacy('对齐与偏好优化', 'DPO 直接偏好优化', ['Direct Preference Optimization'], '不显式训练 reward model，直接用偏好 pair 更新 policy。', '\\(-\\log\\sigma(\\beta[\\log\\frac{\\pi_\\theta(y_w)}{\\pi_{ref}(y_w)}-\\log\\frac{\\pi_\\theta(y_l)}{\\pi_{ref}(y_l)}])\\)。', 'LLM alignment 中常用的偏好微调目标。', 'DPO 简化流程但不消除数据偏差、reference 选择和 KL drift 问题。'),
    legacy('对齐与偏好优化', 'RLAIF', ['AI feedback'], '用 AI 反馈替代或辅助人类偏好反馈。', '偏好或 reward 来源是 AI judge。', '低成本扩展对齐数据。', 'AI judge bias 会传给被训练模型，需校准和抽检。'),
    legacy('对齐与偏好优化', 'Reward Hacking', ['specification gaming'], '模型找到提高 reward 的捷径，但不真正满足人类意图。', '优化代理目标导致真实目标偏离。', 'RLHF、agent、生成模型奖励优化。', '只看 reward 曲线会上当，需要人工审查和分布外测试。'),

    legacy('评测统计与泛化', 'Benchmark', ['eval set'], '固定任务集合，用来比较模型表现。', '指标是样本上的统计量。', 'LLM leaderboard、图像生成指标、代码评测。', 'benchmark 可能污染、过拟合或不能代表真实使用场景。'),
    legacy('评测统计与泛化', 'Win-rate 胜率', ['pairwise win rate'], '成对比较中一个模型胜出的比例。', '\\(\\hat p=\\text{wins}/n\\)。', '偏好评测、A/B test、LLM-as-judge。', '需要置信区间和 judge bias 检查；单个胜率不够。'),
    legacy('评测统计与泛化', 'Confidence Interval 置信区间', ['CI'], '估计值的不确定范围。', '均值近似 \\(\\bar x\\pm1.96s/\\sqrt n\\)。', 'benchmark、win-rate、人工评测报告。', 'CI 窄不代表评测无偏；只表示抽样不确定性较小。'),
    legacy('评测统计与泛化', 'Bootstrap', ['resampling'], '通过重复重采样估计统计量波动。', '从样本中有放回抽取多次，计算指标分布。', '复杂指标的置信区间和稳健性分析。', '样本不独立或分布偏移时 bootstrap 也会误导。'),
    legacy('评测统计与泛化', 'P-value', ['significance'], '在零假设下观察到至少这么极端结果的概率。', '\\(p=P(T\\ge T_{obs}|H_0)\\)。', '比较实验显著性。', 'p-value 不是效果大小，也不是假设为真的概率。'),
    legacy('评测统计与泛化', 'Multiple Comparison 多重比较', ['multiple testing'], '同时做很多检验会增加误报概率。', '需要 Bonferroni、FDR 等校正。', '大量 benchmark、ablation、prompt 子集比较。', '只报告最好的显著结果容易 p-hacking。'),
    legacy('评测统计与泛化', 'Calibration 校准', ['calibrated confidence', 'ECE', 'Expected Calibration Error'], '模型置信度和真实正确率是否一致。', '预测 80% 置信的样本应约 80% 正确。', '分类器、LLM uncertainty、AI judge。', '准确率高不代表校准好。'),
    legacy('评测统计与泛化', 'OOD 分布外', ['out-of-distribution'], '测试样本来自训练分布之外。', '\\(p_{test}\\ne p_{train}\\)。', '鲁棒性、安全、泛化评估。', 'IID benchmark 好不代表 OOD 表现好。'),
    legacy('评测统计与泛化', 'Bias-Variance Tradeoff 偏差-方差权衡', ['bias variance'], '模型误差来自系统偏差和对数据扰动的敏感性。', 'MSE 可分解为 bias² + variance + noise。', '泛化、模型容量、评测稳定性。', '深度学习中的双下降让传统直觉需要谨慎使用。'),
    legacy('评测统计与泛化', 'Overfitting 过拟合', ['memorization'], '训练集表现好，但新数据表现差。', '泛化差距 \\(L_{test}-L_{train}\\)。', '小数据微调、benchmark 泄漏、reward model 过拟合。', '训练 loss 低不是问题本身，关键看 held-out 和真实分布。'),
    legacy('评测统计与泛化', 'Rademacher Complexity', ['complexity measure'], '衡量函数类拟合随机噪声标签的能力。', '\\(\\mathfrak{R}_n(\\mathcal{F})=E_\\sigma[\\sup_{f\\in\\mathcal{F}}\\frac1n\\sum_i\\sigma_if(x_i)]\\)。', '泛化理论中的容量度量。', '对现代大模型常难直接给紧界，更多是理论视角。'),
    legacy('评测统计与泛化', 'VC Dimension', ['VC dim'], '函数类能打散多少样本的最大数量。', '若任意标记都能实现，则该样本集被 shatter。', '经典统计学习理论。', 'VC 维对深度网络常过于宽松，不能直接预测实际泛化。'),
    legacy('评测统计与泛化', 'PAC-Bayes', ['PAC Bayes'], '用后验分布和 KL 项给泛化界。', '界通常包含经验风险和 \\(D_{KL}(Q\\Vert P)\\)。', '神经网络泛化、随机化预测器理论。', '界的数值紧不紧和先验/后验选择高度相关。')
  ];
})();
