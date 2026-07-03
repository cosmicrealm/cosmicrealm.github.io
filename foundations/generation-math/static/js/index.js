(function () {
  const root = document;

  const stepperData = {
    gan: [
      {
        title: '二分类目标',
        formula: '\\[\\min_G\\max_D\\;\\mathbb{E}_{p_{data}}\\log D(x)+\\mathbb{E}_{p_g}\\log(1-D(x))\\]',
        body: '判别器的任务是估计样本来自真实分布而不是生成分布的概率。'
      },
      {
        title: '固定生成器求最优判别器',
        formula: '\\[D^*(x)=\\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}\\]',
        body: '对每个 x 单独最大化判别器目标，得到密度比形式的闭式解。'
      },
      {
        title: '代回得到 JS divergence',
        formula: '\\[V(G,D^*)=-\\log4+2D_{JS}(p_{data}\\Vert p_g)\\]',
        body: '原始 GAN 理论上在最小化 Jensen-Shannon divergence。'
      },
      {
        title: '稳定训练的改写',
        formula: '\\[\\mathcal{J}_G=-\\mathbb{E}_{z}\\log D(G(z))\\]',
        body: 'non-saturating loss 在判别器较强时给生成器更有用的梯度。'
      },
      {
        title: 'hinge / LSGAN / PatchGAN',
        formula: '\\[\\mathcal{L}_{D}^{hinge}=\\mathbb{E}\\max(0,1-D(x))+\\mathbb{E}\\max(0,1+D(G(z)))\\]',
        body: 'hinge 用 margin 改善梯度，LSGAN 用平方误差缓解饱和，PatchGAN 把判别约束放到局部纹理网格上。'
      },
      {
        title: 'Wasserstein 与 gradient penalty',
        formula: '\\[\\lambda\\mathbb{E}_{\\hat{x}}(\\|\\nabla_{\\hat{x}}f(\\hat{x})\\|_2-1)^2\\]',
        body: 'WGAN-GP 用 Lipschitz 约束让 critic 提供更平滑的训练信号。'
      }
    ],
    diffusion: [
      {
        title: 'forward Markov noising',
        formula: '\\[q(x_t|x_{t-1})=\\mathcal{N}(\\sqrt{1-\\beta_t}x_{t-1},\\beta_t I)\\]',
        body: '前向过程是固定的，不需要学习；它把数据逐步破坏为高斯噪声。'
      },
      {
        title: '任意噪声等级闭式采样',
        formula: '\\[x_t=\\sqrt{\\bar{\\alpha}_t}x_0+\\sqrt{1-\\bar{\\alpha}_t}\\epsilon\\]',
        body: '训练时可以直接采样任意 t，而不必真的跑完整前向链。'
      },
      {
        title: 'reverse process',
        formula: '\\[p_\\theta(x_{t-1}|x_t)=\\mathcal{N}(\\mu_\\theta(x_t,t),\\Sigma_\\theta(x_t,t))\\]',
        body: '生成时从 x_T 开始逐步采样反向 transition，最终得到 x_0。'
      },
      {
        title: 'noise prediction',
        formula: '\\[\\mathbb{E}_{x_0,\\epsilon,t}\\|\\epsilon-\\epsilon_\\theta(x_t,t)\\|_2^2\\]',
        body: 'DDPM 的 ELBO 可化简为噪声预测损失，这是最常见的训练目标。'
      },
      {
        title: 'score relation',
        formula: '\\[\\nabla_{x_t}\\log q(x_t|x_0)=-\\epsilon/\\sqrt{1-\\bar{\\alpha}_t}\\]',
        body: '预测噪声等价于学习 log density 的梯度方向，即 score field。'
      }
    ],
    flow: [
      {
        title: 'probability path',
        formula: '\\[p_0\\rightarrow p_t\\rightarrow p_1=p_{data}\\]',
        body: 'Flow Matching 先指定从噪声分布到数据分布的连续路径。'
      },
      {
        title: 'continuity equation',
        formula: '\\[\\partial_t p_t(x)+\\nabla\\cdot(p_t(x)v_t(x))=0\\]',
        body: '若样本沿速度场运动，密度随时间的变化必须满足连续性方程。'
      },
      {
        title: 'velocity target',
        formula: '\\[\\mathcal{J}_{FM}=\\mathbb{E}\\|v_\\theta(x_t,t)-u_t(x_t)\\|_2^2\\]',
        body: '训练目标是监督回归路径上的速度，而不是直接估计密度。'
      },
      {
        title: 'ODE sampling',
        formula: '\\[x_1=x_0+\\int_0^1 v_\\theta(x_t,t)\\,dt\\]',
        body: '采样就是从 base noise 出发，沿模型速度场积分到数据时间。'
      },
      {
        title: 'MeanFlow average velocity',
        formula: '\\[\\bar{v}(x_t,r,t)\\approx (x_t-x_r)/(t-r)\\]',
        body: '平均速度目标试图把多步积分压缩为 one-step 或 few-step 生成。'
      }
    ],
    post: [
      {
        title: 'SFT 是条件最大似然',
        formula: '\\[\\mathcal{J}_{SFT}=-\\sum_t\\log\\pi_\\theta(y_t^*|x,y_{\\lt t}^*)\\]',
        body: '监督微调先让模型模仿高质量示范，但不直接比较两个输出的优劣。'
      },
      {
        title: 'Reward Model 来自偏好概率',
        formula: '\\[P(y_w\\succ y_l|x)=\\sigma(r(x,y_w)-r(x,y_l))\\]',
        body: 'Bradley-Terry 模型把 pairwise preference 转成可训练的标量 reward。'
      },
      {
        title: 'RLHF 是 KL 正则化奖励最大化',
        formula: '\\[\\max_\\pi\\;\\mathbb{E}_{y\\sim\\pi}r(x,y)-\\beta D_{KL}(\\pi\\Vert\\pi_{ref})\\]',
        body: 'PPO 只是优化该目标的一种策略梯度估计方式。'
      },
      {
        title: 'DPO 消去显式 reward',
        formula: '\\[-\\log\\sigma(\\beta[\\log\\pi_\\theta/\\pi_{ref}(y_w)-\\log\\pi_\\theta/\\pi_{ref}(y_l)])\\]',
        body: 'DPO 将最优策略形式代入偏好似然，直接训练 policy。'
      },
      {
        title: 'GRPO 使用组内相对优势',
        formula: '\\[A_i=(R_i-\\operatorname{mean}(R))/(\\operatorname{std}(R)+\\epsilon)\\]',
        body: '同一 prompt 的候选样本互为 baseline，因此可以减少对 critic 的依赖。'
      },
      {
        title: 'DDPO 与 Flow-DPO 面向生成轨迹',
        formula: '\\[\\nabla_\\theta J\\approx\\mathbb{E}_{\\tau}R(c,x_0)\\sum_t\\nabla_\\theta\\log p_\\theta(x_{t-1}|x_t,c)\\]',
        body: '图像和视频模型的 action 是去噪或 velocity trajectory，不能简单照搬 token-level DPO。'
      }
    ]
  };

  const flowData = {
    vae: {
      train: [
        {
          title: '训练伪代码',
          code: [
            'for each minibatch x:',
            '  mu, logvar = Encoder_phi(x)',
            '  sigma = exp(0.5 * logvar)',
            '  epsilon ~ Normal(0, I)',
            '  z = mu + sigma * epsilon',
            '  recon_params = Decoder_theta(z)',
            '  recon_loss = -log p_theta(x | z)',
            '  kl = KL(N(mu, diag(sigma^2)) || N(0, I))',
            '  loss = recon_loss + beta * kl',
            '  update phi, theta by gradient descent'
          ].join('\n'),
          note: '随机量只来自 epsilon；reparameterization 让 z 的采样路径仍能对 encoder 参数反向传播。'
        }
      ],
      sample: [
        {
          title: '采样伪代码',
          code: [
            'z ~ Normal(0, I)',
            'recon_params = Decoder_theta(z)',
            'if decoder is Gaussian:',
            '  x_hat = mean(recon_params) or sample Normal(recon_params)',
            'else if decoder is Bernoulli:',
            '  x_hat = sample Bernoulli(logits)',
            'return x_hat'
          ].join('\n'),
          note: '采样从 prior 开始，而不是从 q_phi(z|x) 开始；prior-posterior gap 会直接影响新样本质量。'
        }
      ]
    },
    vqvae: {
      train: [
        {
          title: '训练伪代码',
          code: [
            'for each minibatch x:',
            '  z_e = Encoder_phi(x)',
            '  k = argmin_j ||z_e - e_j||_2',
            '  z_q = e_k with straight-through gradient',
            '  x_hat = Decoder_theta(z_q)',
            '  loss = reconstruction(x, x_hat)',
            '       + ||sg(z_e) - e_k||_2^2',
            '       + beta * ||z_e - sg(e_k)||_2^2',
            '  if EMA codebook:',
            '    update N_i, m_i, e_i by moving averages',
            '  update encoder, decoder, and codebook state'
          ].join('\n'),
          note: '最近邻选择是离散操作；straight-through 负责传梯度，EMA 更新负责让 code center 更像 online k-means。'
        }
      ],
      sample: [
        {
          title: '采样伪代码',
          code: [
            'code_indices = Prior.sample(condition)',
            'z_q = lookup(codebook, code_indices)',
            'x_hat = Decoder_theta(z_q)',
            'return x_hat'
          ].join('\n'),
          note: 'VQ 模型本身更像 tokenizer；真正的生成通常由离散 prior 决定全局结构。'
        }
      ]
    },
    gan: {
      train: [
        {
          title: '训练伪代码',
          code: [
            'for each step:',
            '  x_real ~ p_data',
            '  z ~ p(z)',
            '  x_fake = G_theta(z).detach()',
            '  loss_D = discriminator_loss(D_psi(x_real), D_psi(x_fake))',
            '  if WGAN-GP:',
            '    loss_D += lambda * (||grad_x D_psi(x_hat)||_2 - 1)^2',
            '  update psi',
            '',
            '  z ~ p(z)',
            '  x_fake = G_theta(z)',
            '  loss_G = generator_loss(D_psi(x_fake))',
            '  update theta'
          ].join('\n'),
          note: '训练是两方优化；hinge、LSGAN、non-saturating、WGAN-GP 都是在改善同一分布匹配博弈的梯度形态。'
        }
      ],
      sample: [
        {
          title: '采样伪代码',
          code: [
            'z ~ p(z)',
            'x_hat = G_theta(z)',
            'return x_hat'
          ].join('\n'),
          note: 'GAN 推断通常只需一次生成器前向，速度快；风险主要在训练稳定性和 mode coverage。'
        }
      ]
    },
    diffusion: {
      train: [
        {
          title: '训练伪代码',
          code: [
            'for each minibatch x_0, condition c:',
            '  t ~ Uniform({1, ..., T})',
            '  epsilon ~ Normal(0, I)',
            '  x_t = sqrt(alpha_bar_t) * x_0',
            '      + sqrt(1 - alpha_bar_t) * epsilon',
            '  epsilon_hat = epsilon_theta(x_t, t, c)',
            '  loss = ||epsilon - epsilon_hat||_2^2',
            '  update theta'
          ].join('\n'),
          note: '前向 q 是固定加噪过程；训练只学习反向去噪所需的信息。'
        }
      ],
      sample: [
        {
          title: '反向采样伪代码',
          code: [
            'x_T ~ Normal(0, I)',
            'for t = T, ..., 1:',
            '  epsilon_hat = epsilon_theta(x_t, t, c)',
            '  mu = (1 / sqrt(alpha_t))',
            '       * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_hat)',
            '  if DDPM and t > 1:',
            '    z ~ Normal(0, I)',
            '    x_{t-1} = mu + sigma_t * z',
            '  else:',
            '    x_{t-1} = deterministic DDIM/probability-flow step',
            'return x_0'
          ].join('\n'),
          note: 'DDPM 是随机反向链；DDIM/probability flow ODE 使用同一网络但减少或移除随机噪声。'
        }
      ]
    },
    flow: {
      train: [
        {
          title: '训练伪代码',
          code: [
            'for each minibatch:',
            '  x_base ~ p_0',
            '  x_data ~ p_data',
            '  t ~ Uniform(0, 1)',
            '  x_t = psi_t(x_base, x_data)',
            '  u_t = d psi_t(x_base, x_data) / dt',
            '  v_hat = v_theta(x_t, t, c)',
            '  loss = ||v_hat - u_t||_2^2',
            '  update theta'
          ].join('\n'),
          note: '训练时可以构造 endpoints；采样时没有真实 x_data，只能沿学到的速度场积分。'
        }
      ],
      sample: [
        {
          title: 'ODE 采样伪代码',
          code: [
            'x = sample p_0',
            'for k = 0, ..., K - 1:',
            '  t = k / K',
            '  dt = 1 / K',
            '  v = v_theta(x, t, c)',
            '  x = x + dt * v       # Euler',
            'return x'
          ].join('\n'),
          note: 'Flow Matching 通常不是 learned reverse Markov chain；核心是从 base distribution 积分到 data distribution。'
        }
      ]
    },
    ar: {
      train: [
        {
          title: '训练伪代码',
          code: [
            'for each sequence x_1, ..., x_T and condition c:',
            '  input = [BOS, x_1, ..., x_{T-1}]',
            '  targets = [x_1, ..., x_T]',
            '  logits = Transformer_theta(input, c, causal_mask=True)',
            '  loss = sum_t CrossEntropy(logits_t, targets_t)',
            '  update theta'
          ].join('\n'),
          note: 'teacher forcing 使用真实前缀，训练可并行计算所有位置的 next-token loss。'
        }
      ],
      sample: [
        {
          title: '逐 token 采样伪代码',
          code: [
            'prefix = prompt tokens',
            'while not stop and len(prefix) < max_len:',
            '  logits = Transformer_theta(prefix, c)',
            '  probs = softmax(logits[-1] / temperature)',
            '  probs = apply_top_k_top_p_filter(probs)',
            '  token = sample(probs)',
            '  prefix.append(token)',
            'return prefix'
          ].join('\n'),
          note: 'decoding 改变取样策略，不更新模型参数；过低温度易重复，过高温度易漂移。'
        }
      ]
    },
    post: {
      train: [
        {
          title: '后训练伪代码',
          code: [
            'start from pi_ref or SFT policy',
            'collect demonstrations / preferences / rewards',
            'if SFT:',
            '  minimize token-level conditional NLL',
            'if PPO or GRPO:',
            '  rollout y ~ pi_theta(. | x)',
            '  compute reward and KL to pi_ref',
            '  estimate advantage and update policy',
            'if DPO:',
            '  read (x, y_w, y_l)',
            '  optimize reference log-ratio margin',
            'if DDPO / Flow-DPO:',
            '  assign final reward or preference to generation trajectory'
          ].join('\n'),
          note: '后训练改变参数与条件分布；post-processing 只筛选或修补当前样本，不改变下一次采样分布。'
        }
      ],
      sample: [
        {
          title: '推断伪代码',
          code: [
            'given condition x or c:',
            '  use post-trained policy/model pi_theta',
            '  if LLM: autoregressive token sampling',
            '  if diffusion: reverse denoising sampler',
            '  if flow: ODE velocity sampler',
            '  optionally apply post-processing filters',
            'return output'
          ].join('\n'),
          note: '推断仍沿用基础生成器的 sampler；后训练让 sampler 背后的条件分布更偏向被奖励或被偏好的输出。'
        }
      ]
    }
  };

  let activeDist = 'gaussian';
  let activeGanMode = 'balanced';
  let activeFlowVisMode = 'instant';
  let activeVqMode = 'ema';

  function typeset(element) {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise(element ? [element] : undefined).catch(function () {});
    }
  }

  function escapeHTML(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function setActive(buttons, selected) {
    buttons.forEach(function (button) {
      const active = button === selected;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
  }

  function initSteppers() {
    root.querySelectorAll('[data-stepper]').forEach(function (widget) {
      const key = widget.dataset.stepper;
      const steps = stepperData[key] || [];
      const output = widget.querySelector('[data-step-output]');
      const prev = widget.querySelector('[data-step-prev]');
      const next = widget.querySelector('[data-step-next]');
      let index = 0;
      if (!steps.length || !output || !prev || !next) return;

      function render() {
        const step = steps[index];
        output.innerHTML = [
          '<p class="step-count">Step ' + (index + 1) + ' / ' + steps.length + '</p>',
          '<h4>' + step.title + '</h4>',
          '<div class="equation">' + step.formula + '</div>',
          '<p>' + step.body + '</p>'
        ].join('');
        typeset(output);
      }

      prev.addEventListener('click', function () {
        index = (index - 1 + steps.length) % steps.length;
        render();
      });
      next.addEventListener('click', function () {
        index = (index + 1) % steps.length;
        render();
      });
      render();
    });
  }

  function initFlowPanels() {
    root.querySelectorAll('[data-flow-panel]').forEach(function (panel) {
      const key = panel.dataset.flowPanel;
      const output = panel.querySelector('[data-flow-output]');
      const buttons = Array.from(panel.querySelectorAll('[data-flow-mode]'));
      let mode = 'train';
      if (!flowData[key] || !output || !buttons.length) return;

      function render() {
        const cards = flowData[key][mode] || [];
        output.innerHTML = cards.map(function (card) {
          const note = card.note ? '<p class="algorithm-note">' + escapeHTML(card.note) + '</p>' : '';
          return '<article class="flow-card"><h4>' + escapeHTML(card.title) + '</h4><pre class="algorithm"><code>' + escapeHTML(card.code || '') + '</code></pre>' + note + '</article>';
        }).join('');
        typeset(output);
      }

      buttons.forEach(function (button) {
        button.addEventListener('click', function () {
          mode = button.dataset.flowMode || 'train';
          setActive(buttons, button);
          render();
        });
      });
      render();
    });
  }

  function initVqUsage() {
    const container = root.getElementById('vqUsageBars');
    const buttons = Array.from(root.querySelectorAll('[data-vq-mode]'));
    if (!container) return;

    const usageData = {
      ema: [
        { code: 'e_04', usage: 78, note: 'frequent visual token' },
        { code: 'e_11', usage: 58, note: 'stable assignment' },
        { code: 'e_23', usage: 34, note: 'mid-frequency code' },
        { code: 'e_31', usage: 16, note: 'low but alive' },
        { code: 'e_38', usage: 4, note: 'dead-code risk' }
      ],
      gradient: [
        { code: 'e_04', usage: 92, note: 'dominant code' },
        { code: 'e_11', usage: 46, note: 'moving centroid' },
        { code: 'e_23', usage: 22, note: 'unstable assignment' },
        { code: 'e_31', usage: 7, note: 'under-used' },
        { code: 'e_38', usage: 2, note: 'restart candidate' }
      ]
    };

    function render() {
      const rows = usageData[activeVqMode] || usageData.ema;
      container.innerHTML = '';
      rows.forEach(function (row) {
        const item = root.createElement('div');
        const state = row.usage < 10 ? ' low' : row.usage > 75 ? ' high' : '';
        item.className = 'code-usage-row' + state;
        item.innerHTML = '<span>' + row.code + '</span><i style="--usage: ' + row.usage + '%"></i><b>' + row.note + '</b>';
        container.appendChild(item);
      });
    }

    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        activeVqMode = button.dataset.vqMode || 'ema';
        setActive(buttons, button);
        render();
      });
    });
    render();
  }

  function initCalculators() {
    const vaeMu = root.getElementById('vaeMu');
    const vaeSigma = root.getElementById('vaeSigma');
    const vaeMuOut = root.getElementById('vaeMuOut');
    const vaeSigmaOut = root.getElementById('vaeSigmaOut');
    const vaeKlOut = root.getElementById('vaeKlOut');
    const grpoRewards = root.getElementById('grpoRewards');
    const grpoOut = root.getElementById('grpoOut');
    const dpoDelta = root.getElementById('dpoDelta');
    const dpoBeta = root.getElementById('dpoBeta');
    const dpoDeltaOut = root.getElementById('dpoDeltaOut');
    const dpoBetaOut = root.getElementById('dpoBetaOut');
    const dpoOut = root.getElementById('dpoOut');

    function updateVae() {
      if (!vaeMu || !vaeSigma || !vaeKlOut || !vaeMuOut || !vaeSigmaOut) return;
      const mu = Number(vaeMu.value);
      const sigma = Number(vaeSigma.value);
      const kl = 0.5 * (mu * mu + sigma * sigma - Math.log(sigma * sigma) - 1);
      vaeMuOut.textContent = mu.toFixed(1);
      vaeSigmaOut.textContent = sigma.toFixed(1);
      vaeKlOut.textContent = 'KL = ' + kl.toFixed(4);
      const vaeKlBar = root.getElementById('vaeKlBar');
      const vaeReconBar = root.getElementById('vaeReconBar');
      if (vaeKlBar && vaeReconBar) {
        const klLevel = Math.max(8, Math.min(92, 18 + kl * 18));
        vaeKlBar.style.setProperty('--level', klLevel.toFixed(0) + '%');
        vaeReconBar.style.setProperty('--level', Math.max(8, 100 - klLevel * 0.62).toFixed(0) + '%');
      }
    }

    function updateGrpo() {
      if (!grpoRewards || !grpoOut) return;
      const values = grpoRewards.value.split(',').map(function (value) {
        return Number(value.trim());
      }).filter(Number.isFinite);
      if (!values.length) {
        grpoOut.textContent = 'A = []';
        return;
      }
      const mean = values.reduce(function (sum, value) { return sum + value; }, 0) / values.length;
      const variance = values.reduce(function (sum, value) {
        return sum + Math.pow(value - mean, 2);
      }, 0) / values.length;
      const std = Math.sqrt(variance) || 1;
      const advantages = values.map(function (value) {
        return ((value - mean) / (std + 1e-8)).toFixed(2);
      });
      grpoOut.textContent = 'mean = ' + mean.toFixed(3) + ', std = ' + std.toFixed(3) + ', A = [' + advantages.join(', ') + ']';
      updatePostVisual();
    }

    function updateDpo() {
      if (!dpoDelta || !dpoBeta || !dpoOut || !dpoDeltaOut || !dpoBetaOut) return;
      const delta = Number(dpoDelta.value);
      const beta = Number(dpoBeta.value);
      const probability = 1 / (1 + Math.exp(-beta * delta));
      const loss = Math.log(1 + Math.exp(-beta * delta));
      dpoDeltaOut.textContent = delta.toFixed(1);
      dpoBetaOut.textContent = beta.toFixed(1);
      dpoOut.textContent = 'loss = ' + loss.toFixed(4) + ', sigma(beta * delta) = ' + probability.toFixed(4);
      updatePostVisual();
    }

    [vaeMu, vaeSigma].forEach(function (input) {
      if (input) input.addEventListener('input', updateVae);
    });
    if (grpoRewards) grpoRewards.addEventListener('input', updateGrpo);
    [dpoDelta, dpoBeta].forEach(function (input) {
      if (input) input.addEventListener('input', updateDpo);
    });
    updateVae();
    updateGrpo();
    updateDpo();
  }

  function prepareCanvas(canvas, minHeight) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(rect.width));
    const height = Math.max(minHeight || 260, Math.floor(rect.height));
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function drawPoint(ctx, x, y, radius, color, stroke) {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1.3;
      ctx.stroke();
    }
  }

  function drawCanvasArrow(ctx, x1, y1, x2, y2, color) {
    const angle = Math.atan2(y2 - y1, x2 - x1);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - 7 * Math.cos(angle - 0.45), y2 - 7 * Math.sin(angle - 0.45));
    ctx.lineTo(x2 - 7 * Math.cos(angle + 0.45), y2 - 7 * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
  }

  function initGanCanvas() {
    const canvas = root.getElementById('ganCanvas');
    const buttons = Array.from(root.querySelectorAll('[data-gan-mode]'));
    if (!canvas) return;

    function draw() {
      const prepared = prepareCanvas(canvas, 280);
      const ctx = prepared.ctx;
      const width = prepared.width;
      const height = prepared.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#ece7dc';
      for (let x = 40; x < width; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 20);
        ctx.lineTo(x, height - 34);
        ctx.stroke();
      }
      for (let y = 36; y < height - 34; y += 36) {
        ctx.beginPath();
        ctx.moveTo(32, y);
        ctx.lineTo(width - 24, y);
        ctx.stroke();
      }
      const realCenters = [[0.28, 0.34], [0.66, 0.28], [0.48, 0.68]];
      const genCenters = activeGanMode === 'collapse'
        ? [[0.64, 0.3]]
        : activeGanMode === 'wgan'
          ? [[0.32, 0.38], [0.62, 0.34], [0.5, 0.66]]
          : [[0.25, 0.42], [0.6, 0.34], [0.54, 0.72]];
      function jitter(center, index, total) {
        const a = index * 2.399 + total * 0.31;
        return [center[0] + Math.cos(a) * 0.045 + Math.sin(index) * 0.012, center[1] + Math.sin(a) * 0.055];
      }
      realCenters.forEach(function (center, ci) {
        for (let i = 0; i < 16; i += 1) {
          const p = jitter(center, i, ci);
          drawPoint(ctx, p[0] * width, p[1] * (height - 30), 4, 'rgba(15,111,104,0.78)', '#084a45');
        }
      });
      genCenters.forEach(function (center, ci) {
        for (let i = 0; i < (activeGanMode === 'collapse' ? 34 : 14); i += 1) {
          const p = jitter(center, i, ci + 4);
          drawPoint(ctx, p[0] * width, p[1] * (height - 30), 4, 'rgba(169,67,47,0.75)', '#7f2e20');
        }
      });
      ctx.strokeStyle = activeGanMode === 'wgan' ? '#9a6d19' : '#314f78';
      ctx.lineWidth = activeGanMode === 'wgan' ? 3 : 2;
      ctx.setLineDash(activeGanMode === 'wgan' ? [8, 6] : []);
      ctx.beginPath();
      ctx.moveTo(58, height - 82);
      ctx.bezierCurveTo(width * 0.24, height * 0.3, width * 0.72, height * 0.74, width - 52, 70);
      ctx.stroke();
      ctx.setLineDash([]);
      if (activeGanMode === 'wgan') {
        for (let x = width * 0.22; x < width * 0.82; x += width * 0.12) {
          drawCanvasArrow(ctx, x, height * 0.78, x + 18, height * 0.62, '#9a6d19');
        }
      }
      ctx.fillStyle = '#5f625e';
      ctx.font = '12px SFMono-Regular, Menlo, monospace';
      ctx.fillText('blue: p_data samples', 18, height - 14);
      ctx.fillText('red: p_g samples', 190, height - 14);
      ctx.fillText(activeGanMode === 'collapse' ? 'mode collapse: missing modes' : activeGanMode === 'wgan' ? 'WGAN-GP: smoother critic gradient' : 'discriminator boundary estimates density ratio', width * 0.52, height - 14);
    }

    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        activeGanMode = button.dataset.ganMode || 'balanced';
        setActive(buttons, button);
        draw();
      });
    });
    window.addEventListener('resize', draw);
    draw();
  }

  function initFlowCanvas() {
    const canvas = root.getElementById('flowCanvas');
    const buttons = Array.from(root.querySelectorAll('[data-flow-vis]'));
    if (!canvas) return;

    function draw() {
      const prepared = prepareCanvas(canvas, 280);
      const ctx = prepared.ctx;
      const width = prepared.width;
      const height = prepared.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#ece7dc';
      ctx.lineWidth = 1;
      for (let y = 40; y < height - 28; y += 38) {
        ctx.beginPath();
        ctx.moveTo(30, y);
        ctx.lineTo(width - 24, y);
        ctx.stroke();
      }
      const start = { x: width * 0.16, y: height * 0.72 };
      const end = { x: width * 0.82, y: height * 0.28 };
      ctx.strokeStyle = 'rgba(15,111,104,0.24)';
      ctx.lineWidth = 13;
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.bezierCurveTo(width * 0.32, height * 0.35, width * 0.56, height * 0.74, end.x, end.y);
      ctx.stroke();
      ctx.lineWidth = 2.3;
      ctx.strokeStyle = '#0f6f68';
      ctx.beginPath();
      ctx.moveTo(start.x, start.y);
      ctx.bezierCurveTo(width * 0.32, height * 0.35, width * 0.56, height * 0.74, end.x, end.y);
      ctx.stroke();
      drawPoint(ctx, start.x, start.y, 7, '#314f78', '#1f3552');
      drawPoint(ctx, end.x, end.y, 7, '#a9432f', '#7f2e20');
      if (activeFlowVisMode === 'average') {
        drawCanvasArrow(ctx, start.x + 12, start.y - 8, end.x - 12, end.y + 8, '#a9432f');
        ctx.fillStyle = '#a9432f';
        ctx.fillText('average velocity chord', width * 0.38, height * 0.52);
      } else {
        for (let i = 0; i < 7; i += 1) {
          const t = i / 6;
          const x = start.x * (1 - t) + end.x * t + Math.sin(t * Math.PI) * 44;
          const y = start.y * (1 - t) + end.y * t + Math.cos(t * Math.PI) * 24;
          drawCanvasArrow(ctx, x, y, x + 28, y - 8 + Math.sin(i) * 18, '#0f6f68');
        }
        ctx.fillStyle = '#0f6f68';
        ctx.fillText('instantaneous velocity field', width * 0.38, height * 0.52);
      }
      ctx.fillStyle = '#5f625e';
      ctx.font = '12px SFMono-Regular, Menlo, monospace';
      ctx.fillText('p_0 noise', start.x - 30, start.y + 28);
      ctx.fillText('p_1 data', end.x - 26, end.y - 18);
    }

    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        activeFlowVisMode = button.dataset.flowVis || 'instant';
        setActive(buttons, button);
        draw();
      });
    });
    window.addEventListener('resize', draw);
    draw();
  }

  function initEbmCanvas() {
    const canvas = root.getElementById('ebmCanvas');
    if (!canvas) return;

    function draw() {
      const prepared = prepareCanvas(canvas, 260);
      const ctx = prepared.ctx;
      const width = prepared.width;
      const height = prepared.height;
      ctx.clearRect(0, 0, width, height);
      const img = ctx.createImageData(width, height);
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const nx = x / width;
          const ny = y / height;
          const e1 = Math.exp(-((nx - 0.33) ** 2 + (ny - 0.35) ** 2) / 0.018);
          const e2 = Math.exp(-((nx - 0.68) ** 2 + (ny - 0.66) ** 2) / 0.028);
          const energy = 1 - Math.min(1, e1 + e2 * 0.9);
          const i = (y * width + x) * 4;
          img.data[i] = 245 - energy * 80;
          img.data[i + 1] = 242 - energy * 120;
          img.data[i + 2] = 232 - energy * 125;
          img.data[i + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      const path = [
        [0.12, 0.82], [0.2, 0.73], [0.28, 0.62], [0.36, 0.52],
        [0.46, 0.5], [0.56, 0.58], [0.64, 0.64], [0.69, 0.66]
      ];
      ctx.strokeStyle = '#171817';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      path.forEach(function (p, i) {
        const x = p[0] * width;
        const y = p[1] * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      path.forEach(function (p, i) {
        drawPoint(ctx, p[0] * width, p[1] * height, i === path.length - 1 ? 5 : 3.5, i === path.length - 1 ? '#a9432f' : '#ffffff', '#171817');
      });
      ctx.fillStyle = '#171817';
      ctx.font = '12px SFMono-Regular, Menlo, monospace';
      ctx.fillText('energy landscape: darker = lower E(x), Langevin path descends', 14, height - 14);
    }

    window.addEventListener('resize', draw);
    draw();
  }

  function initArVisual() {
    const mask = root.getElementById('causalMask');
    const bars = root.getElementById('arBars');
    const slider = root.getElementById('temperatureSlider');
    const output = root.getElementById('temperatureOut');
    if (!mask || !bars || !slider || !output) return;

    const size = 7;
    mask.innerHTML = '';
    for (let row = 0; row < size; row += 1) {
      for (let col = 0; col < size; col += 1) {
        const cell = root.createElement('span');
        cell.textContent = col <= row ? '1' : '0';
        cell.className = col <= row ? 'allowed' : '';
        mask.appendChild(cell);
      }
    }

    function renderBars() {
      const temp = Number(slider.value);
      output.textContent = temp.toFixed(1);
      const tokens = ['the', 'image', 'moves', 'bright', 'slowly'];
      const logits = [2.5, 1.7, 1.05, 0.55, 0.15].map(function (v) { return v / temp; });
      const maxLogit = Math.max.apply(null, logits);
      const exps = logits.map(function (v) { return Math.exp(v - maxLogit); });
      const sum = exps.reduce(function (a, b) { return a + b; }, 0);
      bars.innerHTML = '';
      exps.map(function (v) { return v / sum; }).forEach(function (prob, i) {
        const row = root.createElement('div');
        row.className = 'token-bar';
        row.innerHTML = '<span>' + tokens[i] + '</span><i style="--level: ' + (prob * 100).toFixed(1) + '%"></i><b>' + prob.toFixed(2) + '</b>';
        bars.appendChild(row);
      });
    }

    slider.addEventListener('input', renderBars);
    renderBars();
  }

  function parseRewardValues() {
    const grpoRewards = root.getElementById('grpoRewards');
    if (!grpoRewards) return [];
    return grpoRewards.value.split(',').map(function (value) {
      return Number(value.trim());
    }).filter(Number.isFinite);
  }

  function updatePostVisual() {
    const drift = root.getElementById('ppoDrift');
    const driftOut = root.getElementById('ppoDriftOut');
    const driftBar = root.getElementById('ppoDriftBar');
    const dpoDelta = root.getElementById('dpoDelta');
    const dpoBeta = root.getElementById('dpoBeta');
    const dpoBar = root.getElementById('dpoMarginBar');
    const grpoBar = root.getElementById('grpoSpreadBar');
    const summary = root.getElementById('postVisualSummary');
    if (!summary) return;
    const driftValue = drift ? Number(drift.value) : 0.35;
    if (driftOut) driftOut.textContent = driftValue.toFixed(2);
    if (driftBar) driftBar.style.setProperty('--level', Math.round(driftValue * 100) + '%');
    const delta = dpoDelta ? Number(dpoDelta.value) : 1.2;
    const beta = dpoBeta ? Number(dpoBeta.value) : 0.8;
    const marginProb = 1 / (1 + Math.exp(-beta * delta));
    if (dpoBar) dpoBar.style.setProperty('--level', Math.round(marginProb * 100) + '%');
    const values = parseRewardValues();
    let spread = 0.48;
    if (values.length) {
      const mean = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
      const variance = values.reduce(function (sum, value) { return sum + Math.pow(value - mean, 2); }, 0) / values.length;
      spread = Math.min(1, Math.sqrt(variance) / 2.5);
    }
    if (grpoBar) grpoBar.style.setProperty('--level', Math.round(spread * 100) + '%');
    summary.textContent = 'DPO preference probability = ' + marginProb.toFixed(3) + ', PPO KL drift = ' + driftValue.toFixed(2) + ', GRPO reward spread = ' + spread.toFixed(2) + '.';
  }

  function initPostVisual() {
    const drift = root.getElementById('ppoDrift');
    if (drift) drift.addEventListener('input', updatePostVisual);
    updatePostVisual();
  }

  function gaussianDensity(x, mu, sigma) {
    const z = (x - mu) / sigma;
    return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
  }

  function gaussianScore(x, mu, sigma) {
    return -(x - mu) / (sigma * sigma);
  }

  function laplaceDensity(x, mu, scale) {
    return Math.exp(-Math.abs(x - mu) / scale) / (2 * scale);
  }

  function laplaceScore(x, mu, scale) {
    if (Math.abs(x - mu) < 1e-6) return 0;
    return x > mu ? -1 / scale : 1 / scale;
  }

  function mixtureParts(x, mu, sigma) {
    const m1 = mu - 1.45;
    const m2 = mu + 1.45;
    const p1 = 0.5 * gaussianDensity(x, m1, sigma);
    const p2 = 0.5 * gaussianDensity(x, m2, sigma);
    const total = p1 + p2;
    const s1 = gaussianScore(x, m1, sigma);
    const s2 = gaussianScore(x, m2, sigma);
    return { density: total, score: total ? (p1 * s1 + p2 * s2) / total : 0 };
  }

  function distributionAt(x, mu, scale) {
    if (activeDist === 'mixture') return mixtureParts(x, mu, scale);
    if (activeDist === 'laplace') {
      return { density: laplaceDensity(x, mu, scale), score: laplaceScore(x, mu, scale) };
    }
    return { density: gaussianDensity(x, mu, scale), score: gaussianScore(x, mu, scale) };
  }

  function initScoreLab() {
    const canvas = root.getElementById('scoreCanvas');
    const buttons = Array.from(root.querySelectorAll('[data-dist]'));
    const meanSlider = root.getElementById('meanSlider');
    const scaleSlider = root.getElementById('scaleSlider');
    const meanValue = root.getElementById('meanValue');
    const scaleValue = root.getElementById('scaleValue');
    const scoreNote = root.getElementById('scoreNote');
    if (!canvas || !meanSlider || !scaleSlider || !meanValue || !scaleValue || !scoreNote) return;

    function prepareCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const width = Math.max(320, Math.floor(rect.width));
      const height = Math.max(260, Math.floor(rect.height));
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width, height };
    }

    function drawArrow(ctx, x1, y, x2, color) {
      const direction = x2 >= x1 ? 1 : -1;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(x1, y);
      ctx.lineTo(x2, y);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x2, y);
      ctx.lineTo(x2 - direction * 6, y - 4);
      ctx.lineTo(x2 - direction * 6, y + 4);
      ctx.closePath();
      ctx.fill();
    }

    function draw() {
      const mean = Number(meanSlider.value);
      const scale = Number(scaleSlider.value);
      meanValue.textContent = mean.toFixed(1);
      scaleValue.textContent = scale.toFixed(1);
      if (activeDist === 'mixture') {
        scoreNote.textContent = 'Mixture score = sum_k responsibility_k * score_k.';
      } else if (activeDist === 'laplace') {
        scoreNote.textContent = 'Laplace score 在均值两侧近似为常量方向，均值处不可导。';
      } else {
        scoreNote.textContent = 'Gaussian score 是线性回拉力，方向指向均值。';
      }

      const prepared = prepareCanvas();
      const ctx = prepared.ctx;
      const width = prepared.width;
      const height = prepared.height;
      const pad = { left: 42, right: 24, top: 24, bottom: 58 };
      const minX = -5;
      const maxX = 5;
      const values = [];
      for (let i = 0; i <= 260; i += 1) {
        const x = minX + (i / 260) * (maxX - minX);
        values.push({ x, ...distributionAt(x, mean, scale) });
      }
      const maxDensity = Math.max.apply(null, values.map(function (value) { return value.density; })) || 1;
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      function sx(x) { return pad.left + ((x - minX) / (maxX - minX)) * plotW; }
      function sy(y) { return pad.top + plotH - (y / maxDensity) * plotH * 0.9; }

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = '#ece7dc';
      ctx.lineWidth = 1;
      for (let gx = -4; gx <= 4; gx += 1) {
        ctx.beginPath();
        ctx.moveTo(sx(gx), pad.top);
        ctx.lineTo(sx(gx), pad.top + plotH);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(15,111,104,0.12)';
      ctx.beginPath();
      values.forEach(function (value, index) {
        const x = sx(value.x);
        const y = sy(value.density);
        if (index === 0) ctx.moveTo(x, pad.top + plotH);
        ctx.lineTo(x, y);
      });
      ctx.lineTo(sx(maxX), pad.top + plotH);
      ctx.closePath();
      ctx.fill();

      ctx.strokeStyle = '#0f6f68';
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      values.forEach(function (value, index) {
        const x = sx(value.x);
        const y = sy(value.density);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      const arrowY = height - 32;
      for (let x = -4.5; x <= 4.6; x += 0.75) {
        const score = distributionAt(x, mean, scale).score;
        const length = Math.max(-34, Math.min(34, score * 10));
        drawArrow(ctx, sx(x) - length * 0.18, arrowY, sx(x) + length, score >= 0 ? '#0f6f68' : '#a9432f');
      }

      ctx.fillStyle = '#5f625e';
      ctx.font = '12px SFMono-Regular, Menlo, monospace';
      ctx.fillText('density', pad.left, 16);
      ctx.fillText('score direction', pad.left, height - 40);
    }

    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        activeDist = button.dataset.dist || 'gaussian';
        setActive(buttons, button);
        draw();
      });
    });
    [meanSlider, scaleSlider].forEach(function (input) {
      input.addEventListener('input', draw);
    });
    window.addEventListener('resize', draw);
    draw();
  }

  function init() {
    initSteppers();
    initFlowPanels();
    initVqUsage();
    initCalculators();
    initGanCanvas();
    initFlowCanvas();
    initEbmCanvas();
    initArVisual();
    initPostVisual();
    initScoreLab();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
