(function () {
  const root = document;

  const algorithmSnippets = {
    kd: {
      train: `# Algorithm 1A: classic classification KD
for x in dataloader:
    with no_grad():
        q_T = softmax(teacher.logits(x) / temperature)
    q_S = softmax(student.logits(x) / temperature)
    loss = temperature**2 * KL(q_T || q_S)
    update(student, loss)

# Algorithm 1B: generic generative distillation
for condition c in prompts_or_labels:
    z = sample_noise()
    teacher_signal = query_teacher(Phi_T, z, c)
    student_output = Phi_theta(z, c)
    loss = match(student_output, teacher_signal)
    update(student, loss)`,
      sample: `# Inference after generative distillation
for c in user_conditions:
    z = sample_noise()
    x = fast_student_sampler(Phi_theta, z, c, num_steps=1_or_few)
    return decode_if_needed(x)`
    },
    progressive: {
      train: `# Algorithm 2: progressive diffusion distillation
for round in distillation_rounds:
    teacher = freeze(previous_sampler)
    student = initialize_from(teacher)
    new_schedule = halve_schedule(teacher.schedule)

    for x0, c in dataloader:
        t = sample_timestep_from(new_schedule)
        xt = forward_noise(x0, t)
        with no_grad():
            x_mid = teacher.step(xt, t, t - delta, c)
            x_target = teacher.step(x_mid, t - delta, t - 2 * delta, c)
        x_pred = student.step(xt, t, t - 2 * delta, c)
        loss = mse_or_lpips(x_pred, x_target)
        update(student, loss)

    previous_sampler = student`,
      sample: `# Inference: few-step distilled diffusion sampler
x = normal_noise(shape)
for t_cur, t_next in distilled_schedule:
    x = student.step(x, t_cur, t_next, condition)
return decode_latent_or_image(x)`
    },
    guided: {
      train: `# Algorithm 3: classifier-free guidance distillation
for x0, c in dataloader:
    t = sample_timestep()
    eps = normal_noise_like(x0)
    xt = alpha(t) * x0 + sigma(t) * eps

    with no_grad():
        eps_cond = teacher.eps(xt, t, c)
        eps_uncond = teacher.eps(xt, t, null_condition)
        eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    eps_student = student.eps(xt, t, c)
    loss = mse(eps_student, eps_cfg)
    update(student, loss)`,
      sample: `# Inference: student no longer needs two forward passes for CFG
x = normal_noise(shape)
for t in distilled_schedule:
    eps = student.eps(x, t, condition)
    x = sampler_update(x, eps, t)
return decode(x)`
    },
    consistency: {
      train: `# Algorithm 4: consistency distillation
for x0, c in dataloader:
    t = sample_large_timestep()
    s = sample_smaller_timestep(t)
    eps = normal_noise_like(x0)
    xt = forward_noise(x0, t, eps)

    with no_grad():
        xs = teacher_ode.solve(xt, t_start=t, t_end=s, condition=c)
        target = f_ema(xs, s, c)

    pred = f_theta(xt, t, c)
    loss = distance(pred, target)
    update(f_theta, loss)
    f_ema = ema_update(f_ema, f_theta)`,
      sample: `# One-step consistency inference
xT = normal_noise(shape)
x0 = f_theta(xT, t_max, condition)
return decode(x0)

# Few-step consistency inference
x = normal_noise(shape)
for t_cur, t_next in consistency_schedule:
    x0_hat = f_theta(x, t_cur, condition)
    if t_next == 0:
        x = x0_hat
    else:
        x = forward_noise(x0_hat, t_next, normal_noise_like(x))
return decode(x)`
    },
    dmd: {
      train: `# Algorithm 5: basic DMD training
for iteration in training:
    for _ in range(fake_score_updates):
        z, c = sample_noise_and_condition()
        with no_grad():
            x_fake = G_theta(z, c)
        t = sample_timestep()
        xt = alpha(t) * x_fake + sigma(t) * normal_noise_like(x_fake)
        fake_score_loss = denoising_score_matching_loss(F_phi, xt, t, x_fake, c)
        update(F_phi, fake_score_loss)

    z, c = sample_noise_and_condition()
    x_fake = G_theta(z, c)
    t = sample_timestep()
    xt = alpha(t) * x_fake + sigma(t) * normal_noise_like(x_fake)
    with no_grad():
        s_real = T.score(xt, t, c)
        s_fake = F_phi.score(xt, t, c)
        grad_target = weight(t) * (s_fake - s_real)
    dmd_loss = surrogate_dot(x_fake, grad_target)
    update(G_theta, dmd_loss)`,
      sample: `# Inference: one-step DMD student
z = normal_noise(shape)
x = G_theta(z, condition)
return decode(x)`
    },
    dmd2: {
      train: `# Algorithm 6: DMD2 one-step + multi-step recipe
for iteration in training:
    # A. two-time-scale fake critic update
    for _ in range(k_fake):
        z, c = sample_noise_and_condition()
        with no_grad():
            x_fake = G_theta(z, c)
        xt_fake = noise(x_fake, sample_timestep())
        update(F_phi, denoising_score_matching_loss(F_phi, xt_fake, c))

    # B. GAN discriminator with real data and generated samples
    x_real, c_real = sample_real_batch()
    x_fake = stopgrad(G_theta(sample_noise(), c))
    loss_D = bce(D_psi(noise(x_real), c_real), 1) + bce(D_psi(noise(x_fake), c), 0)
    update(D_psi, loss_D)

    # C. generator with DMD gradient surrogate + GAN loss
    x_fake = G_theta(sample_noise(), c)
    xt_fake = noise(x_fake, sample_timestep())
    s_real = stopgrad(T.score(xt_fake, t, c))
    s_fake = stopgrad(F_phi.score(xt_fake, t, c))
    loss_dmd = surrogate_dot(x_fake, weight(t) * (s_fake - s_real))
    loss_adv = bce(D_psi(xt_fake, t, c), 1)
    update(G_theta, loss_dmd + lambda_adv * loss_adv)

    # D. multi-step backward simulation
    # simulate previous student steps so each stage sees inference-time inputs`,
      sample: `# One-step DMD2 inference
z = normal_noise(shape)
x = G_theta(z, condition)
return decode(x)

# Multi-step DMD2 inference
x = normal_noise(shape)
for t in schedule:
    x0_hat = G_theta(x, t, condition)
    if is_last(t):
        x = x0_hat
    else:
        x = forward_noise_or_consistency_reinject(x0_hat, next_t(t))
return decode(x)`
    },
    add: {
      train: `# Algorithm 7: ADD training
for iteration in training:
    c = sample_prompt()
    z = sample_noise()
    x_fake = G_theta(z, c, num_steps=student_steps)

    t = sample_timestep()
    xt_fake = alpha(t) * x_fake + sigma(t) * normal_noise_like(x_fake)
    with no_grad():
        teacher_score = T.score(xt_fake, t, c)
    loss_score = score_distillation_surrogate(x_fake, teacher_score, t)

    x_real, c_real = sample_real_batch()
    loss_D = bce(D_psi(x_real, c_real), 1) + bce(D_psi(detach(x_fake), c), 0)
    update(D_psi, loss_D)

    loss_adv = bce(D_psi(x_fake, c), 1)
    update(G_theta, lambda_score * loss_score + lambda_adv * loss_adv)`,
      sample: `# ADD inference
z = normal_noise(shape)
x = G_theta(z, prompt, num_steps=1_or_4)
return x`
    },
    ladd: {
      train: `# Algorithm 8: LADD training
for iteration in training:
    c = sample_prompt()
    with no_grad():
        # synthetic target latent produced by the frozen teacher
        z_target = teacher_generate_latent(T, c, cfg=constant_cfg, steps=teacher_steps)

    eps = normal_noise(latent_shape_for(c))
    z_student = G_theta(eps, c, num_steps=student_steps)

    # noise level feedback: high t emphasizes global structure, low t local texture
    t = sample_logit_normal_noise_level()
    zt_target = alpha(t) * z_target + sigma(t) * normal_noise_like(z_target)
    zt_student = alpha(t) * z_student + sigma(t) * normal_noise_like(z_student)

    feats_target = T.extract_block_features(zt_target, t, c)
    feats_student_detached = T.extract_block_features(detach(zt_student), t, c)
    update_discriminator_heads(D_heads, feats_target, feats_student_detached)

    feats_student = T.extract_block_features(zt_student, t, c)
    loss_G = sum_bce_real(D_heads, feats_student)
    update(G_theta, lambda_adv * loss_G + optional_regularizers)`,
      sample: `# LADD / SD3-Turbo style 4-step inference in latent space
latent = normal_noise(latent_shape(prompt, aspect_ratio))
for t in ladd_student_schedule[:4]:
    latent = G_theta.step(latent, t, prompt)
image = vae_decode(latent)
return image`
    },
    flowdistill: {
      train: `# Algorithm 9A: rectified flow / reflow-style training
for iteration in training:
    x0 = sample_base_noise()
    x1 = sample_data_or_teacher_sample()
    t = uniform(0, 1)
    xt = (1 - t) * x0 + t * x1
    target_velocity = x1 - x0
    update(v_theta, mse(v_theta(xt, t, condition), target_velocity))

# Algorithm 9B: average-velocity / shortcut-style training
for iteration in training:
    r, t = sample_interval()
    xr, xt = sample_two_points_on_teacher_or_data_path(r, t)
    avg_velocity = (xt - xr) / (t - r)
    update(vbar_theta, mse(vbar_theta(xt, r, t, condition), avg_velocity))`,
      sample: `# Flow inference with Euler steps
x = sample_base_noise()
for k in range(num_steps):
    t = k / num_steps
    x = x + (1 / num_steps) * v_theta(x, t, condition)
return decode(x)

# One-step or few-step average-velocity inference
x = sample_base_noise()
for r, t in chosen_intervals:
    v_avg = vbar_theta(x, r, t, condition)
    x = x + (t - r) * v_avg
return decode(x)`
    },
    selfforcing: {
      train: `# Algorithm 10A: Self Forcing training for AR video diffusion
for iteration in training:
    c = sample_prompt_or_action_condition()
    X_generated = []
    KV = empty_cache()
    s = randint(1, J)  # stochastic gradient truncation depth

    for i in range(1, N + 1):
        x = normal_noise(chunk_shape)
        for j in range(J, s - 1, -1):
            t_j = denoise_schedule[j]
            if j == s:
                x0_hat = G_theta.denoise(x, t_j, context_kv=KV, condition=c, grad=True)
                X_generated.append(x0_hat)
                kv_i = G_theta.extract_kv(stopgrad(x0_hat), t=0, context_kv=KV, condition=c)
                KV.append(kv_i)
                KV = maybe_roll_cache(KV)
            else:
                with no_grad():
                    x0_hat = G_theta.denoise(x, t_j, context_kv=KV, condition=c)
                    x = forward_noise(x0_hat, denoise_schedule[j - 1])

    loss = holistic_distribution_matching_loss(X_generated, condition=c)
    update(G_theta, loss)`,
      sample: `# Algorithm 10B: Self Forcing inference
KV = empty_cache()
X = []
for i in range(1, M + 1):
    x = normal_noise(chunk_shape)
    for j in range(J, 0, -1):
        t_j = denoise_schedule[j]
        x0_hat = G_theta.denoise(x, t_j, context_kv=KV, condition=c)
        if j > 1:
            x = forward_noise(x0_hat, denoise_schedule[j - 1])
    X.append(x0_hat)
    KV.append(G_theta.extract_kv(x0_hat, t=0, context_kv=KV, condition=c))
    KV = maybe_evict_old_entries(KV, max_cache_frames=L)
    stream_decode_and_emit(x0_hat)
return X`
    },
    selfforcingpp: {
      train: `# Algorithm 11A: Self-Forcing++ long-horizon training
for iteration in training:
    c = sample_prompt_or_action_condition()
    KV = empty_cache()
    X_long = []
    for i in range(1, N + 1):
        x_i = ar_few_step_generate(G_theta, context_kv=KV, condition=c)
        X_long.append(x_i)
        KV.append(G_theta.extract_kv(stopgrad(x_i), t=0, context_kv=KV, condition=c))
        KV = rolling_cache_update(KV)

    start = randint(1, N - H + 1)
    W = X_long[start : start + H]
    t = sample_timestep()
    # backward noise initialization: diffuse the selected clean window back to t
    W_t = alpha(t) * W + sigma(t) * normal_noise_like(W)

    with no_grad():
        s_teacher = T.score_or_velocity(W_t, t, condition=c)
    s_student = F_phi.score(W_t, t, c) if use_fake_score else G_theta.score_proxy(W_t, t, c)
    # extended DMD: align long-rollout windows with the short-horizon teacher
    loss_ext_dmd = surrogate_dot(W, weight(t) * stopgrad(s_student - s_teacher))
    loss = loss_ext_dmd + lambda_rl * optional_group_relative_stability_loss()
    update(G_theta, loss)
    if use_fake_score:
        update_fake_score_model(F_phi, generated_windows=W, timestep=t)`,
      sample: `# Algorithm 11B: Self-Forcing++ long video inference
KV = empty_cache()
video = []
for chunk_idx in range(1, target_num_chunks + 1):
    x_chunk = ar_few_step_generate(G_theta, context_kv=KV, condition=c)
    video.append(x_chunk)
    KV.append(G_theta.extract_kv(x_chunk, t=0, context_kv=KV, condition=c))
    KV = rolling_cache_update(KV, max_cache_chunks=L)
    if stream_output:
        decode_and_emit(x_chunk)
return video`
    }
  };

  const objectText = {
    logits: 'logits matching：学习 teacher 的单步 token 或分类概率。',
    transition: 'transition matching：学习 teacher sampler 的局部或复合 transition。',
    endpoint: 'endpoint consistency：同一条 ODE 轨迹上的不同状态映射到同一终点。',
    distribution: 'distribution matching：不绑定路径，只匹配 teacher 或数据样本云。',
    reward: 'reward matching：匹配偏好或 critic 定义的行为分数。',
    rollout: 'self-rollout matching：训练时暴露 student 推理会遇到的状态分布。'
  };

  function bindAlgorithmPanels() {
    root.querySelectorAll('[data-flow-panel]').forEach((panel) => {
      const key = panel.dataset.flowPanel;
      const output = panel.querySelector('[data-flow-output]');
      const buttons = panel.querySelectorAll('[data-flow-mode]');
      function render(mode) {
        output.textContent = algorithmSnippets[key]?.[mode] || '';
        buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.flowMode === mode));
      }
      buttons.forEach((btn) => btn.addEventListener('click', () => render(btn.dataset.flowMode)));
      render('train');
    });
  }

  function ctxFor(id) {
    const canvas = root.getElementById(id);
    return canvas ? canvas.getContext('2d') : null;
  }

  function clearCanvas(ctx) {
    const { width, height } = ctx.canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#fffdf8';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(23,24,23,0.08)';
    ctx.lineWidth = 1;
    for (let x = 36; x < width; x += 36) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 36; y < height; y += 36) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  }

  function text(ctx, label, x, y, color = '#171817', size = 16, weight = '800') {
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
    ctx.fillText(label, x, y);
  }

  function arrow(ctx, from, to, color = '#0f6f68', width = 4) {
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    const head = 8 + width;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(to.x, to.y);
    ctx.lineTo(to.x - head * Math.cos(angle - Math.PI / 6), to.y - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(to.x - head * Math.cos(angle + Math.PI / 6), to.y - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();
  }

  function point(ctx, x, y, color = '#0f6f68', label = '') {
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    if (label) text(ctx, label, x + 12, y - 10, color, 14, '850');
  }

  function pathPoint(t, width, height, curve) {
    return {
      x: 80 + (width - 160) * t,
      y: height - 58 - (height - 120) * t - Math.sin(Math.PI * t) * curve
    };
  }

  function bindDistillObjectLab() {
    const ctx = ctxFor('distillObjectCanvas');
    if (!ctx) return;
    const readout = root.getElementById('distillObjectReadout');
    const buttons = root.querySelectorAll('[data-object]');
    function draw(mode) {
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      const teacher = { x: 140, y: height * 0.35 };
      const student = { x: width - 160, y: height * 0.35 };
      const sample = { x: width * 0.5, y: height * 0.72 };
      point(ctx, teacher.x, teacher.y, '#314f78', 'teacher');
      point(ctx, student.x, student.y, '#0f6f68', 'student');
      point(ctx, sample.x, sample.y, '#a9432f', 'output');
      if (mode === 'logits') arrow(ctx, teacher, student, '#314f78', 5);
      if (mode === 'transition') {
        arrow(ctx, teacher, sample, '#9a6d19', 4);
        arrow(ctx, sample, student, '#9a6d19', 4);
      }
      if (mode === 'endpoint') {
        [0.18, 0.36, 0.54].forEach((t, i) => {
          const p = pathPoint(t, width, height, 60);
          point(ctx, p.x, p.y, '#0f6f68', `t${i + 1}`);
          arrow(ctx, p, sample, '#0f6f68', 3);
        });
      }
      if (mode === 'distribution') {
        ctx.strokeStyle = '#314f78';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.ellipse(width * 0.38, height * 0.62, 80, 48, -0.2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = '#0f6f68';
        ctx.beginPath();
        ctx.ellipse(width * 0.58, height * 0.58, 96, 54, 0.2, 0, Math.PI * 2);
        ctx.stroke();
        arrow(ctx, { x: width * 0.44, y: height * 0.58 }, { x: width * 0.53, y: height * 0.57 }, '#a9432f', 4);
      }
      if (mode === 'reward') {
        arrow(ctx, sample, { x: sample.x, y: height * 0.28 }, '#a9432f', 5);
        text(ctx, 'reward / preference', sample.x - 78, height * 0.24, '#a9432f', 16);
      }
      if (mode === 'rollout') {
        for (let i = 0; i < 6; i += 1) {
          const x = 170 + i * 90;
          ctx.strokeStyle = i < 3 ? '#0f6f68' : '#a9432f';
          ctx.strokeRect(x, height * 0.58 + Math.sin(i) * 16, 58, 40);
          if (i > 0) arrow(ctx, { x: x - 32, y: height * 0.64 }, { x: x - 4, y: height * 0.64 }, i < 3 ? '#0f6f68' : '#a9432f', 3);
        }
      }
      readout.textContent = objectText[mode];
      buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.object === mode));
    }
    buttons.forEach((btn) => btn.addEventListener('click', () => draw(btn.dataset.object)));
    draw('logits');
  }

  function bindTrajectoryLab() {
    const ctx = ctxFor('trajectoryCanvas');
    if (!ctx) return;
    const curve = root.getElementById('trajCurve');
    const step = root.getElementById('trajStep');
    const round = root.getElementById('trajRound');
    function draw() {
      const c = Number(curve.value);
      const s = Number(step.value);
      const r = Number(round.value);
      root.getElementById('trajCurveOut').textContent = c;
      root.getElementById('trajStepOut').textContent = s;
      root.getElementById('trajRoundOut').textContent = r;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      ctx.setLineDash([6, 6]);
      ctx.strokeStyle = 'rgba(23,24,23,0.36)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 120; i += 1) {
        const p = pathPoint(i / 120, width, height, c);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      for (let i = 0; i < s; i += 1) {
        const a = pathPoint(i / s, width, height, c);
        const b = pathPoint((i + 1) / s, width, height, c);
        arrow(ctx, a, b, '#0f6f68', 4);
      }
      point(ctx, 80, height - 58, '#171817', 'x_T');
      point(ctx, width - 80, 62, '#a9432f', 'x_0');
      const error = Math.max(0.02, (c / 170) * (s / 8) / (r + 1));
      root.getElementById('trajectoryReadout').textContent = `NFE reduction = ${Math.pow(2, r)}x · toy error = ${error.toFixed(2)}`;
    }
    [curve, step, round].forEach((el) => el.addEventListener('input', draw));
    draw();
  }

  function bindConsistencyLab() {
    const ctx = ctxFor('consistencyCanvas');
    if (!ctx) return;
    const noise = root.getElementById('consNoise');
    function draw() {
      const n = Number(noise.value);
      root.getElementById('consNoiseOut').textContent = n;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      const endpoint = { x: width - 120, y: height * 0.48 };
      for (let i = 0; i < 5; i += 1) {
        const t = 0.12 + i * 0.14;
        const p = pathPoint(t, width, height, n);
        point(ctx, p.x, p.y, '#0f6f68', `x_${i}`);
        arrow(ctx, p, endpoint, '#0f6f68', 3);
      }
      point(ctx, endpoint.x, endpoint.y, '#a9432f', 'same endpoint');
      root.getElementById('consistencyReadout').textContent = `consistency error = ${(n / 180).toFixed(2)}`;
    }
    noise.addEventListener('input', draw);
    draw();
  }

  function bindDmdLab() {
    const ctx = ctxFor('dmdCanvas');
    if (!ctx) return;
    const mode = root.getElementById('dmdMode');
    const critic = root.getElementById('dmdCritic');
    const gan = root.getElementById('dmdGan');
    function draw() {
      const c = Number(critic.value);
      const g = Number(gan.value);
      root.getElementById('dmdCriticOut').textContent = c;
      root.getElementById('dmdGanOut').textContent = g;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      if (mode.value === 'trajectory') {
        for (let i = 0; i < 5; i += 1) {
          const z = { x: 110, y: 82 + i * 48 };
          const target = { x: width - 130, y: 70 + i * 48 + Math.sin(i) * 16 };
          point(ctx, z.x, z.y, '#314f78', i === 0 ? 'same z' : '');
          point(ctx, target.x, target.y, '#a9432f', '');
          arrow(ctx, z, target, '#314f78', 3);
        }
      } else {
        ctx.strokeStyle = '#314f78';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.ellipse(width * 0.68, height * 0.46, 110, 72, -0.22, 0, Math.PI * 2);
        ctx.stroke();
        for (let i = 0; i < 22; i += 1) {
          const x = 130 + (i % 6) * 32 + (c / 100) * 120;
          const y = height * 0.62 + Math.sin(i * 1.7) * 50 + Math.floor(i / 6) * 18 - (g / 100) * 45;
          point(ctx, x, y, i % 2 ? '#0f6f68' : '#a9432f');
        }
        arrow(ctx, { x: width * 0.42, y: height * 0.58 }, { x: width * 0.58, y: height * 0.47 }, '#a9432f', 5);
      }
      const coverage = mode.value === 'distribution' ? Math.max(5, 100 - g + c * 0.2) : 45;
      const sharpness = Math.min(100, g + c * 0.35);
      root.getElementById('dmdReadout').textContent = `coverage score = ${coverage.toFixed(0)} · sharpness = ${sharpness.toFixed(0)}`;
    }
    [mode, critic, gan].forEach((el) => el.addEventListener('input', draw));
    mode.addEventListener('change', draw);
    draw();
  }

  function bindLossBalanceLab() {
    const score = root.getElementById('lossScore');
    if (!score) return;
    const dmd = root.getElementById('lossDmd');
    const adv = root.getElementById('lossAdv');
    const reward = root.getElementById('lossReward');
    function setBar(id, value) {
      root.getElementById(id).style.setProperty('--value', `${Math.max(4, Math.min(100, value))}%`);
    }
    function update() {
      const s = Number(score.value);
      const d = Number(dmd.value);
      const a = Number(adv.value);
      const r = Number(reward.value);
      root.getElementById('lossScoreOut').textContent = s;
      root.getElementById('lossDmdOut').textContent = d;
      root.getElementById('lossAdvOut').textContent = a;
      root.getElementById('lossRewardOut').textContent = r;
      setBar('metricSharp', a * 0.72 + s * 0.26);
      setBar('metricCoverage', d * 0.8 + s * 0.15 - a * 0.24);
      setBar('metricTemporal', s * 0.42 + r * 0.55 + d * 0.12);
    }
    [score, dmd, adv, reward].forEach((el) => el.addEventListener('input', update));
    update();
  }

  function bindLaddLab() {
    const ctx = ctxFor('laddCanvas');
    if (!ctx) return;
    const noise = root.getElementById('laddNoise');
    function draw() {
      const n = Number(noise.value);
      root.getElementById('laddNoiseOut').textContent = n;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      const left = 90;
      const mid = width * 0.46;
      const right = width - 150;
      point(ctx, left, height * 0.55, '#314f78', 'latent');
      point(ctx, mid, height * 0.38, '#0f6f68', 'teacher features');
      point(ctx, right, height * 0.55, '#a9432f', 'heads');
      arrow(ctx, { x: left + 60, y: height * 0.54 }, { x: mid - 30, y: height * 0.42 }, '#0f6f68', 5);
      arrow(ctx, { x: mid + 50, y: height * 0.42 }, { x: right - 38, y: height * 0.54 }, '#a9432f', 5);
      const global = n;
      const local = 100 - n;
      ctx.fillStyle = '#314f78';
      ctx.fillRect(140, height - 72, global * 2.2, 18);
      ctx.fillStyle = '#a9432f';
      ctx.fillRect(140, height - 42, local * 2.2, 18);
      text(ctx, `global structure ${global}`, 140, height - 82, '#314f78', 14);
      text(ctx, `local texture ${local}`, 140, height - 52, '#a9432f', 14);
    }
    noise.addEventListener('input', draw);
    draw();
  }

  function bindSelfForcingLab() {
    const ctx = ctxFor('selfForcingCanvas');
    if (!ctx) return;
    const mode = root.getElementById('sfMode');
    const length = root.getElementById('sfLength');
    const cache = root.getElementById('sfCache');
    function draw() {
      const l = Number(length.value);
      const c = Number(cache.value);
      root.getElementById('sfLengthOut').textContent = l;
      root.getElementById('sfCacheOut').textContent = c;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      const gap = Math.min(58, (width - 120) / l);
      for (let i = 0; i < l; i += 1) {
        const x = 55 + i * gap;
        const generated = mode.value === 'self' || i > Math.floor(l * 0.45);
        ctx.fillStyle = generated ? 'rgba(169,67,47,0.12)' : 'rgba(15,111,104,0.12)';
        ctx.strokeStyle = generated ? '#a9432f' : '#0f6f68';
        ctx.lineWidth = 2;
        ctx.fillRect(x, height * 0.42 + Math.sin(i) * 12, 42, 34);
        ctx.strokeRect(x, height * 0.42 + Math.sin(i) * 12, 42, 34);
        if (i > 0) arrow(ctx, { x: x - 18, y: height * 0.47 }, { x: x - 2, y: height * 0.47 }, generated ? '#a9432f' : '#0f6f68', 2);
      }
      const start = Math.max(0, l - c);
      ctx.strokeStyle = '#314f78';
      ctx.lineWidth = 4;
      ctx.strokeRect(50 + start * gap, height * 0.35, Math.max(44, c * gap), 78);
      text(ctx, 'rolling KV cache window', 50 + start * gap, height * 0.31, '#314f78', 15);
      const warning = mode.value === 'teacher' ? 'high' : c < 4 ? 'medium' : 'lower';
      root.getElementById('sfReadout').textContent = `train-test gap warning = ${warning}`;
    }
    [mode, length, cache].forEach((el) => el.addEventListener('input', draw));
    mode.addEventListener('change', draw);
    draw();
  }

  function bindToc() {
    const toggle = root.getElementById('tocToggle');
    const toc = root.getElementById('lectureToc');
    if (toggle && toc) {
      toggle.addEventListener('click', () => {
        const open = toc.classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
      });
    }
    const links = [...root.querySelectorAll('.lecture-toc a')];
    const targets = links.map((link) => root.querySelector(link.getAttribute('href'))).filter(Boolean);
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          links.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`));
        });
      }, { rootMargin: '-25% 0px -65% 0px' });
      targets.forEach((target) => observer.observe(target));
    }
  }

  function bindProgress() {
    const bar = root.getElementById('readingProgressBar');
    if (!bar) return;
    function update() {
      const max = root.documentElement.scrollHeight - window.innerHeight;
      const percent = max > 0 ? (window.scrollY / max) * 100 : 0;
      bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    }
    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
  }

  function init() {
    bindAlgorithmPanels();
    bindDistillObjectLab();
    bindTrajectoryLab();
    bindConsistencyLab();
    bindDmdLab();
    bindLossBalanceLab();
    bindLaddLab();
    bindSelfForcingLab();
    bindToc();
    bindProgress();
  }

  if (root.readyState === 'loading') {
    root.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
