(function () {
  const root = document;

  const algorithmSnippets = {
    roofline: {
      train: `# Training profiler view
for step in training:
    flops = estimate_forward_backward_flops(model, batch)
    bytes_moved = estimate_activation_weight_optimizer_io(model, batch)
    comm = estimate_collectives(parallel_plan)
    latency = max(flops / peak_flops, bytes_moved / bandwidth) + comm
    log_by_component(step, latency)

# Use the result to choose:
# - FlashAttention / fused kernels if memory-bound
# - activation checkpointing if activation memory dominates
# - FSDP / tensor parallel if model states or layer FLOPs dominate`,
      infer: `# Inference latency lower bound
def estimate_step(flops, bytes_moved, peak_flops, bandwidth):
    compute_time = flops / peak_flops
    memory_time = bytes_moved / bandwidth
    return {
        "compute_time": compute_time,
        "memory_time": memory_time,
        "bottleneck": "compute" if compute_time > memory_time else "memory",
        "lower_bound": max(compute_time, memory_time),
    }`
    },
    flash: {
      train: `# Exact blocked attention with recomputation-friendly backward
for q_block in blocks(Q):
    running_max = -inf
    running_sum = 0
    output = 0
    for k_block, v_block in blocks(K, V):
        scores = q_block @ k_block.T / sqrt(d)
        block_max = row_max(scores)
        block_probs = exp(scores - block_max)
        new_max = maximum(running_max, block_max)
        output = rescale_old(output, running_max, new_max)
        output += rescale_new(block_probs @ v_block, block_max, new_max)
        running_sum = update_denominator(running_sum, running_max, block_probs, new_max)
        running_max = new_max
    write(output / running_sum)

# Backward can recompute score blocks instead of saving T x T attention.`,
      infer: `# Prefill path
K, V = project_prompt_to_kv(prompt)
O = flash_attention(Q, K, V, causal=True)

# Decode path usually combines:
# - one-token query
# - KV cache reads
# - Flash-Decoding / paged KV layout
# - GQA/MQA to reduce KV bandwidth`
    },
    kv: {
      train: `# Standard teacher-forcing training does not need KV cache
tokens = packed_batch(input_ids)
hidden = transformer(tokens, causal_mask=True)
loss = cross_entropy(hidden, labels)

# For training long contexts, prefer:
# FlashAttention, sequence packing, sequence parallel,
# activation checkpointing, and careful batch construction.`,
      infer: `def decode_one_token(model, token_t, kv_cache):
    hidden = embed(token_t)
    for layer_id, layer in enumerate(model.layers):
        q_t, k_t, v_t = layer.project_qkv(hidden)
        kv_cache[layer_id].K.append(k_t)
        kv_cache[layer_id].V.append(v_t)
        K_all = kv_cache[layer_id].K
        V_all = kv_cache[layer_id].V
        hidden = attention(q_t, K_all, V_all)
        hidden = layer.mlp(layer.norm(hidden))
    logits = model.lm_head(hidden)
    return sample(logits), kv_cache`
    },
    paged: {
      train: `# Paged KV is a serving-time memory manager.
# Training normally uses packed dense tensors instead.
# Related training idea: sequence packing reduces padding waste,
# but it does not manage persistent decode KV pages.`,
      infer: `class PagedKVCache:
    def __init__(self, block_size, num_blocks):
        self.free = Queue(range(num_blocks))
        self.table = {}       # request_id -> list[block_id]
        self.ref_count = defaultdict(int)

    def append_kv(self, request_id, k_t, v_t):
        if current_block_full(request_id):
            block = self.free.pop()
            self.table.setdefault(request_id, []).append(block)
            self.ref_count[block] = 1
        write_to_last_block(request_id, k_t, v_t)

    def share_prefix(self, new_request, owner, num_blocks):
        shared = self.table[owner][:num_blocks]
        self.table[new_request] = list(shared)
        for block in shared:
            self.ref_count[block] += 1

    def copy_on_write(self, request_id, block):
        if self.ref_count[block] > 1:
            new_block = self.free.pop()
            copy_block(block, new_block)
            replace_block(request_id, block, new_block)`
    },
    batching: {
      train: `# Training analogue: dynamic packing / bucketing
for batch in bucket_by_length(dataset):
    packed = pack_sequences(batch, max_tokens_per_batch)
    loss = model(packed).loss
    update(model, loss)

# Goal: reduce padding and keep accelerator utilization high.`,
      infer: `def continuous_batching_loop(engine):
    waiting = RequestQueue()
    active = []
    while True:
        while waiting and engine.has_capacity():
            req = waiting.pop()
            req.kv_cache = engine.prefill(req.prompt)
            active.append(req)

        batch_inputs = [req.last_token for req in active]
        next_tokens = engine.decode_step(batch_inputs, [r.kv_cache for r in active])

        still_active = []
        for req, tok in zip(active, next_tokens):
            req.append(tok)
            if req.is_finished():
                engine.free(req.kv_cache)
                emit(req.output)
            else:
                still_active.append(req)
        active = still_active`
    },
    quant: {
      train: `def post_training_quantization(model, calibration_loader, method):
    stats = {}
    for batch in calibration_loader:
        with no_grad():
            acts = forward_and_hook(model, batch)
            update_stats(stats, acts)

    for layer in model.layers:
        if method == "smoothquant":
            S = compute_smooth_scale(stats[layer], layer.weight)
            layer.weight = inverse_scale_weight(layer.weight, S)
        elif method == "awq":
            S = activation_aware_scale(stats[layer], layer.weight)
            layer.weight = protect_salient_channels(layer.weight, S)
        elif method == "gptq":
            layer.weight = gptq_quantize(layer.weight, stats[layer].hessian)
        else:
            layer.weight = uniform_quantize(layer.weight, bits=4)
    return export_quantized_model(model)`,
      infer: `def quantized_linear(x_fp16, W_int4, scale, zero_point=None):
    # Real kernels fuse unpack/dequant inside matmul.
    W = dequantize(W_int4, scale, zero_point)
    y = x_fp16 @ W.T
    return y

# Better runtime kernel:
# load packed int4 -> unpack in registers ->
# multiply with fp16/bf16 activations ->
# accumulate -> write output`
    },
    spec: {
      train: `# Speculative decoding itself is inference-time.
# Optional training targets:
for prompt in corpus:
    draft_logits = draft_model(prompt)
    target_logits = target_model(prompt).detach()
    loss = KL(softmax(target_logits) || softmax(draft_logits))
    update(draft_model, loss)

# Alternatives:
# - train multi-token prediction heads
# - train tree proposal policy
# - train feature-level draft heads`,
      infer: `def speculative_decode(target, draft, prompt, K):
    tokens = list(prompt)
    while not done(tokens):
        proposals, q_probs = draft.propose(tokens, K)
        p_probs = target.verify_positions(tokens, proposals)

        accepted_all = True
        for i, y in enumerate(proposals):
            accept = min(1.0, p_probs[i][y] / q_probs[i][y])
            if uniform(0, 1) < accept:
                tokens.append(y)
            else:
                residual = normalize(positive_part(p_probs[i] - draft_probs_at(i)))
                tokens.append(sample(residual))
                accepted_all = False
                break
        if accepted_all:
            tokens.append(sample(target.next_token_probs(tokens)))
    return tokens`
    },
    diffusion: {
      train: `# Fast samplers usually do not change base training
for x0, condition in dataloader:
    t = sample_timestep()
    noise = randn_like(x0)
    xt = alpha(t) * x0 + sigma(t) * noise
    pred = model(xt, t, condition)
    loss = mse(pred, target(noise, x0, t))
    update(model, loss)

# Distillation is a separate route when the training objective changes.`,
      infer: `def fast_diffusion_sample(model, scheduler, condition, num_steps):
    x = randn(latent_shape)
    history = []
    for t in scheduler.select_timesteps(num_steps):
        pred = model(x, t, condition)
        # DDIM: deterministic first-order update
        # DPM-Solver: high-order ODE update using history
        # UniPC: predictor-corrector update
        x_next = scheduler.step(x, t, pred, history)
        history.append((t, x, pred))
        x = x_next
    return decode_latent(x)`
    },
    cache: {
      train: `# Feature cache is mainly inference-time.
# If used during training, train with the same cache policy
# so the model sees approximation noise.
for x0, c in dataloader:
    cache_policy = sample_cache_policy()
    pred = model_with_cache_policy(noise(x0), c, cache_policy)
    loss = diffusion_loss(pred, x0)
    update(model, loss)`,
      infer: `def cached_diffusion_sample(model, scheduler, condition, cache_policy):
    x = randn(latent_shape)
    cache = {}
    for step, t in enumerate(scheduler.timesteps()):
        h = model.input_embed(x, t, condition)
        for layer_id, block in enumerate(model.blocks):
            key = (layer_id, cache_policy.group(t, step))
            cached = cache.get(key)
            if cache_policy.should_reuse(layer_id, step, t, h, cached):
                h = cached
            else:
                h = block(h, t, condition)
                if cache_policy.should_store(layer_id, step, t):
                    cache[key] = detach(h)
        pred = model.output_head(h)
        x = scheduler.step(x, t, pred)
    return decode_latent(x)`
    },
    checkpoint: {
      train: `def forward_with_checkpointing(layers, x, interval):
    for start in range(0, len(layers), interval):
        end = start + interval
        # Save only segment input; recompute internals in backward.
        x = checkpoint(run_layers, layers[start:end], x)
    return x

# Saves activation memory, costs extra forward recompute.`,
      infer: `# Inference has no backward pass, so activation checkpointing
# usually does not help latency.
# Related inference memory tools:
# - KV cache paging / quantization
# - weight quantization
# - activation memory planning in compiled runtime`
    },
    fsdp: {
      train: `def fsdp_training_step(model_shards, batch):
    hidden = batch.inputs
    for layer in model.layers:
        W_full = all_gather(layer.weight_shard)
        hidden = layer.forward(hidden, W_full)
        free(W_full)

    loss = compute_loss(hidden, batch.labels)
    loss.backward()

    for layer in reversed(model.layers):
        grad_shard = reduce_scatter(layer.full_grad)
        optimizer.update(layer.weight_shard, grad_shard, layer.state_shard)`,
      infer: `# Inference can also shard weights with tensor/pipeline parallel,
# but FSDP/ZeRO-3 training logic is not automatically the fastest serving plan.
# Serving often prefers:
# - tensor parallel for large matmuls
# - pipeline parallel for very deep models
# - paged KV + continuous batching for decode throughput`
    },
    lora: {
      train: `def lora_finetune(base_model, dataset, rank):
    freeze(base_model.parameters())
    for linear in target_linear_layers(base_model):
        linear.lora_A = Parameter(randn(rank, linear.in_features) * 0.01)
        linear.lora_B = Parameter(zeros(linear.out_features, rank))

    optimizer = AdamW(lora_parameters(base_model))
    for batch in dataset:
        logits = base_model(batch.input_ids, use_lora=True)
        loss = cross_entropy(logits, batch.labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`,
      infer: `def merge_lora(linear):
    delta = (linear.alpha / linear.rank) * (linear.lora_B @ linear.lora_A)
    linear.weight.data += delta
    remove_lora_modules(linear)

# If not merged, inference may add a low-rank matmul.
# Multi-tenant adapter serving also needs adapter cache scheduling.`
    },
    compile: {
      train: `# Training compiler can fuse forward/backward graphs,
# but dynamic shapes and graph breaks may reduce gains.
graph = trace_forward_backward(model, example_batch)
graph = fuse_ops(graph)
graph = schedule_recompute_and_collectives(graph)
compiled_step = build_training_runtime(graph)`,
      infer: `def compile_for_inference(model, example_inputs):
    graph = trace(model, example_inputs)
    graph = fuse_ops(graph)
    graph = choose_kernels(graph, hardware="cuda")
    graph = specialize_shapes(graph, example_inputs.shapes)
    graph = capture_cuda_graph(graph)
    return build_runtime_engine(graph)`
    },
    moe: {
      train: `def moe_forward_train(x, experts, router, top_k=2):
    scores = softmax(router(x))
    selected = topk(scores, k=top_k)
    y = dispatch_tokens_to_experts(x, selected)
    y = all_to_all(y)
    y = expert_parallel_forward(y, experts)
    y = all_to_all(y)
    loss_aux = load_balance_loss(selected)
    return combine_expert_outputs(y), loss_aux`,
      infer: `def moe_forward(x, experts, router, top_k=2):
    scores = softmax(router(x))
    selected = topk(scores, k=top_k)
    y = zeros_like(x)
    for expert_id, weight in selected:
        y += weight * experts[expert_id](x)
    return y

# Real serving must handle expert capacity, all-to-all,
# router imbalance, and small-batch underutilization.`
    }
  };

  function $(id) {
    return root.getElementById(id);
  }

  function all(selector) {
    return Array.from(root.querySelectorAll(selector));
  }

  function value(id) {
    const el = $(id);
    return el ? Number(el.value) : 0;
  }

  function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = text;
  }

  function bindAlgorithmPanels() {
    all('[data-flow-panel]').forEach((panel) => {
      const key = panel.dataset.flowPanel;
      const output = panel.querySelector('[data-flow-output]');
      const buttons = Array.from(panel.querySelectorAll('[data-flow-mode]'));
      if (output) {
        output.setAttribute('role', 'tabpanel');
      }
      function render(mode) {
        const snippet = algorithmSnippets[key]?.[mode];
        if (output && snippet) {
          output.textContent = snippet;
        }
        buttons.forEach((btn) => {
          const active = btn.dataset.flowMode === mode;
          btn.classList.toggle('active', active);
          btn.setAttribute('role', 'tab');
          btn.setAttribute('aria-selected', String(active));
          btn.setAttribute('tabindex', active ? '0' : '-1');
        });
      }
      buttons.forEach((btn) => btn.addEventListener('click', () => render(btn.dataset.flowMode)));
      render('train');
    });
  }

  function ctxFor(id) {
    const canvas = $(id);
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

  function text(ctx, label, x, y, color = '#171817', size = 15, weight = '800') {
    ctx.fillStyle = color;
    ctx.font = `${weight} ${size}px ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif`;
    ctx.fillText(label, x, y);
  }

  function bar(ctx, x, y, w, h, pct, color, label) {
    ctx.fillStyle = '#f0ece4';
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w * Math.max(0, Math.min(1, pct)), h);
    ctx.strokeStyle = '#d7cdbc';
    ctx.strokeRect(x, y, w, h);
    text(ctx, label, x, y - 8, '#171817', 13, '850');
  }

  function line(ctx, pts, color = '#0f6f68', width = 3) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    ctx.beginPath();
    pts.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }

  function drawMetricBars(containerId, rows) {
    const host = $(containerId);
    if (!host) return;
    const max = Math.max(...rows.map((row) => row.value), 1);
    host.innerHTML = rows.map((row) => {
      const pct = Math.max(3, (row.value / max) * 100);
      return `<div class="${row.className || 'memory-bar'}"><span>${row.label}</span><i style="--value:${pct}%"></i><span>${row.display}</span></div>`;
    }).join('');
  }

  function bindBottleneckMap() {
    const profiles = {
      'llm-decode': {
        rows: [['KV bandwidth', 82], ['serial token', 70], ['schedule gap', 46], ['FLOPs', 32]],
        text: 'decode 常被 KV cache 带宽、串行 token 和 batch 利用率限制。'
      },
      'llm-prefill': {
        rows: [['attention FLOPs', 78], ['HBM traffic', 63], ['kernel IO', 58], ['schedule gap', 18]],
        text: 'prefill 更接近长序列 attention 计算和 IO 的混合瓶颈。'
      },
      diffusion: {
        rows: [['NFE', 88], ['UNet/DiT step', 66], ['VAE decode', 22], ['schedule gap', 12]],
        text: 'diffusion 的总延迟通常先看 NFE，再看单步 denoiser 是否太慢。'
      },
      training: {
        rows: [['activation memory', 75], ['optimizer state', 68], ['collectives', 52], ['dataloader', 28]],
        text: '训练慢常是显存、通信和数据输入共同造成，不能只看单卡 FLOPs。'
      }
    };
    const buttons = all('[data-bottleneck-mode]');
    function render(mode) {
      const profile = profiles[mode];
      drawMetricBars('bottleneckBars', profile.rows.map(([label, value]) => ({
        label,
        value,
        display: `${value}%`,
        className: 'bottleneck-bar'
      })));
      setText('bottleneckReadout', profile.text);
      buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.bottleneckMode === mode));
    }
    buttons.forEach((btn) => btn.addEventListener('click', () => render(btn.dataset.bottleneckMode)));
    if (buttons.length) render('llm-decode');
  }

  function initRooflineLab() {
    const ctx = ctxFor('rooflineCanvas');
    if (!ctx) return;
    const ids = ['roofFlops', 'roofBytes', 'roofPeak', 'roofBandwidth', 'roofBatch', 'roofPrecision'];
    function draw() {
      const flops = value('roofFlops');
      const bytes = value('roofBytes') * value('roofPrecision') / 2;
      const peak = value('roofPeak');
      const bandwidth = value('roofBandwidth') * 1000;
      const batch = value('roofBatch');
      const computeMs = (flops / peak) * 1000 / Math.sqrt(batch);
      const memoryMs = (bytes / bandwidth) * 1000;
      const lower = Math.max(computeMs, memoryMs);
      ids.forEach((id) => setText(`${id}Out`, id === 'roofBandwidth' ? value(id).toFixed(1) : String(value(id))));
      clearCanvas(ctx);
      const max = Math.max(computeMs, memoryMs, 1);
      bar(ctx, 120, 120, 600, 44, computeMs / max, '#314f78', `compute ${computeMs.toFixed(1)} ms`);
      bar(ctx, 120, 210, 600, 44, memoryMs / max, '#0f6f68', `memory ${memoryMs.toFixed(1)} ms`);
      const bottleneck = computeMs > memoryMs ? 'compute-bound' : 'memory-bound';
      text(ctx, bottleneck, 120, 320, computeMs > memoryMs ? '#314f78' : '#0f6f68', 24, '900');
      setText('rooflineReadout', `教学估计 lower bound = ${lower.toFixed(1)} ms · ${bottleneck} · arithmetic intensity = ${(flops / Math.max(bytes, 1)).toFixed(2)} TF/GB · batch scaling is a toy sqrt(B) model`);
    }
    ids.forEach((id) => $(id)?.addEventListener('input', draw));
    draw();
  }

  function initAttentionMemoryLab() {
    const ctx = ctxFor('attentionCanvas');
    if (!ctx) return;
    function draw() {
      const T = value('attnSeq');
      const d = value('attnDim');
      const mode = $('attnMode').value;
      setText('attnSeqOut', String(T));
      setText('attnDimOut', String(d));
      const matrixGb = (T * T * 2) / 1024 ** 3;
      const qkvGb = (T * d * 3 * 2) / 1024 ** 3;
      const traffic = mode === 'standard' ? matrixGb * 4 + qkvGb : qkvGb + matrixGb * 0.18;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      if (mode === 'standard') {
        ctx.fillStyle = 'rgba(169,67,47,0.16)';
        ctx.fillRect(120, 70, 260, 260);
        ctx.strokeStyle = '#a9432f';
        ctx.strokeRect(120, 70, 260, 260);
        text(ctx, 'materialized T x T scores', 142, 202, '#a9432f', 18, '900');
        bar(ctx, 470, 128, 300, 30, Math.min(1, traffic / 80), '#a9432f', 'HBM traffic');
      } else {
        const block = 58;
        for (let y = 72; y < 308; y += block) {
          for (let x = 120; x < 356; x += block) {
            ctx.fillStyle = ((x + y) / block) % 2 ? 'rgba(15,111,104,0.16)' : 'rgba(49,79,120,0.12)';
            ctx.fillRect(x, y, block - 6, block - 6);
            ctx.strokeStyle = '#0f6f68';
            ctx.strokeRect(x, y, block - 6, block - 6);
          }
        }
        text(ctx, 'streamed blocks + online softmax', 125, 340, '#0f6f68', 18, '900');
        bar(ctx, 470, 128, 300, 30, Math.min(1, traffic / 80), '#0f6f68', 'HBM traffic');
      }
      bar(ctx, 470, 218, 300, 30, Math.min(1, matrixGb / 16), '#314f78', 'score matrix size');
      setText('attentionReadout', `attention matrix = ${matrixGb.toFixed(2)} GB · estimated HBM traffic = ${traffic.toFixed(2)} GB · materialized = ${mode === 'standard' ? 'yes' : 'no'}`);
    }
    ['attnSeq', 'attnDim', 'attnMode'].forEach((id) => $(id)?.addEventListener('input', draw));
    $('attnMode')?.addEventListener('change', draw);
    draw();
  }

  function initKVCacheLab() {
    const ids = ['kvLayers', 'kvContext', 'kvBatch', 'kvHq', 'kvHkv', 'kvDim'];
    function gb(hkv) {
      return 2 * value('kvBatch') * value('kvLayers') * value('kvContext') * hkv * value('kvDim') * 2 / 1024 ** 3;
    }
    function draw() {
      ids.forEach((id) => setText(`${id}Out`, String(value(id))));
      const hq = value('kvHq');
      const gqa = Math.min(value('kvHkv'), hq);
      const rows = [
        { label: 'MHA', value: gb(hq), display: `${gb(hq).toFixed(1)} GB` },
        { label: 'GQA', value: gb(gqa), display: `${gb(gqa).toFixed(1)} GB` },
        { label: 'MQA', value: gb(1), display: `${gb(1).toFixed(1)} GB` }
      ];
      drawMetricBars('kvMemoryBars', rows);
      setText('kvReadout', `GQA cache = ${gb(gqa).toFixed(1)} GB · reduction vs MHA = ${(100 * (1 - gb(gqa) / gb(hq))).toFixed(1)}%`);
    }
    ids.forEach((id) => $(id)?.addEventListener('input', draw));
    draw();
  }

  function initBatchingLab() {
    const ctx = ctxFor('batchingCanvas');
    if (!ctx) return;
    function draw() {
      const slots = value('batchSlots');
      const variance = value('batchVariance');
      const arrival = value('batchArrival');
      setText('batchSlotsOut', String(slots));
      setText('batchVarianceOut', String(variance));
      setText('batchArrivalOut', String(arrival));
      clearCanvas(ctx);
      const { width } = ctx.canvas;
      const left = 110;
      const rowH = 20;
      const scale = (width - 180) / 100;
      const lengths = Array.from({ length: slots }, (_, i) => 18 + ((i * 19 + variance) % 70));
      text(ctx, 'static batch', 24, 80, '#171817', 16, '900');
      text(ctx, 'continuous batch', 24, 245, '#171817', 16, '900');
      let staticBusy = 0;
      let staticTotal = slots * Math.max(...lengths);
      lengths.forEach((len, i) => {
        const y = 95 + i * rowH;
        ctx.fillStyle = 'rgba(169,67,47,0.16)';
        ctx.fillRect(left, y, Math.max(...lengths) * scale, 12);
        ctx.fillStyle = '#a9432f';
        ctx.fillRect(left, y, len * scale, 12);
        staticBusy += len;
      });
      let contBusy = 0;
      let contTotal = slots * 100;
      for (let i = 0; i < slots; i += 1) {
        let cursor = 0;
        const y = 260 + i * rowH;
        while (cursor < 100) {
          const len = 12 + ((i * 13 + cursor * 3 + variance) % 28);
          ctx.fillStyle = i % 2 ? '#0f6f68' : '#314f78';
          ctx.fillRect(left + cursor * scale, y, Math.min(len, 100 - cursor) * scale - 2, 12);
          contBusy += Math.min(len, 100 - cursor);
          cursor += len + Math.max(1, 9 - arrival);
        }
      }
      const staticUtil = staticBusy / staticTotal;
      const contUtil = Math.min(0.98, contBusy / contTotal);
      bar(ctx, 555, 70, 250, 18, staticUtil, '#a9432f', `static util ${(staticUtil * 100).toFixed(0)}%`);
      bar(ctx, 555, 235, 250, 18, contUtil, '#0f6f68', `continuous util ${(contUtil * 100).toFixed(0)}%`);
      setText('batchingReadout', `continuous batching utilization = ${(contUtil * 100).toFixed(0)}% · static batch utilization = ${(staticUtil * 100).toFixed(0)}%`);
    }
    ['batchSlots', 'batchVariance', 'batchArrival'].forEach((id) => $(id)?.addEventListener('input', draw));
    draw();
  }

  function initQuantizationLab() {
    const ctx = ctxFor('quantCanvas');
    if (!ctx) return;
    function draw() {
      const bits = value('quantBits');
      const outlier = value('quantOutlier');
      const scaleType = $('quantScale').value;
      const method = $('quantMethod').value;
      setText('quantOutlierOut', String(outlier));
      const levels = 2 ** bits;
      const smooth = method === 'smooth' ? 0.55 : method === 'awq' ? 0.72 : 1.0;
      const channelBonus = scaleType === 'channel' ? 0.72 : 1.0;
      const error = (outlier / 100) * (18 / Math.max(bits, 2)) * smooth * channelBonus;
      clearCanvas(ctx);
      const { width, height } = ctx.canvas;
      const baseY = height * 0.55;
      ctx.strokeStyle = '#d7cdbc';
      ctx.beginPath();
      ctx.moveTo(70, baseY);
      ctx.lineTo(width - 70, baseY);
      ctx.stroke();
      for (let i = 0; i < 120; i += 1) {
        const x = 80 + (i / 119) * (width - 160);
        const amp = 55 * Math.sin(i * 0.22) + 22 * Math.sin(i * 0.61);
        const spike = i === 98 ? outlier * 1.2 : 0;
        const y = baseY - amp - spike;
        ctx.fillStyle = i === 98 ? '#a9432f' : '#314f78';
        ctx.fillRect(x, y, 3, Math.max(3, baseY - y));
      }
      const gridCount = Math.min(levels, 32);
      for (let i = 0; i <= gridCount; i += 1) {
        const x = 80 + (i / gridCount) * (width - 160);
        ctx.strokeStyle = 'rgba(15,111,104,0.32)';
        ctx.beginPath();
        ctx.moveTo(x, 80);
        ctx.lineTo(x, height - 70);
        ctx.stroke();
      }
      text(ctx, `${bits}-bit levels shown: ${gridCount}`, 78, 58, '#0f6f68', 16, '900');
      bar(ctx, 520, height - 42, 300, 18, Math.min(1, error / 24), '#a9432f', 'relative reconstruction error');
      setText('quantReadout', `relative error = ${error.toFixed(1)}% · levels = ${levels} · ${scaleType} · ${method}`);
    }
    ['quantBits', 'quantScale', 'quantOutlier', 'quantMethod'].forEach((id) => {
      $(id)?.addEventListener('input', draw);
      $(id)?.addEventListener('change', draw);
    });
    draw();
  }

  function initSpeculativeLab() {
    const ctx = ctxFor('specCanvas');
    if (!ctx) return;
    function draw() {
      const quality = value('specQuality') / 100;
      const K = value('specK');
      const cost = value('specCost') / 100;
      const temp = value('specTemp') / 100;
      setText('specQualityOut', String(value('specQuality')));
      setText('specKOut', String(K));
      setText('specCostOut', `${value('specCost')}%`);
      setText('specTempOut', temp.toFixed(2));
      const acceptProb = Math.max(0.05, Math.min(0.98, quality * (1.1 - temp * 0.55)));
      const expectedAccepted = Array.from({ length: K }, (_, i) => acceptProb ** (i + 1)).reduce((a, b) => a + b, 0);
      const speedup = Math.max(0.35, (expectedAccepted + acceptProb ** K) / (1 + K * cost));
      clearCanvas(ctx);
      const points = [];
      for (let i = 0; i <= K; i += 1) {
        const p = i === 0 ? 1 : acceptProb ** i;
        points.push({ x: 90 + i * (680 / Math.max(K, 1)), y: 300 - p * 210 });
      }
      line(ctx, points, '#0f6f68', 4);
      points.forEach((p, i) => {
        ctx.fillStyle = i <= expectedAccepted ? '#0f6f68' : '#a9432f';
        ctx.beginPath();
        ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
        ctx.fill();
      });
      text(ctx, 'acceptance prefix probability', 88, 58, '#171817', 16, '900');
      bar(ctx, 520, 92, 260, 22, Math.min(1, speedup / 4), speedup >= 1 ? '#0f6f68' : '#a9432f', `speedup ${speedup.toFixed(2)}x`);
      setText('specReadout', `教学估计 accepted tokens = ${expectedAccepted.toFixed(2)} / ${K} · target forwards per token lower when speedup > 1 · estimated speedup = ${speedup.toFixed(2)}x`);
    }
    ['specQuality', 'specK', 'specCost', 'specTemp'].forEach((id) => $(id)?.addEventListener('input', draw));
    draw();
  }

  function initDiffusionNFELab() {
    const ctx = ctxFor('nfeCanvas');
    if (!ctx) return;
    function draw() {
      const sampler = $('nfeSampler').value;
      const steps = value('nfeSteps');
      const guidance = value('nfeGuidance') / 10;
      const order = value('nfeOrder');
      setText('nfeStepsOut', String(steps));
      setText('nfeGuidanceOut', guidance.toFixed(1));
      setText('nfeOrderOut', String(order));
      const methodFactor = { ddpm: 1.25, ddim: 0.72, dpm: 0.42, unipc: 0.36 }[sampler];
      const error = methodFactor * (100 / steps) ** order * (1 + Math.max(0, guidance - 7) * 0.08);
      const latency = steps * 48;
      clearCanvas(ctx);
      const pts = [];
      for (let i = 0; i < steps; i += 1) {
        const t = i / Math.max(steps - 1, 1);
        pts.push({
          x: 82 + t * 720,
          y: 295 - 210 * t + Math.sin(t * Math.PI * 2) * 24 * methodFactor
        });
      }
      line(ctx, pts, sampler === 'ddpm' ? '#a9432f' : '#0f6f68', 3);
      text(ctx, `${sampler} trajectory · NFE ${steps}`, 82, 58, '#171817', 17, '900');
      bar(ctx, 520, 86, 260, 20, Math.min(1, error / 24), '#a9432f', `estimated error ${error.toFixed(2)}`);
      bar(ctx, 520, 136, 260, 20, Math.min(1, latency / 5000), '#314f78', `latency ${latency} ms`);
      setText('nfeReadout', `教学估计 latency = ${latency} ms · quality-risk score = ${error.toFixed(2)} · strong CFG can destabilize high-order jumps`);
    }
    ['nfeSampler', 'nfeSteps', 'nfeGuidance', 'nfeOrder'].forEach((id) => {
      $(id)?.addEventListener('input', draw);
      $(id)?.addEventListener('change', draw);
    });
    draw();
  }

  function initDiffusionCacheLab() {
    const ctx = ctxFor('cacheCanvas');
    if (!ctx) return;
    function draw() {
      const interval = value('cacheInterval');
      const threshold = value('cacheThreshold');
      const refresh = value('cacheRefresh');
      setText('cacheIntervalOut', String(interval));
      setText('cacheThresholdOut', String(threshold));
      setText('cacheRefreshOut', String(refresh));
      clearCanvas(ctx);
      const steps = 14;
      const blocks = 8;
      const cellW = 48;
      const cellH = 26;
      const startX = 110;
      const startY = 70;
      let compute = 0;
      let reuse = 0;
      for (let b = 0; b < blocks; b += 1) {
        text(ctx, `B${b}`, 70, startY + b * cellH + 18, '#5f625e', 12, '850');
        for (let s = 0; s < steps; s += 1) {
          const late = s > steps * (refresh / 100);
          const stable = threshold > 40 + b * 5;
          const isReuse = stable && !late && s % interval !== 0 && b > 1;
          const forced = late && b < 3;
          const color = forced ? '#a9432f' : isReuse ? '#0f6f68' : '#314f78';
          ctx.fillStyle = color;
          ctx.globalAlpha = isReuse ? 0.42 : 0.82;
          ctx.fillRect(startX + s * cellW, startY + b * cellH, cellW - 4, cellH - 4);
          ctx.globalAlpha = 1;
          if (isReuse) reuse += 1;
          else compute += 1;
        }
      }
      text(ctx, 'blue=compute · teal=reuse · red=forced refresh', 110, 325, '#171817', 15, '850');
      const speedup = (compute + reuse) / Math.max(compute, 1);
      const risk = Math.max(0, reuse / (compute + reuse) * 100 - refresh * 0.2);
      setText('cacheReadout', `教学估计 speedup = ${speedup.toFixed(2)}x · cache hit rate = ${(100 * reuse / (compute + reuse)).toFixed(0)}% · quality risk = ${risk.toFixed(0)}`);
    }
    ['cacheInterval', 'cacheThreshold', 'cacheRefresh'].forEach((id) => $(id)?.addEventListener('input', draw));
    draw();
  }

  function initTokenReductionLab() {
    const ctx = ctxFor('tokenCanvas');
    if (!ctx) return;
    function draw() {
      const merge = value('tokenMerge') / 100;
      const sharp = value('tokenSharp');
      const mode = $('tokenMode').value;
      setText('tokenMergeOut', `${value('tokenMerge')}%`);
      setText('tokenSharpOut', String(sharp));
      clearCanvas(ctx);
      const size = 12;
      const cell = 24;
      const startX = 130;
      const startY = 48;
      let active = 0;
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          const idx = y * size + x;
          const importance = (Math.sin(idx * 1.7) + 1) / 2 * 70 + (x + y) / (2 * size) * sharp;
          const affected = importance < merge * 130;
          if (!affected) active += 1;
          ctx.fillStyle = affected ? (mode === 'merge' ? 'rgba(15,111,104,0.28)' : 'rgba(169,67,47,0.20)') : 'rgba(49,79,120,0.78)';
          ctx.fillRect(startX + x * cell, startY + y * cell, cell - 3, cell - 3);
          if (affected && mode === 'merge' && x % 2 === 0) {
            ctx.strokeStyle = '#0f6f68';
            ctx.strokeRect(startX + x * cell, startY + y * cell, cell * 2 - 3, cell - 3);
          }
        }
      }
      const original = size * size;
      const tokenCount = mode === 'merge' ? Math.round(original * (1 - merge * 0.55)) : active;
      const reduction = 1 - (tokenCount * tokenCount) / (original * original);
      text(ctx, `${tokenCount} active tokens from ${original}`, 520, 115, '#171817', 20, '900');
      bar(ctx, 520, 170, 260, 24, reduction, '#0f6f68', 'attention cost reduction');
      bar(ctx, 520, 235, 260, 24, Math.min(1, merge * (mode === 'prune' ? 1.25 : 0.75)), '#a9432f', 'detail loss risk');
      setText('tokenReadout', `attention cost reduction = ${(reduction * 100).toFixed(0)}% · token count = ${tokenCount} · mode = ${mode}`);
    }
    ['tokenMerge', 'tokenSharp', 'tokenMode'].forEach((id) => {
      $(id)?.addEventListener('input', draw);
      $(id)?.addEventListener('change', draw);
    });
    draw();
  }

  function initTrainingMemoryLab() {
    const ids = ['trainParams', 'trainPrecision', 'trainGpus', 'trainSeq'];
    function draw() {
      const params = value('trainParams');
      const precision = value('trainPrecision');
      const gpus = value('trainGpus');
      const seq = value('trainSeq');
      const optimizer = $('trainOptimizer').value;
      const ckpt = $('trainCheckpoint').value;
      setText('trainParamsOut', String(params));
      setText('trainPrecisionOut', String(precision));
      setText('trainGpusOut', String(gpus));
      setText('trainSeqOut', String(seq));
      const paramGb = params * precision;
      const gradGb = params * precision;
      const optGb = optimizer === 'adam' ? params * 8 : params * precision;
      const actGb = (seq / 4096) * params * 0.18 * (ckpt === 'on' ? 0.42 : 1);
      const shard = Math.max(1, gpus);
      const perGpu = (paramGb + gradGb + optGb) / shard + actGb + params * 0.06;
      drawMetricBars('trainingMemoryBars', [
        { label: 'parameters shard', value: paramGb / shard, display: `${(paramGb / shard).toFixed(1)} GB` },
        { label: 'gradients shard', value: gradGb / shard, display: `${(gradGb / shard).toFixed(1)} GB` },
        { label: 'optimizer shard', value: optGb / shard, display: `${(optGb / shard).toFixed(1)} GB` },
        { label: 'activations', value: actGb, display: `${actGb.toFixed(1)} GB` }
      ]);
      setText('trainingReadout', `教学估计 per-GPU memory = ${perGpu.toFixed(1)} GB · optimizer = ${optimizer} · checkpoint = ${ckpt}`);
    }
    ids.forEach((id) => $(id)?.addEventListener('input', draw));
    ['trainOptimizer', 'trainCheckpoint'].forEach((id) => $(id)?.addEventListener('change', draw));
    draw();
  }

  function initMoELab() {
    const ctx = ctxFor('moeCanvas');
    if (!ctx) return;
    function draw() {
      const experts = value('moeExperts');
      const topk = value('moeTopk');
      const batch = value('moeBatch');
      const imbalance = value('moeImbalance') / 100;
      const capacity = value('moeCapacity') / 100;
      setText('moeExpertsOut', String(experts));
      setText('moeTopkOut', String(topk));
      setText('moeBatchOut', String(batch));
      setText('moeImbalanceOut', String(value('moeImbalance')));
      setText('moeCapacityOut', capacity.toFixed(2));
      clearCanvas(ctx);
      const cols = Math.min(experts, 16);
      const rows = Math.ceil(experts / cols);
      const cellW = 42;
      const cellH = 62;
      const startX = 95;
      const startY = 70;
      let dropped = 0;
      let maxUtil = 0;
      for (let e = 0; e < experts; e += 1) {
        const row = Math.floor(e / cols);
        const col = e % cols;
        const skew = 1 + imbalance * Math.sin(e * 1.9) * 1.8 + (e === 0 ? imbalance * 2.2 : 0);
        const load = Math.max(0.1, (batch * topk / experts) * skew);
        const cap = (batch * topk / experts) * capacity;
        const util = Math.min(1.4, load / cap);
        maxUtil = Math.max(maxUtil, util);
        dropped += Math.max(0, load - cap);
        ctx.fillStyle = util > 1 ? '#a9432f' : util > 0.75 ? '#9a6d19' : '#0f6f68';
        ctx.globalAlpha = Math.min(1, 0.25 + util * 0.65);
        ctx.fillRect(startX + col * cellW, startY + row * cellH, cellW - 6, cellH - 10);
        ctx.globalAlpha = 1;
        text(ctx, String(e), startX + col * cellW + 9, startY + row * cellH + 32, '#171817', 12, '900');
      }
      const droppedPct = dropped / Math.max(batch * topk, 1) * 100;
      text(ctx, 'expert utilization heatmap', 95, 48, '#171817', 17, '900');
      bar(ctx, 95, 292, 310, 22, Math.min(1, maxUtil / 1.4), '#314f78', 'peak expert utilization');
      bar(ctx, 510, 292, 270, 22, Math.min(1, droppedPct / 25), '#a9432f', 'capacity overflow');
      setText('moeReadout', `dropped tokens = ${droppedPct.toFixed(1)}% · active experts per token = ${topk} · all-to-all cost rises with experts and imbalance`);
    }
    ['moeExperts', 'moeTopk', 'moeBatch', 'moeImbalance', 'moeCapacity'].forEach((id) => $(id)?.addEventListener('input', draw));
    draw();
  }

  function bindToc() {
    const toggle = $('tocToggle');
    const toc = $('lectureToc');
    if (toggle && toc) {
      toggle.addEventListener('click', () => {
        const open = toc.classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
      });
    }
    const links = all('.lecture-toc a');
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
    const barEl = $('readingProgressBar');
    if (!barEl) return;
    function update() {
      const max = root.documentElement.scrollHeight - window.innerHeight;
      const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
      barEl.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    }
    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
  }

  function init() {
    bindAlgorithmPanels();
    bindBottleneckMap();
    initRooflineLab();
    initAttentionMemoryLab();
    initKVCacheLab();
    initBatchingLab();
    initQuantizationLab();
    initSpeculativeLab();
    initDiffusionNFELab();
    initDiffusionCacheLab();
    initTokenReductionLab();
    initTrainingMemoryLab();
    initMoELab();
    bindToc();
    bindProgress();
  }

  if (root.readyState === 'loading') {
    root.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
