(function () {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const wanModels = [
    {
      name: "Wan2.1-T2V-1.3B",
      category: "Text-to-Video",
      tags: ["t2v"],
      inputs: ["text prompt"],
      outputs: ["video"],
      keyIdea: "轻量文本生成视频模型，适合本地实验、LoRA 和低显存部署。",
      notes: ["Wan2.1 基础路线：Wan-VAE + Video DiT + Flow Matching。", "文本条件通常通过 T5 类文本编码器和 cross-attention 注入。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.1-T2V-14B",
      category: "Text-to-Video",
      tags: ["t2v"],
      inputs: ["text prompt"],
      outputs: ["video"],
      keyIdea: "高质量文本生成视频主模型。",
      notes: ["适合作为 Wan2.1 高质量 T2V 主线理解。", "质量优先时通常比轻量模型更重。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.1-I2V-14B-480P / 720P",
      category: "Image-to-Video",
      tags: ["t2v"],
      inputs: ["image", "text prompt"],
      outputs: ["video"],
      keyIdea: "给定首帧或参考图生成动态视频。",
      notes: ["图像条件可通过 image latent、mask 或端点帧约束注入。", "480P / 720P 表示不同分辨率路线。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.1-FLF2V-14B",
      category: "First-Last-Frame-to-Video",
      tags: ["t2v", "control"],
      inputs: ["first frame", "last frame", "text prompt"],
      outputs: ["transition video"],
      keyIdea: "给定首帧和尾帧，生成中间过渡视频。",
      notes: ["端点帧提供更强的时序约束。", "适合把 I2V 扩展到可控过渡。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.1-VACE-1.3B / 14B",
      category: "Video Creation & Editing",
      tags: ["edit", "control"],
      inputs: ["reference", "editing video", "mask", "text prompt"],
      outputs: ["generated or edited video"],
      keyIdea: "统一视频生成和编辑模型。",
      notes: ["核心抽象是 VCU, Video Condition Unit。", "Context Adapter 将 reference、editing video、mask 等条件注入视频 DiT。"],
      sourceKind: "paper"
    },
    {
      name: "Wan2.2-T2V-A14B",
      category: "Text-to-Video",
      tags: ["t2v"],
      inputs: ["text prompt"],
      outputs: ["video"],
      keyIdea: "timestep-level MoE：高噪阶段负责结构，低噪阶段负责细节。",
      notes: ["A14B 是 Wan2.2 高质量路线。", "总参数量大于单个专家，但每一步只激活一个专家。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.2-I2V-A14B",
      category: "Image-to-Video",
      tags: ["t2v"],
      inputs: ["image", "text prompt"],
      outputs: ["video"],
      keyIdea: "高质量 I2V，使用 timestep-level MoE。",
      notes: ["图像 + 文本共同约束生成轨迹。", "高噪专家偏整体构图和运动，低噪专家偏细节清晰度。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.2-TI2V-5B",
      category: "Unified T2V / I2V",
      tags: ["t2v"],
      inputs: ["text prompt", "optional image"],
      outputs: ["video"],
      keyIdea: "统一 T2V/I2V，高压缩 VAE，部署友好。",
      notes: ["这是 Wan2.2 的高压缩统一路线。", "更偏工程可部署，而不是单纯最大模型质量。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.2-S2V-14B",
      category: "Audio-driven Character Video",
      tags: ["character"],
      inputs: ["image", "audio", "text prompt"],
      outputs: ["character video"],
      keyIdea: "音频驱动电影级角色视频。",
      notes: ["音频是输入条件，不是生成目标本身。", "适合唱歌、对白、配音驱动和角色表演。"],
      sourceKind: "model-card"
    },
    {
      name: "Wan2.2-Animate-14B",
      category: "Character Animation / Replacement",
      tags: ["character", "edit"],
      inputs: ["character image", "reference video"],
      outputs: ["animated or replaced video"],
      keyIdea: "身份保持、动作迁移、表情迁移、Relighting LoRA 环境融合。",
      notes: ["Animation mode 让目标角色模仿参考视频动作。", "Replacement mode 将参考视频人物替换为目标角色。"],
      sourceKind: "paper"
    },
    {
      name: "Wan-Fun / Fun-Control / Fun-Camera",
      category: "Control Branch",
      tags: ["control"],
      inputs: ["Canny", "Depth", "Pose", "MLSD", "trajectory", "camera"],
      outputs: ["controlled video"],
      keyIdea: "结构控制、相机控制和条件生成分支。",
      notes: ["更接近 ControlNet / T2V-Control 风格的专门控制模型。", "应与 VACE、Wan-Animate、Wan-S2V 区分任务边界。"],
      sourceKind: "community"
    }
  ];

  const ltxModels = [
    {
      name: "LTX-Video / LTXV",
      category: "Video Foundation Model",
      tags: ["base"],
      inputs: ["text prompt", "image", "keyframes", "video context"],
      outputs: ["video"],
      keyIdea: "高压缩 latent + DiT + denoising decoder，目标是快速生成。",
      notes: ["支持 T2V、I2V、多关键帧、video extension 和 V2V。", "关键不是单纯堆大模型，而是压缩 token 数。"],
      sourceKind: "paper"
    },
    {
      name: "LTXV 2B",
      category: "Lightweight Base",
      tags: ["base", "efficiency"],
      inputs: ["text prompt", "image"],
      outputs: ["video"],
      keyIdea: "轻量、低显存、快速实验。",
      notes: ["适合作为快速迭代和本地实验入口。", "质量和能力需要结合具体模型卡判断。"],
      sourceKind: "model-card"
    },
    {
      name: "LTXV 2B distilled",
      category: "Distilled Variant",
      tags: ["efficiency"],
      inputs: ["text prompt", "image"],
      outputs: ["video"],
      keyIdea: "更快的蒸馏版本。",
      notes: ["通过减少有效采样步数提升迭代速度。", "蒸馏质量取决于 teacher、数据和目标设置。"],
      sourceKind: "model-card"
    },
    {
      name: "LTXV 13B",
      category: "High-quality Base",
      tags: ["base"],
      inputs: ["text prompt", "image", "keyframes"],
      outputs: ["video"],
      keyIdea: "高质量视频生成基座。",
      notes: ["与 2B 档位构成质量和成本分层。", "适合理解 LTXV 的高质量主线。"],
      sourceKind: "model-card"
    },
    {
      name: "LTXV 13B distilled",
      category: "High-quality Fast Variant",
      tags: ["efficiency"],
      inputs: ["text prompt", "image", "keyframes"],
      outputs: ["video"],
      keyIdea: "少步采样、快速迭代。",
      notes: ["面向更快的生产流试错。", "需要与非蒸馏版本比较质量损失。"],
      sourceKind: "model-card"
    },
    {
      name: "FP8 variants",
      category: "Quantized Deployment",
      tags: ["efficiency"],
      inputs: ["model weights"],
      outputs: ["lower-memory inference"],
      keyIdea: "量化部署、降低显存。",
      notes: ["降低部署成本，但要检查具体算子、硬件和质量退化。", "不要把 FP8 等同于无损压缩。"],
      sourceKind: "model-card"
    },
    {
      name: "Spatial / Temporal upscaler / Detailer",
      category: "Multiscale Components",
      tags: ["efficiency"],
      inputs: ["base video", "latent video", "tiles"],
      outputs: ["higher resolution or higher FPS video"],
      keyIdea: "用多尺度 pipeline 补回空间、时间和局部细节。",
      notes: ["低分辨率阶段负责构图和运动。", "upscaler / detailer 负责分辨率、帧率、纹理和边缘。"],
      sourceKind: "model-card"
    },
    {
      name: "LTX IC-LoRA",
      category: "In-Context Control",
      tags: ["control"],
      inputs: ["depth", "pose", "canny", "edge", "video context", "audio/video reference"],
      outputs: ["controlled generation or editing"],
      keyIdea: "轻量 LoRA 接入结构、参考、上下文等控制信号。",
      notes: ["包括 Depth Control、Pose Control、Canny / Edge Control、Union IC-LoRA。", "ComfyUI 和 trainer 生态支持训练与使用 IC-LoRA。"],
      sourceKind: "model-card"
    },
    {
      name: "Detailer LoRA / Creative Lab LoRA",
      category: "Task LoRA",
      tags: ["control", "efficiency"],
      inputs: ["task reference", "video context"],
      outputs: ["enhanced or edited video"],
      keyIdea: "面向具体 VFX、修复、风格/场景转换和编辑任务。",
      notes: ["示例包括 Day-To-Night、Colorization、Decompression、Deblur、Inpainting / Outpainting。", "也包括 Water Simulation、Ingredients / Reference-based video、Instant-Shave 等方向。"],
      sourceKind: "collection"
    },
    {
      name: "LTX-2 / LTX-2.3",
      category: "Audio-Video Foundation Model",
      tags: ["audio", "base"],
      inputs: ["text prompt", "optional multimodal conditions"],
      outputs: ["synchronized video", "audio"],
      keyIdea: "非对称双流 DiT + 双向 audio-video cross-attention。",
      notes: ["区别于 Wan-S2V：它生成音频和视频，而不是只用音频驱动视频。", "modality-aware CFG 分别控制文本遵循和跨模态一致性。"],
      sourceKind: "model-card"
    },
    {
      name: "LTX-Video-Trainer / LTX-2 Trainer / ComfyUI-LTXVideo",
      category: "Production Workflow",
      tags: ["efficiency", "control"],
      inputs: ["datasets", "workflows", "LoRA configs"],
      outputs: ["trained adapters", "reproducible workflows"],
      keyIdea: "训练、ComfyUI、Diffusers 风格工作流构成生产栈。",
      notes: ["体现 LTX 从生成模型走向后期处理和生产工作流。", "具体能力以对应仓库和模型卡为准。"],
      sourceKind: "official-repo"
    }
  ];

  const comparisonRows = [
    ["基础定位", "高质量视频生成、角色动画、视频编辑、控制扩展", "高效率生成、蒸馏、多尺度、音视频联合、生产工作流"],
    ["基础模型", "Wan2.1 / Wan2.2", "LTX-Video / LTXV / LTX-2 / LTX-2.3"],
    ["生成任务", "T2V、I2V、FLF2V、TI2V", "T2V、I2V、多关键帧、video extension、V2V"],
    ["关键结构", "Wan-VAE、Video DiT、Flow Matching、timestep MoE", "高压缩 VAE、DiT、denoising decoder、multiscale pipeline"],
    ["MoE / 专家", "Wan2.2 使用 high-noise / low-noise experts", "主要不是 Wan 式 timestep MoE，LTX-2 是音频/视频双流"],
    ["编辑模型", "VACE、Wan-Animate、Wan-Fun 等", "VACE-LTX、IC-LoRA、Creative Lab、detailer"],
    ["角色能力", "Wan-Animate、Wan-S2V 很强", "主要依赖 I2V、IC-LoRA、trainer 与任务 LoRA"],
    ["音频能力", "Wan-S2V 是音频驱动视频", "LTX-2 / LTX-2.3 是联合音视频生成"],
    ["控制路线", "VACE、Wan-Fun、Control-Camera、社区控制模型", "IC-LoRA、Union IC-LoRA、Creative Lab LoRA"],
    ["部署策略", "大模型质量优先，TI2V-5B/FP8/GGUF/加速社区降低成本", "distilled、FP8、2B/13B 档位、upscaler、多尺度 pipeline"],
    ["工程生态", "DiffSynth、ComfyUI-WanVideoWrapper、LightX2V、TeaCache", "LTX-Video-Trainer、LTX-2 Trainer、ComfyUI-LTXVideo、Diffusers"]
  ];

  const glossary = [
    ["Video VAE", "将视频压缩到 latent space，并从 latent 解码回视频的自编码器。"],
    ["3D causal VAE", "在时间维保持因果结构的视频 VAE，适合长视频和流式/分段生成。"],
    ["Latent token", "视频被 VAE 压缩后的时空 token，是 DiT 主要处理对象。"],
    ["DiT", "Diffusion Transformer，用 Transformer 替代传统 U-Net 进行扩散去噪。"],
    ["Flow Matching", "学习从噪声分布到数据分布的连续速度场，常用于少步生成。"],
    ["Rectified Flow", "Flow Matching 相关训练/采样范式，常见于现代 DiT 生成模型。"],
    ["Cross-attention", "用文本、图像、音频或控制信号影响视频 latent 的注意力机制。"],
    ["CFG", "Classifier-Free Guidance，推理时增强条件遵循度。"],
    ["Modality-aware CFG", "分别调节文本条件和跨模态条件强度的 guidance 方法。"],
    ["MoE", "Mixture of Experts，专家混合结构。Wan2.2 中按去噪阶段切换专家。"],
    ["High-noise expert", "Wan2.2 中负责高噪早期阶段的专家，偏全局结构和运动。"],
    ["Low-noise expert", "Wan2.2 中负责低噪后期阶段的专家，偏纹理和细节。"],
    ["VACE", "统一视频生成与编辑框架，使用 Video Condition Unit 和 Context Adapter。"],
    ["VCU", "Video Condition Unit，用于统一 reference、mask、editing video 等条件。"],
    ["IC-LoRA", "In-Context LoRA，用轻量 LoRA 接入结构、参考、上下文等控制信号。"],
    ["Spatial upscaler", "提升空间分辨率的 latent/video 放大模型。"],
    ["Temporal upscaler", "提升时间分辨率或帧率的模型。"],
    ["Audio-driven video", "音频作为输入条件驱动视频生成，如 Wan-S2V。"],
    ["Joint audio-video generation", "同时生成视频和音频，如 LTX-2 / LTX-2.3。"]
  ];

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function renderModelCard(model) {
    const notes = model.notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
    const inputs = model.inputs.map((item) => `<span>${escapeHtml(item)}</span>`).join("");
    const outputs = model.outputs.map((item) => `<span>${escapeHtml(item)}</span>`).join("");
    return `
      <article class="model-card">
        <div>
          <strong>${escapeHtml(model.name)}</strong>
          <p>${escapeHtml(model.category)}</p>
        </div>
        <div class="model-meta" aria-label="${escapeHtml(model.name)} inputs">${inputs}</div>
        <div class="model-meta" aria-label="${escapeHtml(model.name)} outputs">${outputs}</div>
        <p>${escapeHtml(model.keyIdea)}</p>
        <ul>${notes}</ul>
        <span class="source-kind">${escapeHtml(model.sourceKind)}</span>
      </article>
    `;
  }

  function renderModels(container, models, filter) {
    if (!container) return;
    const visible = filter === "all" ? models : models.filter((model) => model.tags.includes(filter));
    container.innerHTML = visible.map(renderModelCard).join("");
  }

  function setupModelFilters() {
    const wanContainer = $("#wanModelGrid");
    const ltxContainer = $("#ltxModelGrid");
    renderModels(wanContainer, wanModels, "all");
    renderModels(ltxContainer, ltxModels, "all");

    $$("[data-model-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        $$("[data-model-filter]").forEach((item) => {
          item.classList.toggle("active", item === button);
          item.setAttribute("aria-pressed", String(item === button));
        });
        renderModels(wanContainer, wanModels, button.dataset.modelFilter);
      });
    });

    $$("[data-ltx-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        $$("[data-ltx-filter]").forEach((item) => {
          item.classList.toggle("active", item === button);
          item.setAttribute("aria-pressed", String(item === button));
        });
        renderModels(ltxContainer, ltxModels, button.dataset.ltxFilter);
      });
    });
  }

  function renderComparison() {
    const body = $("#comparisonBody");
    if (body) {
      body.innerHTML = comparisonRows.map(([dimension, wan, ltx]) => `
        <tr>
          <td>${escapeHtml(dimension)}</td>
          <td>${escapeHtml(wan)}</td>
          <td>${escapeHtml(ltx)}</td>
        </tr>
      `).join("");
    }

    const cards = $("#comparisonCards");
    if (cards) {
      cards.innerHTML = comparisonRows.map(([dimension, wan, ltx]) => `
        <article>
          <h3>${escapeHtml(dimension)}</h3>
          <dl>
            <div><dt>Wan 生态</dt><dd>${escapeHtml(wan)}</dd></div>
            <div><dt>LTX 生态</dt><dd>${escapeHtml(ltx)}</dd></div>
          </dl>
        </article>
      `).join("");
    }
  }

  function renderGlossary() {
    const grid = $("#glossaryGrid");
    if (!grid) return;
    grid.innerHTML = glossary.map(([term, explanation]) => `
      <article>
        <strong>${escapeHtml(term)}</strong>
        <p>${escapeHtml(explanation)}</p>
      </article>
    `).join("");
  }

  function formatLarge(value) {
    if (!Number.isFinite(value) || value <= 0) return "0";
    if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
    return String(Math.round(value));
  }

  function percentLog(value, maxValue) {
    const safeValue = Math.max(1, value);
    const safeMax = Math.max(1, maxValue);
    return Math.max(8, Math.min(100, (Math.log10(safeValue) / Math.log10(safeMax)) * 100));
  }

  function setupTokenLab() {
    const controls = {
      frames: $("#frameCount"),
      width: $("#videoWidth"),
      height: $("#videoHeight"),
      strideT: $("#strideT"),
      strideS: $("#strideS"),
      steps: $("#sampleSteps")
    };

    if (Object.values(controls).some((control) => !control)) return;

    const outputs = {
      frames: $("#frameCountOut"),
      width: $("#videoWidthOut"),
      height: $("#videoHeightOut"),
      strideT: $("#strideTOut"),
      strideS: $("#strideSOut"),
      steps: $("#sampleStepsOut")
    };

    const bars = {
      pixel: $("#pixelTokenBar"),
      latent: $("#latentTokenBar"),
      attention: $("#attentionCostBar"),
      sampler: $("#samplerWorkBar")
    };

    const texts = {
      pixel: $("#pixelTokenText"),
      latent: $("#latentTokenText"),
      attention: $("#attentionCostText"),
      sampler: $("#samplerWorkText")
    };

    const readout = $("#tokenCostReadout");
    const explanation = $("#tokenCostExplanation");

    function update() {
      const frames = Math.max(1, Number(controls.frames.value));
      const width = Math.max(1, Number(controls.width.value));
      const height = Math.max(1, Number(controls.height.value));
      const strideT = Math.max(1, Number(controls.strideT.value));
      const strideS = Math.max(1, Number(controls.strideS.value));
      const steps = Math.max(1, Number(controls.steps.value));

      outputs.frames.textContent = String(frames);
      outputs.width.textContent = String(width);
      outputs.height.textContent = String(height);
      outputs.strideT.textContent = String(strideT);
      outputs.strideS.textContent = String(strideS);
      outputs.steps.textContent = String(steps);

      const pixelTokens = frames * width * height;
      const latentTokens = Math.ceil(frames / strideT) * Math.ceil(width / strideS) * Math.ceil(height / strideS);
      const attentionCost = latentTokens * latentTokens;
      const samplerWork = attentionCost * steps;
      const maxValue = Math.max(pixelTokens, latentTokens, attentionCost, samplerWork);

      bars.pixel.style.setProperty("--value", `${percentLog(pixelTokens, maxValue)}%`);
      bars.latent.style.setProperty("--value", `${percentLog(latentTokens, maxValue)}%`);
      bars.attention.style.setProperty("--value", `${percentLog(attentionCost, maxValue)}%`);
      bars.sampler.style.setProperty("--value", `${percentLog(samplerWork, maxValue)}%`);

      texts.pixel.textContent = formatLarge(pixelTokens);
      texts.latent.textContent = formatLarge(latentTokens);
      texts.attention.textContent = formatLarge(attentionCost);
      texts.sampler.textContent = formatLarge(samplerWork);

      const compression = pixelTokens / Math.max(1, latentTokens);
      readout.textContent = `N = ${formatLarge(latentTokens)}, N² ≈ ${formatLarge(attentionCost)}, pixel/latent ≈ ${compression.toFixed(1)}×`;
      explanation.textContent = compression > 900
        ? "压缩率很高时推理更快，但 decoder、upscaler 和 detailer 要承担更多细节补偿。"
        : "压缩率较低时细节更容易保留，但 attention 和采样成本会迅速抬升。";
    }

    Object.values(controls).forEach((control) => control.addEventListener("input", update));
    update();
  }

  function setupMoeLab() {
    const buttons = $$("[data-moe-stage]");
    const fill = $("#moeNoiseFill");
    const pin = $("#moeStagePin");
    const high = $("#moeHighCard");
    const low = $("#moeLowCard");
    const readout = $("#moeReadout");
    if (!buttons.length || !fill || !pin || !high || !low || !readout) return;

    function activate(stage) {
      const isHigh = stage === "high";
      buttons.forEach((button) => {
        const active = button.dataset.moeStage === stage;
        button.classList.toggle("active", active);
        button.setAttribute("aria-pressed", String(active));
      });
      high.classList.toggle("active", isHigh);
      low.classList.toggle("active", !isHigh);
      fill.style.width = isHigh ? "72%" : "30%";
      pin.style.left = isHigh ? "33%" : "78%";
      readout.textContent = isHigh
        ? "High-noise expert 偏整体构图、主体关系、镜头运动和低频结构。"
        : "Low-noise expert 偏纹理、边缘、细节、清晰度和后期修复。";
    }

    buttons.forEach((button) => {
      button.addEventListener("click", () => activate(button.dataset.moeStage));
      button.addEventListener("mouseenter", () => activate(button.dataset.moeStage));
    });
    activate("high");
  }

  function setupProgressAndToc() {
    const progress = $("#readingProgressBar");
    const toc = $("#lectureToc");
    const toggle = $("#tocToggle");
    const links = $$(".lecture-toc a");
    const sections = links
      .map((link) => $(link.getAttribute("href")))
      .filter(Boolean);

    function updateProgress() {
      if (!progress) return;
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - window.innerHeight;
      const pct = height > 0 ? Math.min(100, Math.max(0, (scrollTop / height) * 100)) : 0;
      progress.style.width = `${pct}%`;
    }

    if (toggle && toc) {
      toggle.addEventListener("click", () => {
        const open = !toc.classList.contains("open");
        toc.classList.toggle("open", open);
        toggle.setAttribute("aria-expanded", String(open));
      });
      links.forEach((link) => {
        link.addEventListener("click", () => {
          toc.classList.remove("open");
          toggle.setAttribute("aria-expanded", "false");
        });
      });
    }

    if ("IntersectionObserver" in window && sections.length) {
      const observer = new IntersectionObserver((entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const id = `#${visible.target.id}`;
        links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === id));
      }, { rootMargin: "-22% 0px -62% 0px", threshold: [0.05, 0.2, 0.5] });
      sections.forEach((section) => observer.observe(section));
    }

    updateProgress();
    window.addEventListener("scroll", updateProgress, { passive: true });
    window.addEventListener("resize", updateProgress);
  }

  function init() {
    setupProgressAndToc();
    setupModelFilters();
    renderComparison();
    renderGlossary();
    setupTokenLab();
    setupMoeLab();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
