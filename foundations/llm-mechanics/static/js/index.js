"use strict";

(function (global) {
  const PAIR_SEPARATOR = "\u0001";

  function bpeMergeStep(tokens) {
    if (!Array.isArray(tokens) || tokens.length < 2) return null;
    const counts = new Map();
    for (let index = 0; index < tokens.length - 1; index += 1) {
      const key = `${tokens[index]}${PAIR_SEPARATOR}${tokens[index + 1]}`;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    let bestKey = null;
    let bestCount = -1;
    counts.forEach((count, key) => {
      if (count > bestCount) {
        bestKey = key;
        bestCount = count;
      }
    });
    if (!bestKey) return null;
    const pair = bestKey.split(PAIR_SEPARATOR);
    const mergedToken = pair.join("");
    const merged = [];
    for (let index = 0; index < tokens.length; index += 1) {
      if (index + 1 < tokens.length && tokens[index] === pair[0] && tokens[index + 1] === pair[1]) {
        merged.push(mergedToken);
        index += 1;
      } else {
        merged.push(tokens[index]);
      }
    }
    return { tokens: merged, pair, count: bestCount, mergedToken };
  }

  function causalVisible(queryPosition, keyPosition) {
    return keyPosition <= queryPosition;
  }

  function rotate2D(vector, angle) {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    return [vector[0] * cosine - vector[1] * sine, vector[0] * sine + vector[1] * cosine];
  }

  function rmsNormalize(values, epsilon = 1e-6) {
    if (!values.length) return [];
    const meanSquare = values.reduce((sum, value) => sum + value * value, 0) / values.length;
    const denominator = Math.sqrt(meanSquare + epsilon);
    return values.map((value) => value / denominator);
  }

  function silu(value) {
    return value / (1 + Math.exp(-value));
  }

  function prefillDecodeWork(promptLength, generatedLength) {
    const prompt = Math.max(1, Math.round(promptLength));
    const generated = Math.max(1, Math.round(generatedLength));
    let naive = 0;
    for (let step = 0; step < generated; step += 1) naive += prompt + step;
    return { naive, cached: prompt + generated - 1 };
  }

  function kvCacheMetrics({ layers, kvHeads, headDim, dtypeBytes, batch, context }) {
    const values = [layers, kvHeads, headDim, dtypeBytes, batch, context].map(Number);
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new RangeError("KV cache parameters must be positive finite numbers");
    }
    const bytesPerToken = values[0] * 2 * values[1] * values[2] * values[3];
    return { bytesPerToken, totalBytes: bytesPerToken * values[4] * values[5] };
  }

  function softmax(logits, temperature = 1) {
    if (!Number.isFinite(temperature) || temperature <= 0) {
      throw new RangeError("temperature must be positive for softmax");
    }
    const scaled = logits.map((value) => value / temperature);
    const maximum = Math.max(...scaled);
    const exponents = scaled.map((value) => Math.exp(value - maximum));
    const total = exponents.reduce((sum, value) => sum + value, 0);
    return exponents.map((value) => value / total);
  }

  function topPIndices(probabilities, threshold) {
    const p = Math.min(1, Math.max(Number.EPSILON, Number(threshold)));
    const ranked = probabilities.map((probability, index) => ({ probability, index }))
      .sort((left, right) => right.probability - left.probability || left.index - right.index);
    const kept = [];
    let cumulative = 0;
    for (const item of ranked) {
      kept.push(item.index);
      cumulative += item.probability;
      if (cumulative >= p) break;
    }
    return kept.length ? kept : [ranked[0].index];
  }

  function applyRepetitionPenalty(logits, seenTokenIds, penalty) {
    const factor = Number(penalty);
    if (!Number.isFinite(factor) || factor < 1) throw new RangeError("repetition penalty must be at least 1");
    return logits.map((value, index) => {
      if (!seenTokenIds.has(index)) return value;
      return value >= 0 ? value / factor : value * factor;
    });
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) return "invalid";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return `${value.toFixed(unit === 0 ? 0 : 2)} ${units[unit]}`;
  }

  function entropy(probabilities) {
    return -probabilities.reduce((sum, probability) => probability > 0 ? sum + probability * Math.log(probability) : sum, 0);
  }

  function seededRandom(seed) {
    let state = seed >>> 0;
    return function random() {
      state += 0x6D2B79F5;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  function categoricalSample(probabilities, allowedIndices, seed) {
    const allowed = new Set(allowedIndices);
    const total = probabilities.reduce((sum, probability, index) => allowed.has(index) ? sum + probability : sum, 0);
    let draw = seededRandom(seed)() * total;
    for (let index = 0; index < probabilities.length; index += 1) {
      if (!allowed.has(index)) continue;
      draw -= probabilities[index];
      if (draw <= 0) return index;
    }
    return allowedIndices[allowedIndices.length - 1];
  }

  const api = {
    applyRepetitionPenalty,
    bpeMergeStep,
    causalVisible,
    categoricalSample,
    entropy,
    formatBytes,
    kvCacheMetrics,
    prefillDecodeWork,
    rmsNormalize,
    rotate2D,
    silu,
    softmax,
    topPIndices
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.LLMMechanics = api;

  if (typeof document === "undefined") return;

  const byId = (id) => document.getElementById(id);
  const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));

  function safeInit(name, initialize) {
    try {
      initialize();
    } catch (error) {
      console.error(`[llm-mechanics:${name}]`, error);
      const root = document.querySelector(`[data-lab="${name}"]`);
      const message = root && root.querySelector(".lab-error");
      if (message) message.hidden = false;
    }
  }

  function initializeNavigation() {
    const progressBar = byId("readingProgressBar");
    const progressText = byId("tocProgressText");
    const toc = byId("lectureToc");
    const toggle = byId("tocToggle");
    const links = Array.from(toc.querySelectorAll('a[href^="#"]'));
    const sections = links.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);

    function updateProgress() {
      const scrollable = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
      const progress = clamp(window.scrollY / scrollable, 0, 1);
      progressBar.style.width = `${(progress * 100).toFixed(2)}%`;
      progressText.textContent = `${Math.round(progress * 100)}%`;
    }

    toggle.addEventListener("click", () => {
      const open = !toc.classList.contains("open");
      toc.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", String(open));
    });
    links.forEach((link) => link.addEventListener("click", () => {
      toc.classList.remove("open");
      toggle.setAttribute("aria-expanded", "false");
    }));

    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        if (!visible) return;
        links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
      }, { rootMargin: "-18% 0px -65% 0px", threshold: [0, 0.1, 0.5] });
      sections.forEach((section) => observer.observe(section));
    }

    updateProgress();
    window.addEventListener("scroll", updateProgress, { passive: true });
  }

  function initializeCopyButtons() {
    async function copyText(text) {
      if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      const copied = document.execCommand("copy");
      textarea.remove();
      if (!copied) throw new Error("copy command failed");
    }

    document.querySelectorAll("[data-copy-target]").forEach((button) => {
      button.addEventListener("click", async () => {
        const target = byId(button.dataset.copyTarget);
        if (!target) return;
        const original = button.textContent;
        try {
          await copyText(target.textContent);
          button.textContent = "已复制";
          button.classList.add("copied");
        } catch (error) {
          console.error("[llm-mechanics:copy]", error);
          button.textContent = "请手动选择";
        }
        window.setTimeout(() => {
          button.textContent = original;
          button.classList.remove("copied");
        }, 1400);
      });
    });
  }

  function initializeBPELab() {
    const input = byId("bpeInput");
    const tokenBoard = byId("bpeTokens");
    const rulesList = byId("bpeRules");
    const stepButton = byId("bpeStep");
    const autoButton = byId("bpeAuto");
    const resetButton = byId("bpeReset");
    let state;
    let timer = null;

    function stopAuto() {
      if (timer) window.clearInterval(timer);
      timer = null;
      autoButton.textContent = "自动执行";
    }

    function render() {
      tokenBoard.replaceChildren(...state.tokens.map((token) => {
        const span = document.createElement("span");
        span.textContent = token === " " ? "␠" : token;
        if (token === state.latest) span.classList.add("new");
        return span;
      }));
      rulesList.replaceChildren(...state.rules.map((rule, index) => {
        const item = document.createElement("li");
        item.textContent = `${index + 1}. ${rule.pair.join(" + ")} → ${rule.mergedToken} (${rule.count}×)`;
        return item;
      }));
      byId("bpeStepCount").textContent = String(state.rules.length);
      byId("bpePair").textContent = state.rules.length ? state.rules.at(-1).pair.join("+") : "—";
      byId("bpeVocab").textContent = String(state.rules.length);
      byId("bpeCompression").textContent = `${(state.initialLength / Math.max(1, state.tokens.length)).toFixed(2)}×`;
      byId("bpeReadout").textContent = state.rules.length
        ? `第 ${state.rules.length} 步合并 ${state.rules.at(-1).pair.join(" + ")}；序列 ${state.initialLength} → ${state.tokens.length} tokens。`
        : "尚未执行 merge。先预测哪个 pair 最常见，再点击验证。";
      stepButton.disabled = state.tokens.length < 2 || state.rules.length >= 12;
      if (stepButton.disabled) stopAuto();
    }

    function reset() {
      stopAuto();
      const tokens = Array.from(input.value || "");
      state = { tokens, initialLength: tokens.length, rules: [], latest: "" };
      render();
    }

    function step() {
      const result = bpeMergeStep(state.tokens);
      if (!result) {
        stepButton.disabled = true;
        stopAuto();
        return;
      }
      state.tokens = result.tokens;
      state.rules.push(result);
      state.latest = result.mergedToken;
      render();
    }

    stepButton.addEventListener("click", step);
    autoButton.addEventListener("click", () => {
      if (timer) return stopAuto();
      autoButton.textContent = "暂停";
      timer = window.setInterval(step, 520);
    });
    resetButton.addEventListener("click", reset);
    input.addEventListener("change", reset);
    reset();
  }

  function initializeDecoderLab() {
    const tabs = Array.from(document.querySelectorAll("[data-decoder-mode]"));
    const panels = Array.from(document.querySelectorAll("[data-mode-panel]"));
    const position = byId("decoderPosition");
    const ropePosition = byId("ropePosition");
    const rmsScale = byId("rmsScale");
    const swigluInput = byId("swigluInput");
    const readout = byId("decoderReadout");
    let activeMode = "causal";

    function renderCausal() {
      const query = Number(position.value);
      byId("decoderPositionOut").textContent = String(query);
      const cells = [];
      for (let row = 0; row < 6; row += 1) {
        for (let key = 0; key < 6; key += 1) {
          const cell = document.createElement("span");
          const visible = causalVisible(row, key);
          cell.textContent = visible ? "1" : "0";
          cell.classList.toggle("visible", visible);
          cell.classList.toggle("focus", row === query);
          cell.title = `query ${row}, key ${key}: ${visible ? "visible" : "masked"}`;
          cells.push(cell);
        }
      }
      byId("causalMatrix").replaceChildren(...cells);
      readout.textContent = `query=${query} 只能读取 key 0..${query}；未来 attention mass = 0。`;
    }

    function fitCanvas(canvas) {
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      const width = Math.max(280, canvas.clientWidth);
      const height = Math.max(220, canvas.clientHeight);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { context, width, height };
    }

    function renderRoPE() {
      const step = Number(ropePosition.value);
      byId("ropePositionOut").textContent = String(step);
      const canvas = byId("ropeCanvas");
      const { context, width, height } = fitCanvas(canvas);
      const center = [width / 2, height / 2];
      const original = [82, -46];
      const angle = step * Math.PI / 8;
      const rotated = rotate2D(original, angle);
      context.clearRect(0, 0, width, height);
      context.strokeStyle = "#496156";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(24, center[1]); context.lineTo(width - 24, center[1]);
      context.moveTo(center[0], 20); context.lineTo(center[0], height - 20);
      context.stroke();
      function arrow(vector, color, label) {
        context.strokeStyle = color;
        context.fillStyle = color;
        context.lineWidth = 4;
        context.beginPath();
        context.moveTo(center[0], center[1]);
        context.lineTo(center[0] + vector[0], center[1] + vector[1]);
        context.stroke();
        context.font = "12px monospace";
        context.fillText(label, center[0] + vector[0] + 8, center[1] + vector[1]);
      }
      arrow(original, "#afc0b8", "q");
      arrow(rotated, "#c39a4a", `RoPE(q, ${step})`);
      readout.textContent = `旋转角 ${(angle * 180 / Math.PI).toFixed(1)}°；范数 ${Math.hypot(...original).toFixed(2)} → ${Math.hypot(...rotated).toFixed(2)}。`;
    }

    function barRow(label, value, maximum = 5) {
      const row = document.createElement("div");
      row.className = "bar-row";
      const name = document.createElement("span");
      name.textContent = label;
      const track = document.createElement("div");
      track.className = "bar-track";
      const bar = document.createElement("span");
      const width = clamp(Math.abs(value) / maximum * 50, 0, 50);
      bar.style.width = `${width}%`;
      if (value < 0) bar.classList.add("negative");
      track.appendChild(bar);
      const number = document.createElement("strong");
      number.textContent = value.toFixed(3);
      row.append(name, track, number);
      return row;
    }

    function renderRMS() {
      const scale = Number(rmsScale.value);
      byId("rmsScaleOut").textContent = `${scale.toFixed(1)}×`;
      const input = [1, -2, 0.6, 3].map((value) => value * scale);
      const output = rmsNormalize(input);
      const rows = [];
      input.forEach((value, index) => rows.push(barRow(`x${index}`, value)));
      output.forEach((value, index) => rows.push(barRow(`norm${index}`, value)));
      byId("rmsBars").replaceChildren(...rows);
      const inputRMS = Math.sqrt(input.reduce((sum, value) => sum + value * value, 0) / input.length);
      const outputRMS = Math.sqrt(output.reduce((sum, value) => sum + value * value, 0) / output.length);
      readout.textContent = `输入 RMS ${inputRMS.toFixed(3)} → 输出 RMS ${outputRMS.toFixed(3)}；整体缩放被归一化。`;
    }

    function renderSwiGLU() {
      const input = Number(swigluInput.value);
      byId("swigluInputOut").textContent = input.toFixed(1);
      const gate = silu(input);
      const value = 1.2 * input;
      const output = gate * value;
      const reluSquared = Math.max(0, input) ** 2;
      byId("swigluBars").replaceChildren(
        barRow("SiLU gate", gate, 16),
        barRow("value", value, 16),
        barRow("SwiGLU", output, 16),
        barRow("ReLU²", reluSquared, 16)
      );
      readout.textContent = `SiLU(${input.toFixed(1)}) × ${(1.2 * input).toFixed(2)} = ${output.toFixed(3)}；nanochat 当前 ReLU² = ${reluSquared.toFixed(3)}。`;
    }

    function render() {
      if (activeMode === "causal") renderCausal();
      if (activeMode === "rope") renderRoPE();
      if (activeMode === "rms") renderRMS();
      if (activeMode === "swiglu") renderSwiGLU();
    }

    tabs.forEach((tab) => tab.addEventListener("click", () => {
      activeMode = tab.dataset.decoderMode;
      tabs.forEach((item) => {
        const active = item === tab;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      panels.forEach((panel) => { panel.hidden = panel.dataset.modePanel !== activeMode; });
      render();
    }));
    [position, ropePosition, rmsScale, swigluInput].forEach((control) => control.addEventListener("input", render));
    byId("decoderReset").addEventListener("click", () => {
      position.value = "3"; ropePosition.value = "2"; rmsScale.value = "2"; swigluInput.value = "0.8";
      tabs[0].click();
    });
    let resizeTimer;
    window.addEventListener("resize", () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => { if (activeMode === "rope") renderRoPE(); }, 100);
    });
    render();
  }

  function initializeTeacherLab() {
    const control = byId("teacherPosition");
    const fault = byId("teacherFault");
    const inputTokens = ["<bos>", "我", "喜欢", "机器", "学习"];
    const targetTokens = ["我", "喜欢", "机器", "学习", "<eos>"];
    const probabilities = [0.78, 0.66, 0.72, 0.58, 0.81];

    function renderTokens(container, tokens, active) {
      container.replaceChildren(...tokens.map((token, index) => {
        const span = document.createElement("span");
        span.textContent = token;
        span.classList.toggle("active", index === active);
        return span;
      }));
    }

    function render() {
      const position = Number(control.value);
      const faulty = fault.checked;
      const shownTargets = faulty ? inputTokens : targetTokens;
      const shownProbabilities = faulty ? [0.92, 0.89, 0.91, 0.9, 0.88] : probabilities;
      const losses = shownProbabilities.map((probability) => -Math.log(probability));
      const meanLoss = losses.reduce((sum, value) => sum + value, 0) / losses.length;
      byId("teacherPositionOut").textContent = String(position);
      renderTokens(byId("teacherInputs"), inputTokens, position);
      renderTokens(byId("teacherTargets"), shownTargets, position);
      byId("teacherLosses").replaceChildren(...losses.map((loss, index) => {
        const item = document.createElement("span");
        item.style.boxShadow = `inset 0 ${Math.max(4, Math.min(64, loss * 70))}px #c39a4a`;
        item.textContent = loss.toFixed(2);
        if (index === position) item.style.outline = "2px solid #edf3ef";
        return item;
      }));
      byId("teacherProbability").textContent = shownProbabilities[position].toFixed(2);
      byId("teacherNll").textContent = losses[position].toFixed(3);
      byId("teacherPpl").textContent = Math.exp(meanLoss).toFixed(3);
      byId("teacherReadout").textContent = faulty
        ? "错误模式把输入 token 当成同位置目标：数值看似更容易，却没有训练 next-token prediction。"
        : `位置 ${position} 用真实前缀预测 ${targetTokens[position]}；五个位置在一次 causal forward 中并行监督。`;
    }

    control.addEventListener("input", render);
    fault.addEventListener("change", render);
    byId("teacherReset").addEventListener("click", () => { control.value = "2"; fault.checked = false; render(); });
    render();
  }

  function initializePrefillLab() {
    const promptControl = byId("promptLength");
    const generatedControl = byId("generatedLength");

    function render() {
      const prompt = Number(promptControl.value);
      const generated = Number(generatedControl.value);
      const work = prefillDecodeWork(prompt, generated);
      byId("promptLengthOut").textContent = String(prompt);
      byId("generatedLengthOut").textContent = String(generated);
      byId("prefillTokens").textContent = String(prompt);
      byId("decodeSteps").textContent = String(generated);
      byId("naiveWork").textContent = String(work.naive);
      byId("cachedWork").textContent = String(work.cached);
      const items = [];
      for (let index = 0; index < prompt; index += 1) {
        const item = document.createElement("span");
        item.textContent = `p${index + 1}`;
        item.style.height = `${45 + (index / Math.max(1, prompt - 1)) * 48}px`;
        items.push(item);
      }
      for (let index = 0; index < generated; index += 1) {
        const item = document.createElement("span");
        item.className = "decode";
        item.textContent = `d${index + 1}`;
        item.style.height = `${98 + index * 3}px`;
        items.push(item);
      }
      byId("prefillTimeline").replaceChildren(...items);
      byId("prefillReadout").textContent = `prefill 并行处理 ${prompt} 个 prompt token；随后 ${generated} 次串行选择。按 token-forward 单位，无 cache ${work.naive}，有 cache ${work.cached}。`;
    }

    [promptControl, generatedControl].forEach((control) => control.addEventListener("input", render));
    byId("prefillReset").addEventListener("click", () => { promptControl.value = "8"; generatedControl.value = "5"; render(); });
    render();
  }

  function initializeKVLab() {
    const root = byId("lab-kv");
    const controls = {
      layers: byId("kvLayers"), kvHeads: byId("kvHeads"), headDim: byId("kvHeadDim"),
      dtypeBytes: byId("kvDtype"), batch: byId("kvBatch"), context: byId("kvContext")
    };
    const error = root.querySelector(".lab-error");

    function readValues() {
      const output = {};
      Object.entries(controls).forEach(([name, control]) => { output[name] = Number(control.value); });
      return output;
    }

    function render() {
      try {
        const values = readValues();
        const metrics = kvCacheMetrics(values);
        error.hidden = true;
        byId("kvShape").textContent = `[${values.batch},${values.context},${values.kvHeads},${values.headDim}] × K,V`;
        byId("kvPerToken").textContent = formatBytes(metrics.bytesPerToken * values.batch);
        byId("kvTotal").textContent = formatBytes(metrics.totalBytes);
        byId("kvRead").textContent = `${formatBytes(metrics.totalBytes)} / decode step`;
        const layerBars = [];
        const shown = Math.min(12, Math.round(values.layers));
        const contextScale = clamp(Math.log2(values.context + 1) / 20, 0.12, 1);
        for (let index = 0; index < shown; index += 1) {
          const bar = document.createElement("span");
          bar.style.transform = `scaleX(${contextScale})`;
          bar.title = `layer ${index + 1}: K + V`;
          layerBars.push(bar);
        }
        byId("kvStack").replaceChildren(...layerBars);
        byId("kvReadout").textContent = `${values.layers} 层 × K/V × ${values.kvHeads} KV heads；context 每增加 1 token，batch cache 增加 ${formatBytes(metrics.bytesPerToken * values.batch)}。`;
      } catch (validationError) {
        error.hidden = false;
        byId("kvReadout").textContent = validationError.message;
      }
    }

    Object.values(controls).forEach((control) => control.addEventListener("input", render));
    byId("kvReset").addEventListener("click", () => {
      const defaults = { layers: 32, kvHeads: 8, headDim: 128, dtypeBytes: 2, batch: 1, context: 4096 };
      Object.entries(defaults).forEach(([name, value]) => { controls[name].value = String(value); });
      render();
    });
    render();
  }

  function initializeSamplingLab() {
    const policy = byId("samplingPolicy");
    const temperatureControl = byId("temperature");
    const topPControl = byId("topP");
    const repetitionControl = byId("repetitionPenalty");
    const sampleButton = byId("sampleNext");
    const tokens = ["模型", "可以", "生成", "重复", "<eos>"];
    const baseLogits = [2.8, 2.1, 1.2, 0.9, -0.4];
    let generated = [];
    let stopped = false;

    function distribution() {
      const logits = baseLogits.map((value, index) => index === 4 ? value + generated.length * 1.35 : value - generated.length * index * 0.04);
      const adjusted = applyRepetitionPenalty(logits, new Set(generated), Number(repetitionControl.value));
      const temperature = Number(temperatureControl.value);
      if (policy.value === "greedy" || temperature === 0) {
        const best = adjusted.indexOf(Math.max(...adjusted));
        const probabilities = softmax(adjusted, 1);
        return { adjusted, probabilities, candidates: [best], selected: best };
      }
      const probabilities = softmax(adjusted, temperature);
      const candidates = policy.value === "topp" ? topPIndices(probabilities, Number(topPControl.value)) : probabilities.map((_, index) => index);
      const selected = categoricalSample(probabilities, candidates, 29 + generated.length * 17);
      return { adjusted, probabilities, candidates, selected };
    }

    function render(previewOnly = true) {
      const current = distribution();
      byId("temperatureOut").textContent = Number(temperatureControl.value).toFixed(1);
      byId("topPOut").textContent = Number(topPControl.value).toFixed(2);
      byId("repetitionPenaltyOut").textContent = Number(repetitionControl.value).toFixed(1);
      byId("probabilityBars").replaceChildren(...tokens.map((token, index) => {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.classList.toggle("excluded", !current.candidates.includes(index));
        const name = document.createElement("span"); name.textContent = token;
        const track = document.createElement("div"); track.className = "prob-track";
        const bar = document.createElement("span"); bar.style.width = `${(current.probabilities[index] * 100).toFixed(2)}%`; track.appendChild(bar);
        const number = document.createElement("strong"); number.textContent = current.probabilities[index].toFixed(3);
        row.append(name, track, number);
        return row;
      }));
      byId("candidateSet").textContent = current.candidates.map((index) => tokens[index]).join(" · ");
      byId("generatedTokens").replaceChildren(...generated.map((tokenId) => {
        const item = document.createElement("span");
        item.textContent = tokens[tokenId];
        item.classList.toggle("eos", tokenId === 4);
        return item;
      }));
      byId("selectedToken").textContent = previewOnly ? `next: ${tokens[current.selected]}` : tokens[generated.at(-1)] || "—";
      byId("samplingEntropy").textContent = entropy(current.probabilities).toFixed(3);
      byId("stopState").textContent = stopped ? "stopped" : "running";
      sampleButton.disabled = stopped;
      byId("samplingReadout").textContent = stopped
        ? (generated.at(-1) === 4 ? "采样到 <eos>，当前序列停止；后续 logits 不再消费。" : "达到 max tokens，作为硬边界停止。")
        : `${policy.value} 策略保留 ${current.candidates.length}/${tokens.length} 个候选；重复 token 的 logits 已按 penalty 调整。`;
      return current;
    }

    function sampleNext() {
      const current = distribution();
      generated.push(current.selected);
      if (current.selected === 4 || generated.length >= 8) stopped = true;
      render(false);
    }

    [policy, temperatureControl, topPControl, repetitionControl].forEach((control) => control.addEventListener("input", () => render(true)));
    sampleButton.addEventListener("click", sampleNext);
    byId("samplingReset").addEventListener("click", () => {
      generated = []; stopped = false; policy.value = "temperature"; temperatureControl.value = "1"; topPControl.value = "0.85"; repetitionControl.value = "1.2"; render(true);
    });
    render(true);
  }

  document.addEventListener("DOMContentLoaded", () => {
    safeInit("navigation", initializeNavigation);
    safeInit("copy", initializeCopyButtons);
    safeInit("bpe", initializeBPELab);
    safeInit("decoder", initializeDecoderLab);
    safeInit("teacher", initializeTeacherLab);
    safeInit("prefill", initializePrefillLab);
    safeInit("kv", initializeKVLab);
    safeInit("sampling", initializeSamplingLab);
  });
})(typeof globalThis !== "undefined" ? globalThis : this);
