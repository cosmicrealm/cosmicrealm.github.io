(function () {
  const root = document;

  function $(id) {
    return root.getElementById(id);
  }

  function all(selector) {
    return Array.from(root.querySelectorAll(selector));
  }

  function clamp(n, min, max) {
    const value = Number.isFinite(n) ? n : min;
    return Math.max(min, Math.min(max, value));
  }

  function value(id, fallback) {
    const node = $(id);
    if (!node) return fallback;
    const parsed = Number(node.value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function setText(id, text) {
    const node = $(id);
    if (node) node.textContent = text;
  }

  function canvasContext(id) {
    const canvas = $(id);
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
    const width = Math.max(320, rect.width || canvas.width || 900);
    const height = Math.max(180, rect.height || canvas.height || 360);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function clear(ctx, width, height) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#fffdf8';
    ctx.fillRect(0, 0, width, height);
  }

  function text(ctx, content, x, y, color, size, weight, align) {
    ctx.fillStyle = color || '#171b1c';
    ctx.font = `${weight || '700'} ${size || 14}px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial`;
    ctx.textAlign = align || 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(content), x, y);
  }

  function wrapText(ctx, content, x, y, maxWidth, lineHeight, color, size, weight) {
    const words = String(content).split(/\s+/);
    let line = '';
    let yy = y;
    for (let i = 0; i < words.length; i += 1) {
      const test = line ? `${line} ${words[i]}` : words[i];
      if (ctx.measureText(test).width > maxWidth && line) {
        text(ctx, line, x, yy, color, size, weight);
        line = words[i];
        yy += lineHeight;
      } else {
        line = test;
      }
    }
    if (line) text(ctx, line, x, yy, color, size, weight);
  }

  function bar(ctx, x, y, width, height, pct, color, label, valueLabel) {
    const safePct = clamp(pct, 0, 1);
    ctx.fillStyle = '#eef3ef';
    ctx.fillRect(x, y, width, height);
    ctx.fillStyle = color;
    ctx.fillRect(x, y, width * safePct, height);
    ctx.strokeStyle = '#d4ddd7';
    ctx.strokeRect(x, y, width, height);
    text(ctx, label, x, y - 12, '#5c6466', 12, '800');
    text(ctx, valueLabel || `${Math.round(safePct * 100)}%`, x + width + 12, y + height / 2, '#171b1c', 13, '900');
  }

  function bindToc() {
    const toggle = $('tocToggle');
    const toc = $('lectureToc');
    if (toggle && toc) {
      toggle.addEventListener('click', () => {
        const open = toc.classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
      });
      toc.addEventListener('click', (event) => {
        if (event.target && event.target.tagName === 'A') {
          toc.classList.remove('open');
          toggle.setAttribute('aria-expanded', 'false');
        }
      });
    }

    const links = all('.lecture-toc a');
    const targets = links.map((link) => root.querySelector(link.getAttribute('href'))).filter(Boolean);
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          links.forEach((link) => {
            link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`);
          });
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
      barEl.style.width = `${clamp(pct, 0, 100)}%`;
    }
    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
  }

  function drawPipelineLab() {
    const canvas = canvasContext('pipelineCanvas');
    if (!canvas) return;
    const { ctx, width, height } = canvas;
    const textFocus = value('mixText', 55);
    const editFocus = value('mixEdit', 45);
    const rewardFocus = value('mixReward', 35);
    setText('mixTextOut', String(textFocus));
    setText('mixEditOut', String(editFocus));
    setText('mixRewardOut', String(rewardFocus));
    clear(ctx, width, height);

    const raw = [
      { label: 'general image-text', base: 70, color: '#4c5fb1' },
      { label: 'text-rich OCR', base: 18 + textFocus * 0.9, color: '#0f8a5f' },
      { label: 'editing pairs', base: 14 + editFocus * 0.95, color: '#0f766e' },
      { label: 'preference / RL', base: 8 + rewardFocus * 0.85, color: '#b46b13' },
      { label: 'distillation traces', base: 10 + Math.max(editFocus, rewardFocus) * 0.45, color: '#a43c31' }
    ];
    const total = raw.reduce((sum, item) => sum + item.base, 0);
    const items = raw.map((item) => ({ ...item, pct: item.base / Math.max(total, 1) }));
    const pad = 44;
    const chartW = width - pad * 2;
    let y = 62;
    text(ctx, 'Toy data budget allocation', pad, 28, '#171b1c', 18, '900');
    items.forEach((item) => {
      bar(ctx, pad, y, chartW * 0.72, 24, item.pct * 3.1, item.color, item.label, `${Math.round(item.pct * 100)}%`);
      y += 52;
    });
    const specialty = clamp((textFocus + editFocus + rewardFocus) / 300, 0, 1);
    const risk = clamp((rewardFocus * 0.55 + textFocus * 0.25 - editFocus * 0.1) / 100, 0, 1);
    bar(ctx, width - 220, 92, 130, 24, specialty, '#0f766e', 'specialization');
    bar(ctx, width - 220, 174, 130, 24, risk, '#a43c31', 'narrowing risk');
    wrapText(ctx, 'More specialized targets require explicit data pressure: OCR samples, source-target edits, preference pairs, and teacher trajectories.', width - 220, 258, 165, 18, '#5c6466', 13, '760');
    setText('pipelineReadout', `general=${Math.round(items[0].pct * 100)}% · text-rich=${Math.round(items[1].pct * 100)}% · editing=${Math.round(items[2].pct * 100)}% · reward=${Math.round(items[3].pct * 100)}%`);
  }

  function drawVaeLab() {
    const canvas = canvasContext('vaeCanvas');
    if (!canvas) return;
    const { ctx, width, height } = canvas;
    const compression = value('vaeCompression', 16);
    const textData = value('vaeText', 65);
    const align = value('vaeAlign', 55);
    setText('vaeCompressionOut', String(compression));
    setText('vaeTextOut', String(textData));
    setText('vaeAlignOut', String(align));
    clear(ctx, width, height);

    const fidelity = clamp((105 - compression * 2.25 + textData * 0.45 + align * 0.25) / 100, 0, 1);
    const diffusability = clamp((align * 0.65 + (32 - Math.abs(compression - 18)) * 1.5) / 100, 0, 1);
    const ocrRisk = clamp((compression * 2.2 - textData * 0.55 + 25) / 100, 0, 1);
    const leftX = 46;
    const topY = 56;
    text(ctx, 'Reconstruction toy canvas', leftX, 28, '#171b1c', 18, '900');
    ctx.strokeStyle = '#d4ddd7';
    ctx.lineWidth = 1;
    ctx.strokeRect(leftX, topY, width * 0.42, height - 110);
    ctx.strokeRect(leftX + width * 0.46, topY, width * 0.42, height - 110);
    text(ctx, 'input', leftX + 14, topY + 22, '#5c6466', 13, '900');
    text(ctx, 'reconstructed latent decode', leftX + width * 0.46 + 14, topY + 22, '#5c6466', 13, '900');

    function drawPoster(x, y, scale, degrade) {
      ctx.fillStyle = '#eef3ef';
      ctx.fillRect(x, y, 170 * scale, 185 * scale);
      ctx.fillStyle = '#171b1c';
      ctx.fillRect(x + 20 * scale, y + 24 * scale, 112 * scale * (1 - degrade * 0.12), 13 * scale);
      ctx.fillStyle = '#0f766e';
      ctx.fillRect(x + 20 * scale, y + 52 * scale, 132 * scale * (1 - degrade * 0.2), 8 * scale);
      ctx.fillStyle = '#4c5fb1';
      ctx.fillRect(x + 20 * scale, y + 75 * scale, 55 * scale, 55 * scale);
      ctx.fillStyle = '#b46b13';
      ctx.fillRect(x + 88 * scale, y + 75 * scale, 64 * scale, 55 * scale);
      ctx.fillStyle = '#171b1c';
      const lines = Math.round(7 - degrade * 4);
      for (let i = 0; i < lines; i += 1) {
        const w = (122 - i * 7) * scale * (1 - degrade * 0.18);
        ctx.globalAlpha = 1 - degrade * 0.55;
        ctx.fillRect(x + 22 * scale, y + (145 + i * 8) * scale, Math.max(12, w), Math.max(1.2, 2.2 * scale * (1 - degrade)));
      }
      ctx.globalAlpha = 1;
    }

    drawPoster(leftX + 38, topY + 58, 1, 0);
    drawPoster(leftX + width * 0.46 + 38, topY + 58, 1, 1 - fidelity);
    bar(ctx, width - 248, 82, 138, 22, fidelity, '#0f8a5f', 'fidelity');
    bar(ctx, width - 248, 150, 138, 22, diffusability, '#4c5fb1', 'diffusability');
    bar(ctx, width - 248, 218, 138, 22, ocrRisk, '#a43c31', 'OCR risk');
    setText('vaeReadout', `fidelity=${Math.round(fidelity * 100)} · diffusability=${Math.round(diffusability * 100)} · OCR risk=${Math.round(ocrRisk * 100)} · compression=${compression}x`);
  }

  function drawPosttrainingLab() {
    const canvas = canvasContext('posttrainingCanvas');
    if (!canvas) return;
    const { ctx, width, height } = canvas;
    const reward = value('rlReward', 55);
    const distill = value('rlDistill', 65);
    const diversityReserve = value('rlDiversity', 42);
    setText('rlRewardOut', String(reward));
    setText('rlDistillOut', String(distill));
    setText('rlDiversityOut', String(diversityReserve));
    clear(ctx, width, height);

    const alignment = clamp((35 + reward * 0.55 + diversityReserve * 0.12 - distill * 0.06) / 100, 0, 1);
    const speed = clamp((25 + distill * 0.7) / 100, 0, 1);
    const diversity = clamp((65 + diversityReserve * 0.42 - reward * 0.22 - distill * 0.2) / 100, 0, 1);
    const artifactRisk = clamp((distill * 0.34 + reward * 0.22 - diversityReserve * 0.18) / 100, 0, 1);
    const x = 70;
    text(ctx, 'Post-training output frontier', x, 30, '#171b1c', 18, '900');
    bar(ctx, x, 82, width - 260, 26, alignment, '#0f8a5f', 'instruction / preference alignment');
    bar(ctx, x, 150, width - 260, 26, speed, '#4c5fb1', 'sampling speed');
    bar(ctx, x, 218, width - 260, 26, diversity, '#b46b13', 'output diversity');
    bar(ctx, x, 286, width - 260, 26, artifactRisk, '#a43c31', 'artifact risk');
    const noteX = width - 220;
    wrapText(ctx, 'A strong reward model can improve alignment, but aggressive few-step distillation and narrow rewards can reduce diversity or introduce artifacts.', noteX, 105, 165, 18, '#5c6466', 13, '760');
    setText('posttrainingReadout', `alignment=${Math.round(alignment * 100)} · speed=${Math.round(speed * 100)} · diversity=${Math.round(diversity * 100)} · artifact risk=${Math.round(artifactRisk * 100)}`);
  }

  function initLabs() {
    ['mixText', 'mixEdit', 'mixReward'].forEach((id) => $(id)?.addEventListener('input', drawPipelineLab));
    ['vaeCompression', 'vaeText', 'vaeAlign'].forEach((id) => $(id)?.addEventListener('input', drawVaeLab));
    ['rlReward', 'rlDistill', 'rlDiversity'].forEach((id) => $(id)?.addEventListener('input', drawPosttrainingLab));
    drawPipelineLab();
    drawVaeLab();
    drawPosttrainingLab();
    window.addEventListener('resize', () => {
      drawPipelineLab();
      drawVaeLab();
      drawPosttrainingLab();
    });
  }

  function init() {
    bindToc();
    bindProgress();
    initLabs();
  }

  if (root.readyState === 'loading') {
    root.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
