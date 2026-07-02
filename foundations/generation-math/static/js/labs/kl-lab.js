(function () {
  const EPS = 1e-8;

  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  function setupCanvas(canvas, fallbackHeight) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(rect.width || canvas.width || 900));
    const height = Math.max(fallbackHeight, Math.floor(rect.height || fallbackHeight));
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function gaussian(x, mu, sigma) {
    const z = (x - mu) / Math.max(sigma, EPS);
    return Math.exp(-0.5 * z * z) / (Math.max(sigma, EPS) * Math.sqrt(2 * Math.PI));
  }

  function init() {
    const canvas = document.getElementById('klCanvas');
    const mean = document.getElementById('klMean');
    const sigma = document.getElementById('klSigma');
    const mix = document.getElementById('klMix');
    const metric = document.getElementById('klMetric');
    const readout = document.getElementById('klReadout');
    const explanation = document.getElementById('klExplanation');
    if (!canvas || !mean || !sigma || !mix || !metric || !readout) return;
    const outputs = {
      mean: document.getElementById('klMeanOut'),
      sigma: document.getElementById('klSigmaOut'),
      mix: document.getElementById('klMixOut')
    };

    function pData(x) {
      return 0.55 * gaussian(x, -1.35, 0.55) + 0.45 * gaussian(x, 1.25, 0.72);
    }

    function qModel(x, mu, sig, weight) {
      return weight * gaussian(x, mu - 0.8, sig) + (1 - weight) * gaussian(x, mu + 1.05, sig * 1.18);
    }

    function draw() {
      const mu = Number(mean.value);
      const sig = Number(sigma.value);
      const weight = Number(mix.value);
      if (outputs.mean) outputs.mean.textContent = mu.toFixed(2);
      if (outputs.sigma) outputs.sigma.textContent = sig.toFixed(2);
      if (outputs.mix) outputs.mix.textContent = weight.toFixed(2);

      const { ctx, width, height } = setupCanvas(canvas, 330);
      const pad = { left: 44, right: 22, top: 24, bottom: 42 };
      const minX = -4;
      const maxX = 4;
      const n = 260;
      const dx = (maxX - minX) / n;
      const rows = [];
      let fkl = 0;
      let rkl = 0;
      let js = 0;
      let w1 = 0;
      let cdfP = 0;
      let cdfQ = 0;
      for (let i = 0; i <= n; i += 1) {
        const x = minX + i * dx;
        const p = Math.max(EPS, pData(x));
        const q = Math.max(EPS, qModel(x, mu, sig, weight));
        const m = 0.5 * (p + q);
        fkl += p * Math.log(p / q) * dx;
        rkl += q * Math.log(q / p) * dx;
        js += 0.5 * p * Math.log(p / m) * dx + 0.5 * q * Math.log(q / m) * dx;
        cdfP += p * dx;
        cdfQ += q * dx;
        w1 += Math.abs(cdfP - cdfQ) * dx;
        rows.push({ x, p, q });
      }

      const maxY = Math.max.apply(null, rows.map(function (d) { return Math.max(d.p, d.q); })) || 1;
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      function sx(x) { return pad.left + ((x - minX) / (maxX - minX)) * plotW; }
      function sy(y) { return pad.top + plotH - (y / maxY) * plotH * 0.92; }

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#ece7dc';
      ctx.lineWidth = 1;
      for (let g = -4; g <= 4; g += 1) {
        ctx.beginPath();
        ctx.moveTo(sx(g), pad.top);
        ctx.lineTo(sx(g), pad.top + plotH);
        ctx.stroke();
      }

      function path(key, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        rows.forEach(function (d, i) {
          const x = sx(d.x);
          const y = sy(d[key]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
      path('p', '#0f6f68');
      path('q', '#a9432f');

      ctx.fillStyle = '#171817';
      ctx.font = '13px sans-serif';
      ctx.fillText('p_data', pad.left + 8, pad.top + 18);
      ctx.fillStyle = '#a9432f';
      ctx.fillText('p_theta', pad.left + 8, pad.top + 36);

      const values = { fkl, rkl, js, w1 };
      const value = values[metric.value] || fkl;
      readout.textContent = metric.options[metric.selectedIndex].text + ' = ' + value.toFixed(4);
      if (explanation) {
        explanation.textContent = metric.value === 'rkl'
          ? 'reverse KL 倾向避开数据低密度区域，toy 情况下常表现为 mode-seeking。'
          : metric.value === 'js'
            ? 'JS 在两个分布分离时容易饱和，这是早期 GAN 训练不稳定的根源之一。'
            : metric.value === 'w1'
              ? 'Wasserstein 距离关注移动质量所需成本，即使支撑集分离也有连续信号。'
              : 'forward KL 会强惩罚数据 mode 被模型漏掉，因此偏 mode-covering。';
      }
    }

    [mean, sigma, mix, metric].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
