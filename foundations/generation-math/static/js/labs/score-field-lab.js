(function () {
  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  function setup(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.floor(rect.width || 900));
    const height = Math.max(300, Math.floor(rect.height || 330));
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function gaussian(x, mu, sigma) {
    const s = Math.max(0.05, sigma);
    const z = (x - mu) / s;
    return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI));
  }

  function score(dist, x, sigma) {
    if (dist === 'laplace') {
      if (Math.abs(x) < 0.02) return 0;
      return x > 0 ? -1 / sigma : 1 / sigma;
    }
    if (dist === 'mixture') {
      const p1 = 0.5 * gaussian(x, -1.2, sigma);
      const p2 = 0.5 * gaussian(x, 1.3, sigma);
      const total = p1 + p2 || 1;
      return (p1 * (-(x + 1.2) / (sigma * sigma)) + p2 * (-(x - 1.3) / (sigma * sigma))) / total;
    }
    return -x / (sigma * sigma);
  }

  function density(dist, x, sigma) {
    if (dist === 'laplace') return Math.exp(-Math.abs(x) / sigma) / (2 * sigma);
    if (dist === 'mixture') return 0.5 * gaussian(x, -1.2, sigma) + 0.5 * gaussian(x, 1.3, sigma);
    return gaussian(x, 0, sigma);
  }

  function initScore() {
    const canvas = document.getElementById('scoreFieldCanvas');
    const dist = document.getElementById('scoreDist');
    const sigma = document.getElementById('scoreSigma');
    const readout = document.getElementById('scoreFieldReadout');
    if (!canvas || !dist || !sigma || !readout) return;
    const sigmaOut = document.getElementById('scoreSigmaOut');

    function draw() {
      const sig = Number(sigma.value);
      if (sigmaOut) sigmaOut.textContent = sig.toFixed(2);
      const { ctx, width, height } = setup(canvas);
      const pad = { left: 42, right: 24, top: 22, bottom: 52 };
      const minX = -4;
      const maxX = 4;
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      function sx(x) { return pad.left + (x - minX) / (maxX - minX) * plotW; }
      function sy(y, maxY) { return pad.top + plotH - y / maxY * plotH * 0.82; }
      const values = [];
      for (let i = 0; i <= 240; i += 1) {
        const x = minX + i / 240 * (maxX - minX);
        values.push({ x, d: density(dist.value, x, sig), s: score(dist.value, x, sig) });
      }
      const maxY = Math.max.apply(null, values.map(function (v) { return v.d; })) || 1;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#171817';
      ctx.fillText('density curve and score arrows', 42, 28);
      ctx.strokeStyle = '#0f6f68';
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach(function (v, i) {
        const x = sx(v.x);
        const y = sy(v.d, maxY);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      for (let x = -3.5; x <= 3.5; x += 0.7) {
        const baseY = height - 34;
        const sc = Math.max(-1, Math.min(1, score(dist.value, x, sig)));
        const x1 = sx(x);
        const x2 = x1 + sc * 28;
        ctx.strokeStyle = '#a9432f';
        ctx.beginPath();
        ctx.moveTo(x1, baseY);
        ctx.lineTo(x2, baseY);
        ctx.stroke();
        ctx.fillStyle = '#a9432f';
        ctx.beginPath();
        ctx.arc(x2, baseY, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      readout.textContent = dist.value + ': score = d log p(x) / dx; sigma = ' + sig.toFixed(2);
    }

    [dist, sigma].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  function initEbm() {
    const canvas = document.getElementById('ebmCanvas');
    if (!canvas) return;
    function draw() {
      const { ctx, width, height } = setup(canvas);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      const pad = 36;
      function energy(x) { return 0.18 * Math.pow(x * x - 2.2, 2) + 0.15 * x; }
      const samples = [];
      let maxE = 0;
      for (let i = 0; i <= 220; i += 1) {
        const x = -3 + i / 220 * 6;
        const e = energy(x);
        maxE = Math.max(maxE, e);
        samples.push({ x, e });
      }
      function sx(x) { return pad + (x + 3) / 6 * (width - 2 * pad); }
      function sy(e) { return height - pad - e / maxE * (height - 2 * pad); }
      ctx.strokeStyle = '#0f6f68';
      ctx.lineWidth = 2;
      ctx.beginPath();
      samples.forEach(function (p, i) {
        if (i === 0) ctx.moveTo(sx(p.x), sy(p.e));
        else ctx.lineTo(sx(p.x), sy(p.e));
      });
      ctx.stroke();
      ctx.strokeStyle = '#a9432f';
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      let x = 2.6;
      for (let k = 0; k < 18; k += 1) {
        const grad = 0.72 * x * (x * x - 2.2) + 0.15;
        const y = energy(x);
        if (k === 0) ctx.moveTo(sx(x), sy(y));
        else ctx.lineTo(sx(x), sy(y));
        x = x - 0.09 * grad + 0.03 * Math.sin(k);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#171817';
      ctx.fillText('energy E_theta(x); Langevin moves toward low-energy basins', 42, 28);
    }
    draw();
    window.addEventListener('resize', draw);
  }

  ready(function () {
    initScore();
    initEbm();
  });
})();
