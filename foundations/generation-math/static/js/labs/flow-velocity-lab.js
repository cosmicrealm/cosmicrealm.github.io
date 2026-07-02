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

  function init() {
    const canvas = document.getElementById('flowLabCanvas');
    const tInput = document.getElementById('flowT');
    const pathMode = document.getElementById('flowPath');
    const velocityMode = document.getElementById('flowVelocityMode');
    const stepsInput = document.getElementById('flowSteps');
    const readout = document.getElementById('flowLabReadout');
    if (!canvas || !tInput || !pathMode || !velocityMode || !stepsInput || !readout) return;
    const tOut = document.getElementById('flowTOut');
    const stepsOut = document.getElementById('flowStepsOut');

    function point(t) {
      const x = -2.5 + 5 * t;
      const bend = pathMode.value === 'curved' ? Math.sin(t * Math.PI) * 1.1 : 0;
      const y = -0.9 + 1.8 * t + bend;
      return { x, y };
    }

    function draw() {
      const t = Number(tInput.value);
      const steps = Number(stepsInput.value);
      if (tOut) tOut.textContent = t.toFixed(2);
      if (stepsOut) stepsOut.textContent = String(steps);
      const { ctx, width, height } = setup(canvas);
      const pad = 42;
      function sx(x) { return pad + (x + 3) / 6 * (width - 2 * pad); }
      function sy(y) { return height - pad - (y + 2.3) / 4.6 * (height - 2 * pad); }
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#0f6f68';
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 80; i += 1) {
        const p = point(i / 80);
        if (i === 0) ctx.moveTo(sx(p.x), sy(p.y));
        else ctx.lineTo(sx(p.x), sy(p.y));
      }
      ctx.stroke();
      const current = point(t);
      const next = point(Math.min(1, t + 0.07));
      const start = point(0);
      const end = point(1);
      let vx = next.x - current.x;
      let vy = next.y - current.y;
      if (velocityMode.value === 'average') {
        vx = end.x - current.x;
        vy = end.y - current.y;
      }
      ctx.strokeStyle = '#a9432f';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(sx(current.x), sy(current.y));
      ctx.lineTo(sx(current.x + vx * 1.5), sy(current.y + vy * 1.5));
      ctx.stroke();
      for (let k = 0; k <= steps; k += 1) {
        const p = point(k / steps);
        ctx.fillStyle = k === 0 ? '#314f78' : k === steps ? '#0f6f68' : '#9a6d19';
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), 5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = '#171817';
      ctx.fillText('ODE path from base to data', 42, 28);
      ctx.fillText('base distribution', sx(start.x) - 32, sy(start.y) + 22);
      ctx.fillText('data distribution', sx(end.x) - 36, sy(end.y) - 14);
      readout.textContent = velocityMode.value === 'average'
        ? 'MeanFlow 视角：用区间平均速度近似多步积分。'
        : 'Flow Matching 视角：采样时沿 instantaneous velocity field 解 ODE。';
    }

    [tInput, pathMode, velocityMode, stepsInput].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
