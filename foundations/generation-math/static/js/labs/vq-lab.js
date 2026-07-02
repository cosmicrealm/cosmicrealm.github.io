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
    const canvas = document.getElementById('vqCanvas');
    const xInput = document.getElementById('vqPointX');
    const yInput = document.getElementById('vqPointY');
    const mode = document.getElementById('vqUpdateMode');
    const readout = document.getElementById('vqPerplexity');
    const bars = document.getElementById('vqUsageBars');
    if (!canvas || !xInput || !yInput || !mode || !readout || !bars) return;
    const xOut = document.getElementById('vqPointXOut');
    const yOut = document.getElementById('vqPointYOut');
    const codes = [
      { x: 0.16, y: 0.28, usage: 0.28 },
      { x: 0.32, y: 0.76, usage: 0.18 },
      { x: 0.55, y: 0.48, usage: 0.34 },
      { x: 0.78, y: 0.72, usage: 0.07 },
      { x: 0.84, y: 0.25, usage: 0.13 }
    ];

    function draw() {
      const px = Number(xInput.value);
      const py = Number(yInput.value);
      if (xOut) xOut.textContent = px.toFixed(2);
      if (yOut) yOut.textContent = py.toFixed(2);
      let best = 0;
      let bestD = 1e9;
      codes.forEach(function (code, idx) {
        const d = Math.hypot(px - code.x, py - code.y);
        if (d < bestD) {
          bestD = d;
          best = idx;
        }
      });
      const { ctx, width, height } = setup(canvas);
      const pad = 36;
      function sx(x) { return pad + x * (width - 2 * pad); }
      function sy(y) { return height - pad - y * (height - 2 * pad); }
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#171817';
      ctx.fillText('codebook nearest-neighbor quantization', 42, 28);
      codes.forEach(function (code, idx) {
        ctx.fillStyle = idx === best ? '#a9432f' : '#0f6f68';
        ctx.beginPath();
        ctx.arc(sx(code.x), sy(code.y), idx === best ? 8 : 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText('e' + idx, sx(code.x) + 8, sy(code.y) - 8);
      });
      ctx.strokeStyle = '#a9432f';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sx(px), sy(py));
      ctx.lineTo(sx(codes[best].x), sy(codes[best].y));
      ctx.stroke();
      ctx.fillStyle = '#171817';
      ctx.beginPath();
      ctx.arc(sx(px), sy(py), 5, 0, Math.PI * 2);
      ctx.fill();

      const entropy = -codes.reduce(function (sum, code) {
        const u = Math.max(code.usage, 1e-8);
        return sum + u * Math.log(u);
      }, 0);
      const perplexity = Math.exp(entropy);
      readout.textContent = 'nearest code = e' + best + ', perplexity = ' + perplexity.toFixed(2) + ', update = ' + mode.value;
      bars.innerHTML = '';
      codes.forEach(function (code, idx) {
        const row = document.createElement('div');
        row.className = 'lab-row' + (code.usage < 0.08 ? ' warn' : '');
        row.innerHTML = '<span>e' + idx + (code.usage < 0.08 ? ' / dead risk' : '') + '</span><i style="--level:' + (code.usage * 100).toFixed(1) + '%"></i><span>' + code.usage.toFixed(2) + '</span>';
        bars.appendChild(row);
      });
    }

    [xInput, yInput, mode].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
