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
    const canvas = document.getElementById('nfCanvas');
    const a = document.getElementById('nfA');
    const b = document.getElementById('nfB');
    const readout = document.getElementById('nfReadout');
    if (!canvas || !a || !b || !readout) return;
    const aOut = document.getElementById('nfAOut');
    const bOut = document.getElementById('nfBOut');

    function warp(x, y, av, bv) {
      return {
        x: x * Math.exp(av) + 0.12 * Math.sin(2.2 * y),
        y: y * Math.exp(bv) + 0.12 * Math.sin(2.0 * x)
      };
    }

    function draw() {
      const av = Number(a.value);
      const bv = Number(b.value);
      if (aOut) aOut.textContent = av.toFixed(2);
      if (bOut) bOut.textContent = bv.toFixed(2);
      const { ctx, width, height } = setup(canvas);
      const pad = 34;
      const scale = Math.min(width - 2 * pad, height - 2 * pad) / 5.2;
      const cx = width / 2;
      const cy = height / 2;
      function sx(x) { return cx + x * scale; }
      function sy(y) { return cy - y * scale; }
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#d7cdbc';
      ctx.lineWidth = 1;
      for (let i = -3; i <= 3; i += 1) {
        ctx.beginPath();
        for (let t = -3; t <= 3; t += 0.05) {
          const p = warp(i, t, av, bv);
          if (t === -3) ctx.moveTo(sx(p.x), sy(p.y));
          else ctx.lineTo(sx(p.x), sy(p.y));
        }
        ctx.stroke();
        ctx.beginPath();
        for (let t = -3; t <= 3; t += 0.05) {
          const p = warp(t, i, av, bv);
          if (t === -3) ctx.moveTo(sx(p.x), sy(p.y));
          else ctx.lineTo(sx(p.x), sy(p.y));
        }
        ctx.stroke();
      }
      const logdet = av + bv;
      ctx.fillStyle = '#0f6f68';
      ctx.beginPath();
      const p = warp(0, 0, av, bv);
      ctx.arc(sx(p.x), sy(p.y), 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#171817';
      ctx.fillText('invertible warp grid', 42, 28);
      readout.textContent = 'log |det J| ≈ ' + logdet.toFixed(3) + '；密度校正项为 -log |det J|。';
    }

    [a, b].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
