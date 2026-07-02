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
    const canvas = document.getElementById('ganLabCanvas');
    const loss = document.getElementById('ganLoss');
    const mode = document.getElementById('ganModeLab');
    const readout = document.getElementById('ganLabReadout');
    if (!canvas || !loss || !mode || !readout) return;

    function draw() {
      const { ctx, width, height } = setup(canvas);
      const pad = 34;
      function sx(x) { return pad + (x + 3) / 6 * (width - 2 * pad); }
      function sy(y) { return height - pad - (y + 2.2) / 4.4 * (height - 2 * pad); }
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#d7cdbc';
      ctx.lineWidth = 1;
      for (let x = -3; x <= 3; x += 1) {
        ctx.beginPath();
        ctx.moveTo(sx(x), pad);
        ctx.lineTo(sx(x), height - pad);
        ctx.stroke();
      }
      const real = [[-1.8, 0.8], [-1.4, 0.55], [1.4, -0.2], [1.7, 0.2], [0.2, 1.1], [0.0, 0.8]];
      const gen = mode.value === 'collapse'
        ? [[-1.1, -0.8], [-1.0, -0.6], [-0.9, -0.45], [-0.8, -0.65], [-1.2, -0.55]]
        : [[-1.7, -0.7], [-1.2, -0.5], [1.2, -1.0], [1.6, -0.75], [0.1, 0.28], [0.35, 0.1]];
      ctx.strokeStyle = '#0f6f68';
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(sx(-2.8), sy(-1.2));
      ctx.bezierCurveTo(sx(-1.0), sy(1.4), sx(1.0), sy(-1.6), sx(2.8), sy(0.9));
      ctx.stroke();
      ctx.setLineDash([]);
      real.forEach(function (p) {
        ctx.fillStyle = '#0f6f68';
        ctx.beginPath();
        ctx.arc(sx(p[0]), sy(p[1]), 5, 0, Math.PI * 2);
        ctx.fill();
      });
      gen.forEach(function (p) {
        ctx.fillStyle = '#a9432f';
        ctx.beginPath();
        ctx.rect(sx(p[0]) - 5, sy(p[1]) - 5, 10, 10);
        ctx.fill();
      });
      ctx.strokeStyle = loss.value === 'wgan' ? '#314f78' : '#9a6d19';
      ctx.lineWidth = 2.5;
      const from = mode.value === 'collapse' ? [-1.0, -0.6] : [0.7, -0.8];
      const to = loss.value === 'minimax' ? [from[0] + 0.25, from[1] + 0.05] : [from[0] + 0.75, from[1] + 0.55];
      ctx.beginPath();
      ctx.moveTo(sx(from[0]), sy(from[1]));
      ctx.lineTo(sx(to[0]), sy(to[1]));
      ctx.stroke();
      ctx.fillStyle = '#171817';
      ctx.fillText('real samples', 42, 28);
      ctx.fillStyle = '#a9432f';
      ctx.fillText('generated samples', 42, 48);
      readout.textContent = loss.value === 'wgan'
        ? 'WGAN-GP 用 critic gradient norm 约束提供更平滑信号。'
        : loss.value === 'minimax'
          ? 'original minimax 在判别器过强时生成器梯度容易变弱。'
          : 'non-saturating loss 在 D 很自信时仍给 G 较强梯度。';
    }

    [loss, mode].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
