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
    const canvas = document.getElementById('vaePlaneCanvas');
    const mu = document.getElementById('vaeLabMu');
    const sigma = document.getElementById('vaeLabSigma');
    const beta = document.getElementById('vaeBeta');
    const readout = document.getElementById('vaeLabReadout');
    const explanation = document.getElementById('vaeLabExplanation');
    if (!canvas || !mu || !sigma || !beta || !readout) return;
    const muOut = document.getElementById('vaeLabMuOut');
    const sigmaOut = document.getElementById('vaeLabSigmaOut');
    const betaOut = document.getElementById('vaeBetaOut');

    function draw() {
      const m = Number(mu.value);
      const s = Math.max(0.05, Number(sigma.value));
      const b = Number(beta.value);
      if (muOut) muOut.textContent = m.toFixed(1);
      if (sigmaOut) sigmaOut.textContent = s.toFixed(1);
      if (betaOut) betaOut.textContent = b.toFixed(1);
      const kl = 0.5 * (m * m + s * s - Math.log(s * s) - 1);
      const { ctx, width, height } = setup(canvas);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      const cx = width * 0.5;
      const cy = height * 0.53;
      const scale = Math.min(width, height) / 7;
      ctx.strokeStyle = '#d7cdbc';
      ctx.lineWidth = 1;
      for (let g = -3; g <= 3; g += 1) {
        ctx.beginPath();
        ctx.moveTo(cx + g * scale, 24);
        ctx.lineTo(cx + g * scale, height - 34);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(36, cy + g * scale);
        ctx.lineTo(width - 36, cy + g * scale);
        ctx.stroke();
      }
      ctx.strokeStyle = '#0f6f68';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, scale, scale, 0, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = '#a9432f';
      ctx.fillStyle = 'rgba(169,67,47,0.12)';
      ctx.beginPath();
      ctx.ellipse(cx + m * scale, cy, s * scale, Math.max(0.25, s) * scale * 0.68, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#0f6f68';
      ctx.fillText('prior N(0,I)', 44, 30);
      ctx.fillStyle = '#a9432f';
      ctx.fillText('q_phi(z|x)', 44, 50);
      const reconPressure = Math.max(0, 1 - b * kl / 5);
      ctx.fillStyle = '#9a6d19';
      ctx.fillRect(44, height - 44, Math.max(4, reconPressure * (width - 88)), 8);
      ctx.fillStyle = '#171817';
      ctx.fillText('reconstruction pressure', 44, height - 54);
      readout.textContent = 'KL(q_phi || p) = ' + kl.toFixed(4) + ', beta * KL = ' + (b * kl).toFixed(4);
      if (explanation) {
        explanation.textContent = b > 2.2
          ? 'beta 较大时 latent 信息率被压低，posterior collapse 风险上升。'
          : 'beta 控制重构细节与 prior 对齐之间的 rate-distortion 权衡。';
      }
    }

    [mu, sigma, beta].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
