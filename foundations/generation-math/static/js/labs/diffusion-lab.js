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
    const canvas = document.getElementById('diffCanvas');
    const tInput = document.getElementById('diffT');
    const schedule = document.getElementById('diffSchedule');
    const sampler = document.getElementById('diffSampler');
    const readout = document.getElementById('diffReadout');
    const explanation = document.getElementById('diffExplanation');
    if (!canvas || !tInput || !schedule || !sampler || !readout) return;
    const tOut = document.getElementById('diffTOut');

    function alphaBar(t) {
      if (schedule.value === 'cosine') {
        return Math.pow(Math.cos((t + 0.008) / 1.008 * Math.PI / 2), 2);
      }
      return Math.max(0.001, 1 - 0.94 * t);
    }

    function draw() {
      const t = Number(tInput.value);
      const ab = Math.max(0.001, Math.min(0.999, alphaBar(t)));
      const signal = Math.sqrt(ab);
      const noise = Math.sqrt(1 - ab);
      if (tOut) tOut.textContent = t.toFixed(2);
      const snr = ab / Math.max(1e-6, 1 - ab);
      const { ctx, width, height } = setup(canvas);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      const pad = 42;
      const plotW = width - 2 * pad;
      const mid = height * 0.46;
      ctx.strokeStyle = '#d7cdbc';
      ctx.beginPath();
      ctx.moveTo(pad, mid);
      ctx.lineTo(width - pad, mid);
      ctx.stroke();
      for (let i = 0; i <= 12; i += 1) {
        const x = pad + i / 12 * plotW;
        const ti = i / 12;
        const abi = Math.max(0.001, Math.min(0.999, schedule.value === 'cosine' ? Math.pow(Math.cos((ti + 0.008) / 1.008 * Math.PI / 2), 2) : 1 - 0.94 * ti));
        const radius = 6 + (1 - abi) * 24;
        ctx.fillStyle = 'rgba(15,111,104,' + (0.18 + abi * 0.55).toFixed(3) + ')';
        ctx.beginPath();
        ctx.arc(x, mid, radius, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.strokeStyle = sampler.value === 'ddim' ? '#314f78' : '#a9432f';
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      for (let i = 0; i <= 24; i += 1) {
        const x = width - pad - i / 24 * plotW;
        const y = mid + Math.sin(i * 0.65) * (sampler.value === 'ddpm' ? 22 : 8);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = '#171817';
      ctx.fillText('signal coeff sqrt(alpha_bar) = ' + signal.toFixed(3), pad, 28);
      ctx.fillText('noise coeff sqrt(1-alpha_bar) = ' + noise.toFixed(3), pad, 48);
      readout.textContent = 'SNR = ' + snr.toFixed(3) + ', sampler = ' + sampler.options[sampler.selectedIndex].text;
      if (explanation) explanation.textContent = sampler.value === 'ddpm'
        ? 'DDPM 反向步包含新噪声，轨迹是随机 Markov chain。'
        : 'DDIM / probability flow ODE 去掉新噪声，给出确定性采样轨迹。';
    }

    [tInput, schedule, sampler].forEach(function (el) { el.addEventListener('input', draw); });
    draw();
    window.addEventListener('resize', draw);
  }

  ready(init);
})();
