(function () {
  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  function init() {
    const margin = document.getElementById('postDpoMargin');
    const beta = document.getElementById('postBeta');
    const drift = document.getElementById('postKlDrift');
    const rewards = document.getElementById('postRewards');
    const bars = document.getElementById('postBars');
    const readout = document.getElementById('postLabReadout');
    const explanation = document.getElementById('postLabExplanation');
    if (!margin || !beta || !drift || !rewards || !bars || !readout) return;
    const marginOut = document.getElementById('postDpoMarginOut');
    const betaOut = document.getElementById('postBetaOut');
    const driftOut = document.getElementById('postKlDriftOut');

    function parseRewards() {
      return rewards.value.split(',').map(function (v) { return Number(v.trim()); }).filter(Number.isFinite);
    }

    function row(label, level, text, warn) {
      const div = document.createElement('div');
      div.className = 'lab-row' + (warn ? ' warn' : '');
      div.innerHTML = '<span>' + label + '</span><i style="--level:' + Math.max(0, Math.min(100, level * 100)).toFixed(1) + '%"></i><span>' + text + '</span>';
      return div;
    }

    function update() {
      const m = Number(margin.value);
      const b = Number(beta.value);
      const kld = Number(drift.value);
      if (marginOut) marginOut.textContent = m.toFixed(1);
      if (betaOut) betaOut.textContent = b.toFixed(1);
      if (driftOut) driftOut.textContent = kld.toFixed(2);
      const dpoProb = 1 / (1 + Math.exp(-b * m));
      const dpoLoss = -Math.log(Math.max(1e-8, dpoProb));
      const values = parseRewards();
      const mean = values.length ? values.reduce(function (a, c) { return a + c; }, 0) / values.length : 0;
      const std = values.length ? Math.sqrt(values.reduce(function (a, c) { return a + Math.pow(c - mean, 2); }, 0) / values.length) : 0;
      bars.innerHTML = '';
      bars.appendChild(row('DPO win prob', dpoProb, dpoProb.toFixed(3), dpoProb < 0.5));
      bars.appendChild(row('PPO KL penalty', kld, kld.toFixed(2), kld > 0.65));
      values.forEach(function (value, idx) {
        const adv = std > 1e-6 ? (value - mean) / std : 0;
        bars.appendChild(row('GRPO A' + (idx + 1), 0.5 + adv / 4, adv.toFixed(2), adv < -0.8));
      });
      readout.textContent = 'DPO loss = ' + dpoLoss.toFixed(4) + ', reward mean = ' + mean.toFixed(3) + ', reward std = ' + std.toFixed(3);
      if (explanation) explanation.textContent = kld > 0.65
        ? 'KL drift 过大表示 policy 远离 reference，可能带来 reward hacking 或分布崩坏。'
        : 'reference log-ratio 是 DPO 保留 KL 正则化结构的关键。';
    }

    [margin, beta, drift, rewards].forEach(function (el) { el.addEventListener('input', update); });
    update();
  }

  ready(init);
})();
