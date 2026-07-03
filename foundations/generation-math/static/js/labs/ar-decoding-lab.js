(function () {
  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  function initMask() {
    const mask = document.getElementById('causalMask');
    if (!mask) return;
    mask.innerHTML = '';
    for (let r = 0; r < 8; r += 1) {
      for (let c = 0; c < 8; c += 1) {
        const cell = document.createElement('span');
        if (c <= r) cell.className = 'on';
        cell.setAttribute('aria-label', c <= r ? 'visible' : 'masked');
        mask.appendChild(cell);
      }
    }
  }

  function initDecoding() {
    const temp = document.getElementById('arTemp');
    const topK = document.getElementById('arTopK');
    const topP = document.getElementById('arTopP');
    const bars = document.getElementById('arDecodingBars');
    const formula = document.getElementById('arDecodeFormula');
    const explanation = document.getElementById('arDecodeExplanation');
    const sampleButton = document.getElementById('arSampleNext');
    const tokenStream = document.getElementById('arTokenStream');
    if (!temp || !topK || !topP || !bars) return;
    const tempOut = document.getElementById('arTempOut');
    const topKOut = document.getElementById('arTopKOut');
    const topPOut = document.getElementById('arTopPOut');
    const tokens = [
      { token: 'the', logit: 4.1 },
      { token: 'image', logit: 3.3 },
      { token: 'latent', logit: 2.2 },
      { token: 'noise', logit: 1.6 },
      { token: 'reward', logit: 0.9 },
      { token: 'rare', logit: -0.3 }
    ];
    let generated = ['prompt'];
    let selectedToken = '';
    let lastRows = [];

    function sampleFromRows(rows) {
      if (!rows.length) return '';
      const r = Math.random();
      let acc = 0;
      for (let i = 0; i < rows.length; i += 1) {
        acc += rows[i].prob;
        if (r <= acc) return rows[i].token;
      }
      return rows[rows.length - 1].token;
    }

    function renderStream() {
      if (!tokenStream) return;
      tokenStream.innerHTML = generated.map(function (token, idx) {
        const cls = idx === generated.length - 1 ? ' class="active"' : '';
        return '<span' + cls + '>' + token + '</span>';
      }).join('');
    }

    function update() {
      const t = Math.max(0.2, Number(temp.value));
      const k = Math.max(1, Number(topK.value));
      const p = Math.max(0.05, Number(topP.value));
      if (tempOut) tempOut.textContent = t.toFixed(2);
      if (topKOut) topKOut.textContent = String(k);
      if (topPOut) topPOut.textContent = p.toFixed(2);
      let rows = tokens.map(function (item) {
        return { token: item.token, score: Math.exp(item.logit / t) };
      });
      const total = rows.reduce(function (sum, row) { return sum + row.score; }, 0);
      rows.forEach(function (row) { row.prob = row.score / total; });
      rows.sort(function (a, b) { return b.prob - a.prob; });
      rows = rows.slice(0, k);
      let acc = 0;
      rows = rows.filter(function (row, idx) {
        acc += row.prob;
        return idx === 0 || acc - row.prob < p;
      });
      const renorm = rows.reduce(function (sum, row) { return sum + row.prob; }, 0) || 1;
      rows.forEach(function (row) { row.prob /= renorm; });
      lastRows = rows;
      if (!selectedToken || !rows.some(function (row) { return row.token === selectedToken; })) {
        selectedToken = rows[0] ? rows[0].token : '';
      }
      bars.innerHTML = '';
      rows.forEach(function (row) {
        const prob = row.prob;
        const item = document.createElement('div');
        item.className = 'token' + (row.token === selectedToken ? ' selected' : '');
        item.innerHTML = '<span>' + row.token + '</span><span class="bar"><span style="--level:' + (prob * 100).toFixed(1) + '%"></span></span><span>' + prob.toFixed(3) + '</span>';
        bars.appendChild(item);
      });
      renderStream();
      if (formula) formula.textContent = 'candidate set = {' + rows.map(function (row) { return row.token; }).join(', ') + '}; selected = ' + (selectedToken || 'none');
      if (explanation) explanation.textContent = t < 0.8 ? '低 temperature 会集中概率；causal mask 只允许当前位置看见左侧前缀，因此一旦采到重复 token，后续条件也会被它影响。' : '高 temperature 会摊平分布；top-k/top-p 先截断候选集合，再重新归一化采样。';
    }

    [temp, topK, topP].forEach(function (el) { el.addEventListener('input', update); });
    if (sampleButton) {
      sampleButton.addEventListener('click', function () {
        selectedToken = sampleFromRows(lastRows);
        if (selectedToken) {
          generated.push(selectedToken);
          if (generated.length > 7) generated = generated.slice(generated.length - 7);
        }
        update();
      });
    }
    renderStream();
    update();
  }

  ready(function () {
    initMask();
    initDecoding();
  });
})();
