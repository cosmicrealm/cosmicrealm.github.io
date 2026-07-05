(function () {
  const root = document;
  const tokenNames = ['math', 'model', 'data', 'loss', 'sample', 'align'];
  const baseLogits = [2.35, 1.68, 1.12, 0.72, 0.25, -0.18];
  const groupMeta = [
    {
      group: '线性代数与张量',
      slug: 'linear',
      intro: '先把标量、向量、矩阵、张量和分解工具讲清楚；LLM 里的 attention、embedding、LoRA 与表示空间都建立在这些对象上。',
      relation: '主线关系：shape 约束计算能否发生，矩阵乘法实现表示变换，谱分解和低秩近似解释压缩、投影与参数高效微调。'
    },
    {
      group: '微积分与自动微分',
      slug: 'calculus',
      intro: '这一组概念回答“模型如何知道该往哪里改参数”。从导数、梯度到 VJP/JVP，最终落到反向传播和自动微分。',
      relation: '主线关系：loss 是标量目标，计算图保存前向依赖，反向模式用链式法则把梯度传回每个参数。'
    },
    {
      group: '概率统计与信息论',
      slug: 'probability',
      intro: 'LLM 是条件分布，Diffusion 是随机过程，评测分数也是随机变量。先分清 PMF、PDF、期望、熵、KL 和 Bayes。',
      relation: '主线关系：概率定义建模对象，信息论定义训练代价，统计推断说明估计值是否可信。',
      lab: 'softmaxLabBlock'
    },
    {
      group: '优化与数值计算',
      slug: 'optimization',
      intro: '梯度只是开始；真正的训练稳定性来自优化器、学习率、正则、精度格式、条件数和硬件访存约束共同作用。',
      relation: '主线关系：目标函数给方向，优化器决定步长和尺度，数值系统决定这个更新能否稳定且高效地执行。'
    },
    {
      group: '深度学习机制',
      slug: 'deeplearning',
      intro: '这些概念解释神经网络内部如何组织计算：线性层、非线性、残差、归一化和初始化共同决定可训练性。',
      relation: '主线关系：MLP 提供非线性容量，残差和归一化维持信号流，初始化与正则控制训练早期动态。'
    },
    {
      group: 'Transformer 与 LLM',
      slug: 'transformer',
      intro: 'Transformer 把条件概率建模做成可扩展系统：token 化、位置、attention、MLP、decoding 和高效微调在这里汇合。',
      relation: '主线关系：attention 混合上下文，MLP 改写每个 token 表示，LM head 输出 next-token 分布，decoding 把分布变成文本。',
      lab: 'attentionLabBlock'
    },
    {
      group: '生成模型',
      slug: 'generation',
      intro: 'VAE、GAN、Diffusion、Flow Matching 都在匹配数据分布，但它们暴露的可训练信号和采样接口不同。',
      relation: '主线关系：隐变量定义生成空间，分布距离定义目标，随机过程或 ODE/SDE 定义从噪声到样本的路径。',
      lab: 'generationLabBlock'
    },
    {
      group: '对齐与偏好优化',
      slug: 'alignment',
      intro: '对齐把“哪个回答更好”转成可训练信号。关键是偏好数据、reward、policy、KL 约束与 DPO 目标之间的关系。',
      relation: '主线关系：偏好 pair 给相对监督，reference model 控制漂移，policy objective 把偏好转成分布更新。',
      lab: 'dpoLabBlock'
    },
    {
      group: '评测统计与泛化',
      slug: 'evaluation',
      intro: '模型评测不能只看单个分数。benchmark、win-rate、置信区间、bootstrap 和校准决定结论能否站住。',
      relation: '主线关系：指标是样本统计量，统计量有方差和偏差，泛化分析说明训练/测试/真实使用分布之间的落差。'
    }
  ];
  const groupByName = new Map(groupMeta.map((meta) => [meta.group, meta]));

  function $(id) {
    return root.getElementById(id);
  }

  function all(selector) {
    return Array.from(root.querySelectorAll(selector));
  }

  function clamp(value, min, max) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return min;
    return Math.max(min, Math.min(max, numeric));
  }

  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }

  function getNumber(id, fallback) {
    const el = $(id);
    if (!el) return fallback;
    return clamp(el.value, Number(el.min || -Infinity), Number(el.max || Infinity));
  }

  function bindOutput(inputId, outputId, digits) {
    const input = $(inputId);
    const output = $(outputId);
    if (!input || !output) return;
    const update = () => {
      const value = Number(input.value);
      output.textContent = Number.isFinite(value) ? value.toFixed(digits) : '';
    };
    input.addEventListener('input', update);
    update();
  }

  function setupCanvas(canvas) {
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(rect.width || canvas.width || 900));
    const height = Math.max(240, Math.round(rect.height || canvas.height || 360));
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function canvasColors() {
    const styles = getComputedStyle(root.documentElement);
    return {
      paper: styles.getPropertyValue('--panel-strong').trim() || '#fff',
      ink: styles.getPropertyValue('--ink').trim() || '#151713',
      muted: styles.getPropertyValue('--muted').trim() || '#62695f',
      line: styles.getPropertyValue('--line').trim() || '#cdd5c5',
      soft: styles.getPropertyValue('--soft').trim() || '#e9ede4',
      accent: styles.getPropertyValue('--accent').trim() || '#0c7068',
      red: styles.getPropertyValue('--red').trim() || '#a94a35',
      gold: styles.getPropertyValue('--gold').trim() || '#9b741b',
      blue: styles.getPropertyValue('--blue').trim() || '#315c83',
      violet: styles.getPropertyValue('--violet').trim() || '#70548f'
    };
  }

  function clear(ctx, width, height, colors) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = colors.paper;
    ctx.fillRect(0, 0, width, height);
  }

  function text(ctx, value, x, y, color, size, weight) {
    ctx.fillStyle = color;
    ctx.font = `${weight || '700'} ${size || 14}px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`;
    ctx.fillText(value, x, y);
  }

  function line(ctx, x1, y1, x2, y2, color, width) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width || 1;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }

  function roundRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map((v) => Math.exp(v - maxLogit));
    const denom = expValues.reduce((sum, v) => sum + v, 0) || 1;
    return expValues.map((v) => v / denom);
  }

  function applyTopP(items, topP) {
    const sorted = items.slice().sort((a, b) => b.prob - a.prob);
    let total = 0;
    const kept = new Set();
    for (const item of sorted) {
      total += item.prob;
      kept.add(item.name);
      if (total >= topP) break;
    }
    const filtered = items.map((item) => ({ ...item, prob: kept.has(item.name) ? item.prob : 0 }));
    const denom = filtered.reduce((sum, item) => sum + item.prob, 0) || 1;
    return filtered.map((item) => ({ ...item, prob: item.prob / denom, kept: kept.has(item.name) }));
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function normalizeText(value) {
    return String(value || '').toLowerCase().replace(/\s+/g, ' ').trim();
  }

  function flattenRichText(value) {
    if (Array.isArray(value)) return value.map(flattenRichText).join(' ');
    if (value && typeof value === 'object') return Object.values(value).map(flattenRichText).join(' ');
    return String(value || '');
  }

  function renderRichText(value) {
    if (Array.isArray(value)) {
      const items = value
        .filter((item) => String(item || '').trim())
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join('');
      return `<ul>${items}</ul>`;
    }
    return `<p>${escapeHtml(value || '')}</p>`;
  }

  function conceptBrief(concept) {
    return concept.brief || {
      intuition: concept.intuition || '',
      math: concept.math || '',
      model: concept.model || '',
      pitfall: concept.pitfall || ''
    };
  }

  function conceptDetail(concept) {
    return concept.detail || conceptBrief(concept);
  }

  function conceptQa(concept) {
    return Array.isArray(concept.qa) ? concept.qa : [];
  }

  function conceptSearchText(concept) {
    const brief = conceptBrief(concept);
    const detail = conceptDetail(concept);
    const qa = conceptQa(concept)
      .map((item) => `${item.question || ''} ${flattenRichText(item.answer)}`)
      .join(' ');
    return [
      concept.group,
      concept.name,
      (concept.aliases || []).join(' '),
      flattenRichText(brief.intuition),
      flattenRichText(brief.math),
      flattenRichText(brief.model),
      flattenRichText(brief.pitfall),
      flattenRichText(detail.intuition),
      flattenRichText(detail.math),
      flattenRichText(detail.model),
      flattenRichText(detail.pitfall),
      qa
    ].join(' ');
  }

  function initConceptHandbook() {
    const concepts = Array.isArray(window.AIGC_LLM_MATH_CONCEPTS) ? window.AIGC_LLM_MATH_CONCEPTS : [];
    const modules = $('conceptModules');
    const taxonomy = $('conceptTaxonomy');
    const search = $('conceptSearch');
    const count = $('conceptCount');
    if (!modules || !taxonomy || !search || !count || !concepts.length) return;

    let activeGroup = '全部';
    const conceptRecords = concepts.map((concept, index) => {
      const meta = groupByName.get(concept.group) || { slug: `group-${index}`, group: concept.group };
      return {
        ...concept,
        index,
        id: `concept-card-${meta.slug}-${index}`
      };
    });
    const allGroups = groupMeta
      .map((meta) => meta.group)
      .filter((group) => conceptRecords.some((concept) => concept.group === group));
    const groups = ['全部'].concat(allGroups);

    function renderTaxonomy() {
      taxonomy.innerHTML = groups.map((group) => {
        const total = group === '全部' ? conceptRecords.length : conceptRecords.filter((concept) => concept.group === group).length;
        return `<button type="button" class="${group === activeGroup ? 'active' : ''}" data-group="${escapeHtml(group)}">${escapeHtml(group)}<span>${total}</span></button>`;
      }).join('');
      all('#conceptTaxonomy button').forEach((button) => {
        button.addEventListener('click', () => {
          activeGroup = button.getAttribute('data-group') || '全部';
          render({ scrollToGroup: activeGroup !== '全部' });
        });
      });
    }

    function matches(concept, query) {
      if (!query) return true;
      return normalizeText(conceptSearchText(concept)).includes(query);
    }

    function searchScore(concept, query) {
      if (!query) return 0;
      const name = normalizeText(concept.name);
      const aliases = (concept.aliases || []).map(normalizeText);
      let score = 0;
      if (name === query) score += 500;
      if (name.includes(query)) score += 260;
      if (aliases.some((alias) => alias === query)) score += 240;
      if (aliases.some((alias) => alias.includes(query))) score += 180;
      if (normalizeText(concept.group).includes(query)) score += 60;
      const brief = conceptBrief(concept);
      const detail = conceptDetail(concept);
      const qaText = conceptQa(concept).map((item) => `${item.question || ''} ${flattenRichText(item.answer)}`).join(' ');
      if (normalizeText(flattenRichText(brief.intuition)).includes(query)) score += 35;
      if (normalizeText(flattenRichText(brief.math)).includes(query)) score += 28;
      if (normalizeText(flattenRichText(brief.model)).includes(query)) score += 24;
      if (normalizeText(flattenRichText(brief.pitfall)).includes(query)) score += 12;
      if (normalizeText(flattenRichText(detail.intuition)).includes(query)) score += 18;
      if (normalizeText(flattenRichText(detail.math)).includes(query)) score += 18;
      if (normalizeText(flattenRichText(detail.model)).includes(query)) score += 16;
      if (normalizeText(flattenRichText(detail.pitfall)).includes(query)) score += 10;
      if (normalizeText(qaText).includes(query)) score += 14;
      return score;
    }

    function card(concept, index) {
      const detail = conceptDetail(concept);
      const qa = conceptQa(concept);
      const aliases = (concept.aliases || []).slice(0, 4).map((alias) => `<span>${escapeHtml(alias)}</span>`).join('');
      const qaItems = qa.map((item, qaIndex) => `<article class="concept-qa-item">
          <h4>Q${qaIndex + 1}. ${escapeHtml(item.question || '')}</h4>
          ${renderRichText(item.answer || '')}
        </article>`).join('');
      return `<article class="concept-card" id="${escapeHtml(concept.id)}" data-concept="${escapeHtml(concept.name)}" style="--card-index:${index % 12}">
        <header>
          <p>${escapeHtml(concept.group)}</p>
          <h3>${escapeHtml(concept.name)}</h3>
          <div class="concept-aliases">${aliases}</div>
        </header>
        <dl>
          <div>
            <dt>直觉解释</dt>
            <dd>${renderRichText(detail.intuition)}</dd>
          </div>
          <div>
            <dt>数学形式</dt>
            <dd>${renderRichText(detail.math)}</dd>
          </div>
          <div>
            <dt>模型中的位置</dt>
            <dd>${renderRichText(detail.model)}</dd>
          </div>
          <div>
            <dt>常见误区 / 诊断</dt>
            <dd>${renderRichText(detail.pitfall)}</dd>
          </div>
        </dl>
        <details class="concept-detail">
          <summary>经典问题与答案</summary>
          <div class="concept-detail-body">
            <div class="concept-qa-list">${qaItems}</div>
          </div>
        </details>
      </article>`;
    }

    function moduleSection(meta, groupConcepts, visibleConcepts) {
      const cards = visibleConcepts.map(card).join('');
      const empty = '<p class="concept-empty">这个模块里没有匹配的概念。可以清空搜索，或换一个关键词。</p>';
      const hasLab = meta.lab ? `<div class="module-lab-slot" data-lab-slot="${escapeHtml(meta.lab)}"></div>` : '';
      return `<section class="concept-module" id="concept-${escapeHtml(meta.slug)}" data-concept-group="${escapeHtml(meta.group)}">
        <header class="concept-module-header">
          <div>
            <p class="module-kicker">${escapeHtml(meta.group)}</p>
            <h2>${escapeHtml(meta.group)}</h2>
            <p>${escapeHtml(meta.intro)}</p>
          </div>
          <aside>
            <strong>${groupConcepts.length}</strong>
            <span>个概念</span>
          </aside>
        </header>
        <p class="module-relation">${escapeHtml(meta.relation)}</p>
        ${hasLab}
        <div class="concept-card-grid">${cards || empty}</div>
      </section>`;
    }

    function renderTocConceptLinks() {
      all('.toc-concepts[data-toc-group]').forEach((container) => {
        const group = container.getAttribute('data-toc-group') || '';
        const meta = groupByName.get(group);
        const groupConcepts = conceptRecords.filter((concept) => concept.group === group).slice(0, 8);
        container.innerHTML = groupConcepts.map((concept) => {
          return `<a href="#${escapeHtml(concept.id)}">${escapeHtml(concept.name)}</a>`;
        }).join('');
        const module = container.closest('.toc-module');
        if (module && meta) {
          module.setAttribute('data-target-section', `concept-${meta.slug}`);
        }
      });
    }

    function placeLabBlocks() {
      groupMeta.forEach((meta) => {
        if (!meta.lab) return;
        const lab = $(meta.lab);
        const slot = root.querySelector(`[data-lab-slot="${meta.lab}"]`);
        if (lab && slot && lab.parentElement !== slot) {
          slot.appendChild(lab);
        }
      });
    }

    function bindConceptDetails() {
      all('.concept-detail').forEach((details) => {
        details.addEventListener('toggle', () => {
          if (!details.open || !window.MathJax || !window.MathJax.typesetPromise) return;
          window.MathJax.typesetPromise([details]).catch(() => {});
        });
      });
    }

    function parkLabBlocks() {
      let parking = $('labParking');
      if (!parking) {
        parking = root.createElement('div');
        parking.id = 'labParking';
        parking.hidden = true;
        root.body.appendChild(parking);
      }
      groupMeta.forEach((meta) => {
        if (!meta.lab) return;
        const lab = $(meta.lab);
        if (lab && lab.parentElement !== parking) {
          parking.appendChild(lab);
        }
      });
    }

    function render(options) {
      const query = normalizeText(search.value);
      const scoredEntries = conceptRecords
        .map((concept) => ({ concept, score: searchScore(concept, query) }))
        .filter((entry) => !query || entry.score > 0 || matches(entry.concept, query))
        .sort((a, b) => {
          if (!query) return a.concept.index - b.concept.index;
          return b.score - a.score || a.concept.index - b.concept.index;
        });
      const scoreById = new Map(scoredEntries.map((entry) => [entry.concept.id, entry.score]));
      const filtered = scoredEntries.map((entry) => entry.concept);
      const activeFiltered = filtered.filter((concept) => activeGroup === '全部' || concept.group === activeGroup);
      const countPrefix = activeGroup === '全部' ? filtered.length : activeFiltered.length;
      const searchSuffix = activeGroup === '全部' ? '' : ` · 全库命中 ${filtered.length}`;
      count.textContent = `显示 ${countPrefix} / ${concepts.length} 个概念${searchSuffix}`;
      const visibleGroups = activeGroup === '全部' ? allGroups.slice() : [activeGroup];
      if (query && activeGroup === '全部') {
        visibleGroups.sort((a, b) => {
          const bestA = Math.max(-1, ...activeFiltered.filter((concept) => concept.group === a).map((concept) => scoreById.get(concept.id) || 0));
          const bestB = Math.max(-1, ...activeFiltered.filter((concept) => concept.group === b).map((concept) => scoreById.get(concept.id) || 0));
          return bestB - bestA || allGroups.indexOf(a) - allGroups.indexOf(b);
        });
      }
      parkLabBlocks();
      modules.innerHTML = visibleGroups.map((group) => {
        const meta = groupByName.get(group);
        const groupConcepts = conceptRecords.filter((concept) => concept.group === group);
        const visibleConcepts = activeFiltered
          .filter((concept) => concept.group === group)
          .sort((a, b) => {
            if (!query) return a.index - b.index;
            return (scoreById.get(b.id) || 0) - (scoreById.get(a.id) || 0) || a.index - b.index;
          });
        if (!meta || (!groupConcepts.length && !visibleConcepts.length)) return '';
        return moduleSection(meta, groupConcepts, visibleConcepts);
      }).join('') || '<p class="concept-empty">没有匹配的概念。可以换一个关键词，例如 attention、KL、DPO、置信区间。</p>';
      placeLabBlocks();
      bindConceptDetails();
      renderTaxonomy();
      renderTocConceptLinks();
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([modules]).catch(() => {});
      }
      if (options && options.scrollToGroup) {
        const meta = groupByName.get(activeGroup);
        const target = meta ? $(`concept-${meta.slug}`) : null;
        if (target) {
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    }

    search.addEventListener('input', render);
    render();
  }

  function initSoftmaxLab() {
    const bars = $('softmaxBars');
    const tempEl = $('softmaxTemp');
    const topPEl = $('softmaxTopP');
    if (!bars || !tempEl || !topPEl) return;
    bindOutput('softmaxTemp', 'softmaxTempOut', 2);
    bindOutput('softmaxTopP', 'softmaxTopPOut', 2);

    function draw() {
      const temp = Math.max(0.05, getNumber('softmaxTemp', 0.85));
      const topP = getNumber('softmaxTopP', 0.9);
      const probs = softmax(baseLogits.map((logit) => logit / temp));
      const items = tokenNames.map((name, index) => ({ name, prob: probs[index] || 0 }));
      const filtered = applyTopP(items, topP);
      const entropy = -filtered.reduce((sum, item) => {
        return item.prob > 0 ? sum + item.prob * Math.log(item.prob) : sum;
      }, 0);
      const perplexity = Math.exp(entropy);
      const maxProb = Math.max(...filtered.map((item) => item.prob), 0.001);
      bars.innerHTML = filtered.map((item) => {
        const pct = item.prob * 100;
        const width = Math.max(1, (item.prob / maxProb) * 100);
        const dim = item.kept ? '' : ' style="opacity:0.45"';
        return `<div class="prob-row"${dim}>
          <span class="prob-label">${item.name}</span>
          <span class="prob-track"><span class="prob-fill" style="width:${width.toFixed(2)}%"></span></span>
          <span class="prob-value">${pct.toFixed(1)}%</span>
        </div>`;
      }).join('');
      setText('softmaxReadout', `entropy = ${entropy.toFixed(3)} nats · perplexity = ${perplexity.toFixed(2)} · kept tokens = ${filtered.filter((item) => item.kept).length}/${filtered.length}`);
    }

    tempEl.addEventListener('input', draw);
    topPEl.addEventListener('input', draw);
    draw();
  }

  function initAttentionLab() {
    const canvas = $('attentionCanvas');
    if (!canvas) return;
    ['attnTokens', 'attnHeads', 'attnHeadDim'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('input', draw);
    });
    bindOutput('attnTokens', 'attnTokensOut', 0);
    bindOutput('attnHeads', 'attnHeadsOut', 0);
    bindOutput('attnHeadDim', 'attnHeadDimOut', 0);

    function draw() {
      const setup = setupCanvas(canvas);
      if (!setup) return;
      const { ctx, width, height } = setup;
      const colors = canvasColors();
      clear(ctx, width, height, colors);

      const tokens = Math.round(getNumber('attnTokens', 7));
      const heads = Math.round(getNumber('attnHeads', 8));
      const headDim = Math.round(getNumber('attnHeadDim', 64));
      const gridSize = Math.min(width * 0.46, height - 92);
      const cell = gridSize / tokens;
      const startX = 36;
      const startY = 58;
      text(ctx, 'single-head causal attention scores', startX, 30, colors.ink, 16, '900');

      for (let row = 0; row < tokens; row += 1) {
        for (let col = 0; col < tokens; col += 1) {
          const allowed = col <= row;
          const score = allowed ? (Math.sin((row + 1) * 1.7 + (col + 1) * 0.9) + 1) / 2 : 0;
          ctx.fillStyle = allowed ? `rgba(12, 112, 104, ${0.18 + score * 0.72})` : colors.soft;
          ctx.fillRect(startX + col * cell, startY + row * cell, Math.max(1, cell - 2), Math.max(1, cell - 2));
        }
      }

      text(ctx, 'query t', startX, startY + gridSize + 26, colors.muted, 12, '800');
      text(ctx, 'key <= t', startX + gridSize - 70, startY + gridSize + 26, colors.muted, 12, '800');

      const sideX = startX + gridSize + 42;
      const scoreCount = heads * tokens * tokens;
      const kvBytes = 2 * tokens * heads * headDim * 2;
      const qkvParams = 3 * heads * headDim * heads * headDim;
      const maxBytes = 2 * 12 * 16 * 160 * 2;
      const maxScores = 16 * 12 * 12;
      const byteRatio = Math.min(1, kvBytes / maxBytes);
      const scoreRatio = Math.min(1, scoreCount / maxScores);

      text(ctx, 'cost model', sideX, 72, colors.ink, 18, '900');
      drawMeter(ctx, sideX, 104, width - sideX - 36, 28, scoreRatio, colors.blue, 'score cells H*T*T');
      drawMeter(ctx, sideX, 158, width - sideX - 36, 28, byteRatio, colors.gold, 'KV cache bytes');
      drawMeter(ctx, sideX, 212, width - sideX - 36, 28, Math.min(1, headDim / 160), colors.violet, 'head dimension');

      text(ctx, `scores: ${scoreCount.toLocaleString()} cells`, sideX, 282, colors.ink, 14, '850');
      text(ctx, `KV: ${(kvBytes / 1024).toFixed(1)} KiB per layer`, sideX, 306, colors.ink, 14, '850');
      text(ctx, `toy QKV scale: ${qkvParams.toLocaleString()}`, sideX, 330, colors.muted, 13, '760');
      setText('attentionReadout', `T=${tokens}, H=${heads}, d_h=${headDim} · attention score cells=${scoreCount.toLocaleString()} · KV cache≈${(kvBytes / 1024).toFixed(1)} KiB/layer (bf16 toy estimate)`);
    }

    draw();
    window.addEventListener('resize', draw);
  }

  function drawMeter(ctx, x, y, width, height, ratio, color, label) {
    const colors = canvasColors();
    ctx.fillStyle = colors.soft;
    roundRect(ctx, x, y, width, height, 4);
    ctx.fill();
    ctx.fillStyle = color;
    roundRect(ctx, x, y, Math.max(3, width * clamp(ratio, 0, 1)), height, 4);
    ctx.fill();
    text(ctx, label, x, y - 8, colors.muted, 12, '800');
  }

  function initGenerationLab() {
    const canvas = $('generationCanvas');
    if (!canvas) return;
    ['genSteps', 'genGuidance'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('input', draw);
    });
    bindOutput('genSteps', 'genStepsOut', 0);
    bindOutput('genGuidance', 'genGuidanceOut', 1);

    function density(x) {
      const left = Math.exp(-0.5 * ((x + 2) / 0.55) ** 2);
      const right = Math.exp(-0.5 * ((x - 2) / 0.55) ** 2);
      return 0.5 * left + 0.5 * right;
    }

    function mapX(x, left, width) {
      return left + ((x + 4) / 8) * width;
    }

    function draw() {
      const setup = setupCanvas(canvas);
      if (!setup) return;
      const { ctx, width, height } = setup;
      const colors = canvasColors();
      clear(ctx, width, height, colors);

      const steps = Math.round(getNumber('genSteps', 14));
      const guidance = getNumber('genGuidance', 1.2);
      const left = 52;
      const top = 50;
      const plotW = width - 104;
      const plotH = height - 128;
      const baseY = top + plotH;

      text(ctx, 'toy data density and generation path', left, 28, colors.ink, 16, '900');
      line(ctx, left, baseY, left + plotW, baseY, colors.line, 1);
      line(ctx, left, top, left, baseY, colors.line, 1);

      ctx.strokeStyle = colors.blue;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i <= 240; i += 1) {
        const x = -4 + (8 * i) / 240;
        const y = baseY - density(x) * plotH * 1.55;
        if (i === 0) ctx.moveTo(mapX(x, left, plotW), y);
        else ctx.lineTo(mapX(x, left, plotW), y);
      }
      ctx.stroke();

      const targetMode = guidance > 1.7 ? 2 : -2;
      const start = -3.45;
      const points = [];
      for (let i = 0; i <= steps; i += 1) {
        const t = i / steps;
        const ease = 1 - Math.pow(1 - t, 2.1);
        const wiggle = Math.sin(t * Math.PI * 3) * (0.42 / Math.sqrt(steps));
        const x = start * (1 - ease) + targetMode * ease + wiggle;
        const y = baseY - (0.16 + 0.72 * t + 0.08 * Math.sin(i * 0.9)) * plotH;
        points.push({ x: mapX(x, left, plotW), y, rawX: x });
      }

      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      points.forEach((p, index) => {
        if (index === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.stroke();

      points.forEach((p, index) => {
        const ratio = index / Math.max(1, points.length - 1);
        ctx.fillStyle = ratio > 0.8 ? colors.red : colors.accent;
        ctx.beginPath();
        ctx.arc(p.x, p.y, index === points.length - 1 ? 5 : 3.4, 0, Math.PI * 2);
        ctx.fill();
      });

      text(ctx, 'noise', points[0].x - 22, points[0].y - 12, colors.muted, 12, '800');
      text(ctx, 'data mode', points[points.length - 1].x - 34, points[points.length - 1].y - 12, colors.red, 12, '850');
      text(ctx, 'Diffusion: repeated denoising', left, height - 52, colors.ink, 13, '850');
      text(ctx, 'Flow Matching: velocity field integration', left, height - 28, colors.ink, 13, '850');

      const discretization = 1 / Math.pow(steps, 1.25);
      const diversityPenalty = Math.max(0, guidance - 1.5) * 0.18;
      setText('generationReadout', `NFE=${steps} · toy discretization error≈${discretization.toFixed(3)} · guidance diversity penalty≈${diversityPenalty.toFixed(3)} · target mode=${targetMode > 0 ? '+2' : '-2'}`);
    }

    draw();
    window.addEventListener('resize', draw);
  }

  function initDpoLab() {
    const canvas = $('dpoCanvas');
    if (!canvas) return;
    ['dpoBeta', 'dpoMargin'].forEach((id) => {
      const el = $(id);
      if (el) el.addEventListener('input', draw);
    });
    bindOutput('dpoBeta', 'dpoBetaOut', 1);
    bindOutput('dpoMargin', 'dpoMarginOut', 1);

    function dpoLoss(beta, margin) {
      const z = beta * margin;
      if (z > 32) return Math.exp(-z);
      return Math.log1p(Math.exp(-z));
    }

    function draw() {
      const setup = setupCanvas(canvas);
      if (!setup) return;
      const { ctx, width, height } = setup;
      const colors = canvasColors();
      clear(ctx, width, height, colors);

      const beta = getNumber('dpoBeta', 1.2);
      const margin = getNumber('dpoMargin', 0.7);
      const left = 58;
      const top = 42;
      const plotW = width - 118;
      const plotH = height - 112;
      const baseY = top + plotH;
      const maxLoss = 4.2;

      text(ctx, 'DPO loss = -log sigmoid(beta * margin)', left, 25, colors.ink, 16, '900');
      line(ctx, left, baseY, left + plotW, baseY, colors.line, 1);
      line(ctx, left, top, left, baseY, colors.line, 1);
      text(ctx, 'negative margin', left, baseY + 25, colors.muted, 12, '760');
      text(ctx, 'positive margin', left + plotW - 96, baseY + 25, colors.muted, 12, '760');

      ctx.strokeStyle = colors.violet;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      for (let i = 0; i <= 240; i += 1) {
        const xMargin = -3 + (6 * i) / 240;
        const loss = Math.min(maxLoss, dpoLoss(beta, xMargin));
        const x = left + ((xMargin + 3) / 6) * plotW;
        const y = baseY - (loss / maxLoss) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const currentLoss = dpoLoss(beta, margin);
      const pointX = left + ((margin + 3) / 6) * plotW;
      const pointY = baseY - (Math.min(maxLoss, currentLoss) / maxLoss) * plotH;
      line(ctx, pointX, baseY, pointX, pointY, colors.gold, 1.5);
      ctx.fillStyle = colors.red;
      ctx.beginPath();
      ctx.arc(pointX, pointY, 5.5, 0, Math.PI * 2);
      ctx.fill();
      text(ctx, `loss ${currentLoss.toFixed(3)}`, pointX + 10, Math.max(58, pointY - 8), colors.red, 13, '850');

      const preferenceProb = 1 / (1 + Math.exp(-beta * margin));
      setText('dpoReadout', `beta=${beta.toFixed(1)} · margin=${margin.toFixed(1)} · P(winner preferred)=${preferenceProb.toFixed(3)} · DPO loss=${currentLoss.toFixed(3)}`);
    }

    draw();
    window.addEventListener('resize', draw);
  }

  function bindToc() {
    const toggle = $('tocToggle');
    const toc = $('lectureToc');
    if (toggle && toc) {
      toggle.addEventListener('click', () => {
        const open = toc.classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
      });
      toc.addEventListener('click', (event) => {
        const link = event.target.closest('a[href^="#"]');
        if (link) {
          toc.classList.remove('open');
          toggle.setAttribute('aria-expanded', 'false');
        }
      });
    }

    let ticking = false;
    function updateActiveToc() {
      const moduleLinks = all('.lecture-toc .toc-module > a[href^="#"]');
      let current = moduleLinks[0] || null;
      const anchorLine = 140;
      moduleLinks.forEach((link) => {
        const target = root.querySelector(link.getAttribute('href'));
        if (!target) return;
        const rect = target.getBoundingClientRect();
        if (rect.top <= anchorLine) {
          current = link;
        }
      });
      const currentHref = current ? current.getAttribute('href') : '';
      all('.lecture-toc a[href^="#"]').forEach((link) => {
        link.classList.toggle('active', link.getAttribute('href') === currentHref);
      });
      all('.lecture-toc .toc-module').forEach((module) => {
        const link = module.querySelector(':scope > a[href^="#"]');
        module.classList.toggle('active', Boolean(link && link.getAttribute('href') === currentHref));
      });
      ticking = false;
    }

    function requestActiveTocUpdate() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(updateActiveToc);
    }

    requestActiveTocUpdate();
    window.addEventListener('scroll', requestActiveTocUpdate, { passive: true });
    window.addEventListener('resize', requestActiveTocUpdate);
  }

  function bindProgress() {
    const bar = $('readingProgressBar');
    if (!bar) return;
    function update() {
      const max = Math.max(1, root.documentElement.scrollHeight - window.innerHeight);
      const pct = clamp((window.scrollY / max) * 100, 0, 100);
      bar.style.width = `${pct.toFixed(2)}%`;
    }
    update();
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
  }

  function init() {
    initConceptHandbook();
    initSoftmaxLab();
    initAttentionLab();
    initGenerationLab();
    initDpoLab();
    bindToc();
    bindProgress();
  }

  if (root.readyState === 'loading') {
    root.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
