# LLM Mechanics Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-first Chinese Foundations lecture that teaches 12 LLM mechanisms through five linked chapters, six browser labs, and twelve runnable examples.

**Architecture:** A standalone Jekyll-compatible HTML lecture owns its local CSS, JavaScript, MathJax vendor copy, and downloadable Python examples under `foundations/llm-mechanics/`. A Python verifier treats the page structure, source boundaries, labs, examples, accessibility hooks, and forbidden audience labels as a testable contract; `run_all.py` separately smoke-tests every example.

**Tech Stack:** Static HTML5, CSS, vanilla JavaScript, local MathJax, Python 3.10+, PyTorch 2.x, Jekyll/Docker, Playwright browser QA.

---

### Task 1: Add the failing content contract

**Files:**
- Create: `scripts/verify_llm_mechanics.py`
- Test: `foundations/llm-mechanics/index.html`
- Test: `foundations/llm-mechanics/examples/`

- [ ] **Step 1: Write the verifier before the page exists**

The verifier must parse the HTML with the standard library and fail unless it finds exactly five chapter ids, six lab ids, twelve example links, Stanford/nanochat source anchors, accessibility labels, local assets, and no explicit audience-tier labels.

```python
CHAPTERS = ["tokens", "decoder", "training", "inference", "decoding"]
LABS = ["bpe", "decoder", "teacher", "prefill", "kv", "sampling"]
EXAMPLES = [f"{i:02d}_" for i in range(1, 13)]

for chapter in CHAPTERS:
    require(f'id="{chapter}"' in html, f"missing chapter: {chapter}")
for lab in LABS:
    require(f'id="lab-{lab}"' in html, f"missing lab: {lab}")
for prefix in EXAMPLES:
    require(any(path.name.startswith(prefix) for path in examples), f"missing example: {prefix}")
```

- [ ] **Step 2: Run the verifier and observe the expected failure**

Run: `python3 scripts/verify_llm_mechanics.py`

Expected: non-zero exit with `missing page: foundations/llm-mechanics/index.html`.

- [ ] **Step 3: Commit the executable contract**

```bash
git add scripts/verify_llm_mechanics.py
git commit -m "test: define LLM mechanics page contract"
```

### Task 2: Implement and test the twelve runnable examples

**Files:**
- Create: `foundations/llm-mechanics/examples/README.md`
- Create: `foundations/llm-mechanics/examples/requirements.txt`
- Create: `foundations/llm-mechanics/examples/run_all.py`
- Create: `foundations/llm-mechanics/examples/01_bpe.py`
- Create: `foundations/llm-mechanics/examples/02_decoder_only.py`
- Create: `foundations/llm-mechanics/examples/03_causal_mask.py`
- Create: `foundations/llm-mechanics/examples/04_rope.py`
- Create: `foundations/llm-mechanics/examples/05_rmsnorm.py`
- Create: `foundations/llm-mechanics/examples/06_swiglu.py`
- Create: `foundations/llm-mechanics/examples/07_prefill_decode.py`
- Create: `foundations/llm-mechanics/examples/08_kv_cache.py`
- Create: `foundations/llm-mechanics/examples/09_sampling.py`
- Create: `foundations/llm-mechanics/examples/10_repetition_eos.py`
- Create: `foundations/llm-mechanics/examples/11_teacher_forcing.py`
- Create: `foundations/llm-mechanics/examples/12_perplexity.py`

- [ ] **Step 1: Add a runner that discovers every numbered example**

```python
example_files = sorted(ROOT.glob("[0-9][0-9]_*.py"))
assert len(example_files) == 12
for path in example_files:
    completed = subprocess.run([sys.executable, str(path)], text=True, capture_output=True)
    if completed.returncode:
        failures.append((path.name, completed.stderr))
```

- [ ] **Step 2: Add pure-Python examples for BPE, sampling, stopping, and perplexity**

Each file uses fixed inputs and assertions. `01_bpe.py` learns and applies merge rules; `09_sampling.py` implements stable softmax, greedy, temperature, and nucleus filtering; `10_repetition_eos.py` applies a sign-aware repetition penalty and stops at EOS; `12_perplexity.py` computes `exp(mean NLL)` and demonstrates why tokenization changes the numerical result.

```python
def nucleus(probs, p):
    ranked = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
    kept, cumulative = [], 0.0
    for item in ranked:
        kept.append(item)
        cumulative += item[1]
        if cumulative >= p:
            break
    return kept
```

- [ ] **Step 3: Add PyTorch tensor examples for the model and training mechanisms**

Use deterministic tensors and assert these properties: decoder output shape; future attention weights are zero; RoPE preserves vector norm; RMSNorm produces unit RMS before scale; SwiGLU output shape; cached and uncached final-token outputs agree; KV cache shape grows along sequence; teacher-forcing targets are shifted by one.

```python
future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
attention = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
assert torch.all(attention[future] == 0)
```

- [ ] **Step 4: Run all examples**

Run: `python3 foundations/llm-mechanics/examples/run_all.py`

Expected: `12/12 examples passed`. If PyTorch is absent, create a temporary virtual environment, install `requirements.txt`, and rerun without changing the system Python.

- [ ] **Step 5: Commit the example suite**

```bash
git add foundations/llm-mechanics/examples
git commit -m "feat: add runnable LLM mechanism examples"
```

### Task 3: Build the five-chapter source-first lecture

**Files:**
- Create: `foundations/llm-mechanics/index.html`
- Copy: `foundations/generation-math/static/vendor/mathjax/` → `foundations/llm-mechanics/static/vendor/mathjax/`
- Modify: `_data/foundations.yml`

- [ ] **Step 1: Add the complete document shell and local MathJax**

Use a standalone Chinese HTML document with SEO metadata, canonical URL, theme initialization, skip link, reading progress, site header, mobile TOC button, fixed desktop TOC, `.lecture`, `.paper-header`, five `.chapter` sections, footer, local CSS, local JavaScript, and local MathJax.

```html
<link rel="stylesheet" href="./static/css/index.css">
<script defer src="./static/vendor/mathjax/tex-mml-chtml.js" id="MathJax-script"></script>
<aside class="lecture-toc" id="lectureToc">…</aside>
<main id="main" class="lecture">…</main>
<script src="./static/js/index.js" defer></script>
```

- [ ] **Step 2: Write all five chapters and source-boundary notes**

Every chapter includes a thesis, mechanism derivation, shape trace, algorithm panel, experiment container, minimal-example card, misconception/failure section, CS336 mapping, nanochat mapping, and checkpoints. Use commit-pinned nanochat links and official CS336 links. Explicitly state the verified absences around teacher-forcing terminology and repetition penalty.

- [ ] **Step 3: Add the Foundations index entry**

Insert one new featured entry near the current LLM Foundations items:

```yaml
- name: LLM 核心机制交互讲义
  category: LLM 原理与系统
  status: 12 个机制与 6 个动态实验
  featured: true
  date: 2026-07-22
  display_date: 07/2026
  url: /foundations/llm-mechanics/
```

- [ ] **Step 4: Run the verifier**

Run: `python3 scripts/verify_llm_mechanics.py`

Expected: structure checks pass except CSS/JS behavior checks that are introduced by subsequent tasks.

- [ ] **Step 5: Commit content and registration**

```bash
git add _data/foundations.yml foundations/llm-mechanics/index.html foundations/llm-mechanics/static/vendor
git commit -m "feat: add LLM mechanics lecture content"
```

### Task 4: Implement the responsive lecture visual system

**Files:**
- Create: `foundations/llm-mechanics/static/css/index.css`

- [ ] **Step 1: Implement page tokens and typography**

Define light/dark CSS variables for paper, sheet, ink, muted text, line, green, vermilion, gold, and lab surfaces. Style `.paper-header`, `.chapter`, `.chapter-header`, equations, shape traces, source notes, algorithm panels, code/example panels, misconceptions, checkpoints, tables, and footer.

- [ ] **Step 2: Implement TOC gutter and mobile collapse**

At desktop widths, reserve a real content gutter and fix the TOC. At `max-width: 1100px`, move TOC off-canvas and expose `.toc-toggle`. At `max-width: 720px`, collapse multi-column grids and make wide tables horizontally scroll inside their own wrappers.

```css
@media (min-width: 1101px) {
  .lecture { margin-left: max(18rem, calc((100vw - 82rem) / 2 + 14rem)); }
  .lecture-toc { position: fixed; left: max(1rem, calc((100vw - 92rem) / 2)); width: 13rem; }
}
```

- [ ] **Step 3: Style labs as functional work surfaces**

Controls, visual output, readout, explanation, reset, and error message must have stable layout and visible focus. Use semantic color plus text, never color alone.

- [ ] **Step 4: Run static checks and commit**

Run: `python3 scripts/verify_llm_mechanics.py && git diff --check`

Expected: both pass.

```bash
git add foundations/llm-mechanics/static/css/index.css
git commit -m "feat: style LLM mechanics interactive lecture"
```

### Task 5: Implement navigation, code controls, and labs 1–3

**Files:**
- Create: `foundations/llm-mechanics/static/js/index.js`

- [ ] **Step 1: Add fault-isolated initialization**

```javascript
function safeInit(name, init) {
  try { init(); }
  catch (error) {
    console.error(`[llm-mechanics:${name}]`, error);
    document.querySelector(`[data-lab="${name}"] .lab-error`)?.removeAttribute("hidden");
  }
}
```

Initialize progress, TOC, copy buttons, BPE lab, decoder lab, teacher-forcing lab, prefill lab, KV lab, and sampling lab independently.

- [ ] **Step 2: Add progress, scrollspy, mobile TOC, and copy fallback**

Use `IntersectionObserver` for active chapter, scroll ratio for the progress bar, `aria-expanded` for the mobile TOC, and Clipboard API with a selection fallback.

- [ ] **Step 3: Implement BPE merge lab**

Tokenize user text to Unicode-safe characters for the browser toy model, compute deterministic pair counts, merge the highest-frequency pair, and update tokens, rules, compression, and explanation. Reset reconstructs the original state.

- [ ] **Step 4: Implement decoder anatomy lab**

Support four tabs. Causal mode changes the visible matrix by query position; RoPE mode rotates a 2D vector by position and reports norm; RMSNorm mode rescales an editable vector; SwiGLU mode computes `SiLU(gate) * value` and compares it to nanochat's ReLU² note.

- [ ] **Step 5: Implement teacher-forcing/perplexity lab**

Use a fixed token sequence and probability table. The position control highlights input/target alignment and recomputes NLL/PPL; the faulty toggle intentionally compares unshifted targets and explains leakage/misalignment.

- [ ] **Step 6: Run syntax/static checks and commit**

Run: `node --check foundations/llm-mechanics/static/js/index.js && python3 scripts/verify_llm_mechanics.py`

Expected: both pass.

```bash
git add foundations/llm-mechanics/static/js/index.js
git commit -m "feat: add core LLM mechanism interactions"
```

### Task 6: Implement labs 4–6 and responsive drawing

**Files:**
- Modify: `foundations/llm-mechanics/static/js/index.js`

- [ ] **Step 1: Implement prefill/decode timeline**

Prompt and generated-length controls update the timeline, parallelism markers, naive token-work estimate, cached token-work estimate, and explanatory text.

- [ ] **Step 2: Implement KV cache calculator**

Clamp all numeric inputs and compute:

```javascript
bytesPerToken = layers * 2 * kvHeads * headDim * dtypeBytes;
totalBytes = batch * context * bytesPerToken;
```

Report MiB/GiB, cache shape, and cached-versus-uncached attention work. Zero or invalid values display a local validation message.

- [ ] **Step 3: Implement sampling and stopping lab**

Use stable softmax, greedy, seeded categorical sampling, top-p filtering that always keeps one token, sign-aware repetition penalty, and EOS/max-token state. Render probability bars with numeric labels and candidate-set membership.

- [ ] **Step 4: Add resize-safe visual updates**

Debounce resize, redraw canvas/SVG-dependent visuals, and scale canvases by `devicePixelRatio` without changing CSS dimensions.

- [ ] **Step 5: Run syntax/static checks and commit**

Run: `node --check foundations/llm-mechanics/static/js/index.js && python3 scripts/verify_llm_mechanics.py`

Expected: both pass.

```bash
git add foundations/llm-mechanics/static/js/index.js
git commit -m "feat: add inference and sampling labs"
```

### Task 7: Build, browser-test, and finish

**Files:**
- Modify: `scripts/verify_llm_mechanics.py`
- Modify only if QA finds defects: `foundations/llm-mechanics/index.html`
- Modify only if QA finds defects: `foundations/llm-mechanics/static/css/index.css`
- Modify only if QA finds defects: `foundations/llm-mechanics/static/js/index.js`

- [ ] **Step 1: Run the complete local verification set**

```bash
python3 scripts/verify_llm_mechanics.py
python3 foundations/llm-mechanics/examples/run_all.py
node --check foundations/llm-mechanics/static/js/index.js
git diff --check
```

Expected: all checks pass and examples report `12/12 examples passed`.

- [ ] **Step 2: Build the Jekyll site**

Run the repository's available Docker Compose command and verify `/foundations/llm-mechanics/` returns HTTP 200. If Docker is unavailable, use the existing local Jekyll bundle and report that fallback explicitly.

- [ ] **Step 3: Run browser evidence QA**

At desktop and mobile widths, inspect: first screen, all five chapters, six lab initial states, changed control states, mobile TOC, keyboard focus, source links, copy feedback, and console. Measure and assert `toc.right <= lecture.left`, `scrollWidth <= clientWidth`, and no page errors.

- [ ] **Step 4: Fix only observed defects and rerun the narrow checks**

Every fix must correspond to a captured browser or verification failure. Rerun the failing check plus the complete static verifier.

- [ ] **Step 5: Commit the verified result**

```bash
git add _data/foundations.yml foundations/llm-mechanics scripts/verify_llm_mechanics.py
git commit -m "feat: publish interactive LLM mechanics foundation"
```

- [ ] **Step 6: Report completion evidence**

Report exact commands, passed checks, changed files, browser URLs/screenshots, expensive checks not run, known limits, the pinned CS336/nanochat source versions, and confirmation that unrelated dirty files were untouched.
