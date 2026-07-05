(function () {
  const root = document;
  const $ = (id) => root.getElementById(id);
  const $$ = (selector, scope) => Array.from((scope || root).querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function updateProgress() {
    const bar = $("readingProgressBar");
    if (!bar) return;
    const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    const pct = Math.min(100, Math.max(0, (window.scrollY / max) * 100));
    bar.style.width = pct + "%";

    const top = $("backToTop");
    if (top) top.classList.toggle("visible", window.scrollY > 520);
  }

  function setupToc() {
    const toc = $("lectureToc");
    const toggle = $("tocToggle");
    if (toggle && toc) {
      toggle.addEventListener("click", () => {
        const open = toc.classList.toggle("open");
        toggle.setAttribute("aria-expanded", String(open));
      });
      toc.addEventListener("click", (event) => {
        if (event.target.closest("a")) {
          toc.classList.remove("open");
          toggle.setAttribute("aria-expanded", "false");
        }
      });
    }

    const links = $$(".lecture-toc a[href^=\"#\"]");
    const sections = links.map((link) => root.querySelector(link.getAttribute("href"))).filter(Boolean);
    if (!sections.length) return;
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach((link) => {
        link.classList.toggle("active", link.getAttribute("href") === "#" + visible.target.id);
      });
    }, { rootMargin: "-20% 0px -65% 0px", threshold: [0.05, 0.2, 0.5] });
    sections.forEach((section) => observer.observe(section));
  }

  function renderList(items) {
    return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }

  function renderFormulas(item) {
    if (!Array.isArray(item.formulas) || !item.formulas.length) return "";
    return `<div class="formula-stack">${item.formulas.map((formula) => `<div class="equation">${escapeHtml(formula)}</div>`).join("")}</div>`;
  }

  function renderFollowUps(item) {
    return `<div class="followup-list">${item.followUps.map((entry) => `
      <article>
        <h4>${escapeHtml(entry.q)}</h4>
        <p>${escapeHtml(entry.a)}</p>
      </article>
    `).join("")}</div>`;
  }

  function renderQuestion(item) {
    return `<article class="answer-note" id="${escapeHtml(item.id)}">
      <header class="answer-header">
        <p class="qa-meta"><span>Q${item.number}</span><span>${escapeHtml(item.section)}</span></p>
        <h3>${escapeHtml(item.question)}</h3>
      </header>
      <section class="answer-block lead-answer">
        <h4>速答</h4>
        <p>${escapeHtml(item.oneLiner)}</p>
        <p>${escapeHtml(item.oralAnswer)}</p>
      </section>
      <section class="answer-block">
        <h4>公式与机制</h4>
        ${renderFormulas(item)}
      </section>
      <section class="answer-block">
        <h4>答案解析</h4>
        ${item.deepDive.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("")}
      </section>
      <section class="answer-block">
        <h4>工程视角</h4>
        ${renderList(item.engineering)}
      </section>
      <section class="answer-block">
        <h4>常见追问</h4>
        ${renderFollowUps(item)}
      </section>
      <section class="answer-block">
        <h4>易错点与修订</h4>
        ${renderList(item.pitfalls)}
      </section>
      <p class="memory-hook">${escapeHtml(item.memoryHook)}</p>
    </article>`;
  }

  function renderSection(section, items) {
    return `<section class="chapter answer-section" id="section-${escapeHtml(section.id)}">
      <header class="chapter-header">
        <h2>${escapeHtml(section.title)}</h2>
        <p class="chapter-summary">${escapeHtml(section.range)} · ${items.length} 题</p>
      </header>
      <div class="answer-list">${items.map(renderQuestion).join("")}</div>
    </section>`;
  }

  function renderPage() {
    const data = window.LLM_INTERVIEW_QA_DATA || [];
    const sections = window.LLM_INTERVIEW_SECTIONS || [];
    const mount = $("sectionMount");
    if (!mount || !data.length) return;
    mount.innerHTML = sections.map((section) => {
      const items = data.filter((item) => item.sectionId === section.id);
      return renderSection(section, items);
    }).join("");
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([mount]).catch(() => {});
    }
  }

  function setupBackToTop() {
    $("backToTop")?.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  function boot() {
    setupToc();
    renderPage();
    setupBackToTop();
    updateProgress();
  }

  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);
  root.addEventListener("DOMContentLoaded", boot);
})();
