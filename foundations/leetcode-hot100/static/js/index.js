(function () {
  const root = document;
  const $ = (id) => root.getElementById(id);
  const $$ = (selector) => Array.from(root.querySelectorAll(selector));

  function updateProgress() {
    const bar = $("readingProgressBar");
    if (!bar) return;
    const max = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
    const pct = Math.min(100, Math.max(0, (window.scrollY / max) * 100));
    bar.style.width = pct + "%";
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

  function setupFilters() {
    const search = $("problemSearch");
    const group = $("groupFilter");
    const difficulty = $("difficultyFilter");
    const result = $("resultCount");
    const cards = $$("[data-problem-card]");
    const sections = $$("[data-group-section]");

    function apply() {
      const query = (search && search.value || "").trim().toLowerCase();
      const groupValue = group ? group.value : "all";
      const difficultyValue = difficulty ? difficulty.value : "all";
      let shown = 0;

      cards.forEach((card) => {
        const text = (card.dataset.title || "").toLowerCase();
        const okSearch = !query || text.includes(query);
        const okGroup = groupValue === "all" || card.dataset.group === groupValue;
        const okDifficulty = difficultyValue === "all" || card.dataset.difficulty === difficultyValue;
        const visible = okSearch && okGroup && okDifficulty;
        card.hidden = !visible;
        if (visible) shown += 1;
      });

      sections.forEach((section) => {
        section.hidden = !section.querySelector("[data-problem-card]:not([hidden])");
      });

      if (result) {
        result.textContent = "当前显示 " + shown + " / " + cards.length + " 题";
      }
    }

    [search, group, difficulty].forEach((control) => {
      if (control) control.addEventListener("input", apply);
    });
    $("expandVisible")?.addEventListener("click", () => {
      cards.forEach((card) => {
        if (!card.hidden) card.open = true;
      });
    });
    $("collapseAll")?.addEventListener("click", () => {
      cards.forEach((card) => {
        card.open = false;
      });
    });
    apply();
  }

  function setupCopyButtons() {
    $$("[data-copy-code]").forEach((button) => {
      button.addEventListener("click", async () => {
        const block = button.closest(".code-block");
        const code = block ? block.querySelector("code") : null;
        if (!code) return;
        const text = code.textContent || "";
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = "已复制";
          setTimeout(() => {
            button.textContent = "复制";
          }, 1200);
        } catch (error) {
          const range = document.createRange();
          range.selectNodeContents(code);
          const selection = window.getSelection();
          selection.removeAllRanges();
          selection.addRange(range);
          button.textContent = "已选中";
          setTimeout(() => {
            button.textContent = "复制";
          }, 1200);
        }
      });
    });
  }

  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);
  root.addEventListener("DOMContentLoaded", () => {
    setupToc();
    setupFilters();
    setupCopyButtons();
    updateProgress();
  });
})();
