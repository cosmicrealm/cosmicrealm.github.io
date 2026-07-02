(function () {
  const root = document;

  function init() {
    const bar = root.getElementById('readingProgressBar');
    const toc = root.getElementById('lectureToc');
    const toggle = root.getElementById('tocToggle');
    const links = Array.from(root.querySelectorAll('.lecture-toc a[href^="#"]'));

    function updateProgress() {
      if (!bar) return;
      const max = Math.max(1, root.documentElement.scrollHeight - window.innerHeight);
      const progress = Math.max(0, Math.min(1, window.scrollY / max));
      bar.style.width = (progress * 100).toFixed(2) + '%';
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    window.addEventListener('resize', updateProgress);
    updateProgress();

    if (toggle && toc) {
      toggle.addEventListener('click', function () {
        const open = toc.classList.toggle('open');
        toggle.setAttribute('aria-expanded', String(open));
      });
      links.forEach(function (link) {
        link.addEventListener('click', function () {
          toc.classList.remove('open');
          toggle.setAttribute('aria-expanded', 'false');
        });
      });
    }

    if ('IntersectionObserver' in window && links.length) {
      const linkMap = new Map();
      links.forEach(function (link) {
        linkMap.set(link.getAttribute('href').slice(1), link);
      });
      const observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (link) { link.classList.remove('active'); });
          const active = linkMap.get(entry.target.id);
          if (active) active.classList.add('active');
        });
      }, { rootMargin: '-18% 0px -68% 0px', threshold: 0.01 });
      root.querySelectorAll('section[id]').forEach(function (section) {
        observer.observe(section);
      });
    }
  }

  if (root.readyState === 'loading') {
    root.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
