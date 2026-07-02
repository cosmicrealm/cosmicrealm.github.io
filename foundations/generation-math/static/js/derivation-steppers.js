(function () {
  function init() {
    document.querySelectorAll('details.advanced-derivation').forEach(function (details) {
      details.addEventListener('toggle', function () {
        if (details.open && window.MathJax && window.MathJax.typesetPromise) {
          window.MathJax.typesetPromise([details]).catch(function () {});
        }
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
