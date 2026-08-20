(function () {
  var root = document.documentElement;
  var stored = null;
  var setting = "system";
  try {
    stored = localStorage.getItem("theme");
  } catch (error) {
    stored = null;
  }
  if (stored === "light" || stored === "dark") setting = stored;
  var systemDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var resolved = setting === "dark" || (setting === "system" && systemDark) ? "dark" : "light";
  root.setAttribute("data-theme", resolved);
  root.setAttribute("data-theme-setting", setting);
}());
