import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

const root = document.documentElement;
const theme = root.getAttribute("data-theme") === "dark" ? "dark" : "default";

mermaid.initialize({
  startOnLoad: false,
  theme
});

const diagrams = document.querySelectorAll("code.language-mermaid");

if (diagrams.length > 0) {
  await mermaid.run({ querySelector: "code.language-mermaid" });
}
