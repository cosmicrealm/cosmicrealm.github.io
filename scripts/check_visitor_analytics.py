import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STANDALONE_PAGES = [
    "foundations/aigc-llm-math/index.html",
    "foundations/generation-acceleration/index.html",
    "foundations/generation-distillation/index.html",
    "foundations/generation-math/index.html",
    "foundations/image-generation-data-training/index.html",
    "foundations/leetcode-hot100/index.html",
    "foundations/llm-interview-qa/index.html",
    "foundations/llm-mechanics/index.html",
    "foundations/video-generation/index.html",
    "projects/iconface/index.html",
    "projects/style-talking/index.html",
]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


home = read("_pages/home.md")
visitor_component = read("_includes/home-visitor-stats.html")
scripts = read("_includes/scripts.html")
config = read("_config.yml")
styles = read("_sass/layout/_home.scss")
theme_tokens = read("assets/css/theme-tokens.css")

writing_position = home.index("Recent Writing")
visitor_position = home.index("{% include home-visitor-stats.html %}")
require(visitor_position > writing_position, "Global Reach must render after Recent Writing")
require("Global Reach" in visitor_component, "homepage is missing the Global Reach heading")
require("data-visitor-map-legend" in visitor_component, "visitor map is missing its color scale legend")
for attribute in [
    "data-visitor-total",
    "data-visitor-today",
    "data-visitor-countries",
    "data-visitor-status",
    "data-visitor-accessible-summary",
]:
    require(attribute in visitor_component, f"homepage is missing {attribute}")

require("visitor-analytics.js" in scripts, "Jekyll pages do not load visitor-analytics.js")
require(
    scripts.index("visitor-analytics.js") < scripts.index("site.js"),
    "visitor-analytics.js must load before site.js",
)
for page in STANDALONE_PAGES:
    source = read(page)
    require("visitor-analytics.js" in source, f"{page} does not load visitor analytics")
    require(
        source.index("visitor-analytics.js") < source.index("site.js"),
        f"{page} must load visitor analytics before site.js",
    )

require("services/" in config, "Jekyll must exclude the Cloudflare service workspace")
require(".visitor-reach" in styles, "homepage styles are missing the Global Reach component")
require("[data-theme=\"dark\"]" not in styles, "Global Reach must use shared theme tokens")
require("var(--cr-page-bg)" in styles, "Global Reach must inherit the canonical page background")
require("--visitor-map-saturation" in theme_tokens, "theme tokens are missing visitor map saturation")
require("--visitor-map-lightness" in theme_tokens, "theme tokens are missing visitor map lightness")
require("hsl(var(--visitor-hue" in styles, "visited countries must use the volume hue scale")

world_map = read("_includes/world-map.svg")
country_codes = set(re.findall(r'data-country-code="([A-Z]{2})"', world_map))
require(len(country_codes) >= 160, "world map must expose at least 160 country regions")

worker_source = read("services/visitor-analytics/src/index.mjs")
schema = read("services/visitor-analytics/migrations/0001_initial.sql")
require("CF-Connecting-IP" in worker_source, "Worker must derive the ephemeral visitor key at the edge")
require("VISITOR_HASH_SECRET" in worker_source, "Worker must require a server-side HMAC secret")
require("raw_ip" not in schema.lower() and "ip_address" not in schema.lower(), "D1 schema must not store raw IP addresses")
require("visitor_hash" in schema, "D1 schema is missing the rotating visitor hash")

print("visitor analytics structure: PASS")
