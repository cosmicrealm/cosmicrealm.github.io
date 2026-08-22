import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SOURCE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson";
const WIDTH = 960;
const HEIGHT = 480;
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function project([longitude, latitude]) {
  const x = ((longitude + 180) / 360) * WIDTH;
  const y = ((90 - latitude) / 180) * HEIGHT;
  return [x.toFixed(1), y.toFixed(1)];
}

function ringPath(ring) {
  return ring.map((point, index) => {
    const [x, y] = project(point);
    return `${index === 0 ? "M" : "L"}${x} ${y}`;
  }).join("") + "Z";
}

function geometryPath(geometry) {
  if (geometry.type === "Polygon") {
    return geometry.coordinates.map(ringPath).join("");
  }
  if (geometry.type === "MultiPolygon") {
    return geometry.coordinates.flatMap((polygon) => polygon.map(ringPath)).join("");
  }
  return "";
}

const response = await fetch(SOURCE_URL);
if (!response.ok) throw new Error(`Natural Earth download failed: ${response.status}`);
const geojson = await response.json();
const regions = geojson.features
  .map((feature) => {
    const code = String(feature.properties.ISO_A2_EH || feature.properties.ISO_A2 || "").toUpperCase();
    if (!/^[A-Z]{2}$/.test(code)) return null;
    const shape = geometryPath(feature.geometry);
    if (!shape) return null;
    const name = String(feature.properties.ADMIN || code)
      .replaceAll("&", "&amp;")
      .replaceAll('"', "&quot;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
    return `  <path data-country-code="${code}" data-country-name="${name}" d="${shape}" />`;
  })
  .filter(Boolean)
  .sort();

const svg = `<!-- Generated from Natural Earth public-domain 1:110m country data. -->
<svg class="visitor-map" viewBox="0 0 ${WIDTH} ${HEIGHT}" role="img" aria-labelledby="visitor-map-title visitor-map-description">
  <title id="visitor-map-title">Global visitor distribution</title>
  <desc id="visitor-map-description">Countries with recorded visits are highlighted.</desc>
${regions.join("\n")}
</svg>
`;

await writeFile(path.join(root, "_includes", "world-map.svg"), svg, "utf8");
console.log(`generated ${regions.length} country regions`);
