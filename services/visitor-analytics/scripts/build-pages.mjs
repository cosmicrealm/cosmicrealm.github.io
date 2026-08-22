import { copyFile, mkdir } from "node:fs/promises";

await mkdir(new URL("../dist/", import.meta.url), { recursive: true });
await copyFile(
  new URL("../src/index.mjs", import.meta.url),
  new URL("../dist/_worker.js", import.meta.url),
);
