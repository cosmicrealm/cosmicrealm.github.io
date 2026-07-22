"use strict";

const assert = require("node:assert/strict");
const mechanics = require("../foundations/llm-mechanics/static/js/index.js");


function approx(actual, expected, tolerance = 1e-9) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}


assert.equal(typeof mechanics.bpeMergeStep, "function");

const bpe = mechanics.bpeMergeStep(["l", "o", "w", " ", "l", "o", "w"]);
assert.deepEqual(bpe.pair, ["l", "o"]);
assert.equal(bpe.count, 2);
assert.deepEqual(bpe.tokens, ["lo", "w", " ", "lo", "w"]);

assert.equal(mechanics.causalVisible(3, 3), true);
assert.equal(mechanics.causalVisible(3, 4), false);

const rotated = mechanics.rotate2D([3, 4], Math.PI / 2);
approx(Math.hypot(...rotated), 5);
approx(rotated[0], -4);
approx(rotated[1], 3);

const normalized = mechanics.rmsNormalize([1, 2, 3, 4]);
approx(Math.sqrt(normalized.reduce((sum, value) => sum + value * value, 0) / normalized.length), 1, 1e-6);
assert.deepEqual(mechanics.rmsNormalize([0, 0]), [0, 0]);

const work = mechanics.prefillDecodeWork(8, 5);
assert.equal(work.naive, 50);
assert.equal(work.cached, 12);

const kv = mechanics.kvCacheMetrics({ layers: 32, kvHeads: 8, headDim: 128, dtypeBytes: 2, batch: 1, context: 4096 });
assert.equal(kv.bytesPerToken, 131072);
assert.equal(kv.totalBytes, 536870912);

const cool = mechanics.softmax([2.5, 1.7, 0.4], 0.5);
const warm = mechanics.softmax([2.5, 1.7, 0.4], 1.5);
assert.ok(Math.max(...cool) > Math.max(...warm));
approx(cool.reduce((sum, value) => sum + value, 0), 1);

const candidates = mechanics.topPIndices([0.55, 0.25, 0.15, 0.05], 0.75);
assert.deepEqual(candidates, [0, 1]);
assert.deepEqual(mechanics.topPIndices([0.7, 0.2, 0.1], 0.1), [0]);

assert.deepEqual(mechanics.applyRepetitionPenalty([-1, 1, 3], new Set([0, 2]), 2), [-2, 1, 1.5]);
assert.equal(mechanics.formatBytes(536870912), "512.00 MiB");

console.log("LLM mechanics JavaScript unit checks passed");
