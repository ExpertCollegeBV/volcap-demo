// M8 off-main-thread pack loader (module Worker). fetch -> decode (vpk1) ->
// slice (persons) / static (bg), returning transferable render arrays so the
// main thread only does the GPU upload. This is the blocker-#4 fix: the
// 16-sweep Jacobi eigendecomposition in sliceAttrs runs HERE, not on the
// render thread, so a chunk load never janks playback.
//
// Each reply carries a `t` phase-timing block {fetchMs, decodeMs, sliceMs} —
// the ADR-012 §5 "per-chunk decode wall-time" instrumentation. decodeMs is the
// pure container decode (WebP planes -> attrs for vpk2), separated from the
// slice/eig cost so the codec's own budget is visible.
import { decodePack } from "./vpk.js";
import { sliceAttrs, staticAttrs } from "./slice.js";

self.onmessage = async (e) => {
  const { id, kind, descriptor, packUrl } = e.data;
  try {
    const t0 = performance.now();
    const buf = await (await fetch(packUrl)).arrayBuffer();
    const t1 = performance.now();
    // same format dispatch as vpk.js loadScene (the worker path previously
    // only ever saw vpk1 packs; the P8 arms ship vpk2)
    const attrs =
      descriptor.format === "vpk2"
        ? await (await import("./vpk2.js")).decodePackV2(descriptor, buf)
        : decodePack(descriptor, buf);
    const t2 = performance.now();
    const r = kind === "person" ? sliceAttrs(attrs) : staticAttrs(attrs);
    const t3 = performance.now();
    const t = { fetchMs: t1 - t0, decodeMs: t2 - t1, sliceMs: t3 - t2 };
    const transfer = [];
    for (const k of Object.keys(r)) if (r[k] && r[k].buffer) transfer.push(r[k].buffer);
    self.postMessage({ id, ok: true, r, t }, transfer);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err && err.stack ? err.stack : err) });
  }
};
