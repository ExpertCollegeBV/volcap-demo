// M8 streaming viewer: plays a vpkclip1 multi-chunk clip end-to-end with
// bounded VRAM. Only <=2 dynamic person-chunks are GPU-resident at once (+ the
// static bg, loaded once); the next chunk is prefetched during the single-cover
// phase and evicted after its overlap. Per-splat slicing runs in a Worker
// (never janks playback). In an overlap the two chunks crossfade by scaling
// LINEAR rgba.a with a cosine partition-of-unity weight (the seam-#1 fix).
//
// window.__vc mirrors fps/mem/residency/load-time/worst-frame-gap for a headless
// driver; window.__setFrame/__play/__setCam drive it.
import * as THREE from "three";
import { SparkRenderer, SplatMesh, SparkControls, dyno } from "@sparkjsdev/spark";
import { loadClip, chunkLocalTime, crossfadeWeights } from "./clip.js";

THREE.ColorManagement.enabled = false;
const TEX_W = 2048;
const MAX_RESIDENT = 2; // dynamic person-chunks GPU-resident at once (the VRAM bound)

const hud = document.getElementById("hud");
const errEl = document.getElementById("err");
const canvas = document.getElementById("cv");

const playBtn = document.getElementById("play");
const scrub = document.getElementById("scrub");

const vc = (window.__vc = {
  status: "boot", error: null, clipUrl: null, frame: 0, frameCount: 0, chunks: 0,
  fps: 0, memMB: null, resident: 0, maxResident: 0, builtChunks: 0, playing: true,
  lastLoadMs: null, worstFrameGapMs: 0, span: 10, tier: "full", benchMs: 0,
  // ADR-012 §5 per-chunk decode wall-time: worker-side phase split of the last
  // built chunk (fetch/decode/slice summed over its person packs) + mesh build
  lastTimings: null,
});
function fail(where, e) {
  const msg = `${where}: ${e && e.stack ? e.stack : e}`;
  vc.status = "error"; vc.error = msg; errEl.textContent = msg; console.error(msg);
}
window.addEventListener("error", (e) => fail("window.error", e.error || e.message));
window.addEventListener("unhandledrejection", (e) => fail("unhandledrejection", e.reason));

// --- device tier (same calibration as p8_main): a phone cannot decode + hold
// two full-capacity chunks (~4.7M splats of buffers) — pick the lite scenes
// when the startup sort bench says the device is slow. ?tier= overrides.
function benchSortMs() {
  const n = 1_000_000;
  let a = new Uint32Array(n), b = new Uint32Array(n);
  for (let i = 0; i < n; i++) a[i] = (Math.random() * 0xffffffff) >>> 0;
  const t0 = performance.now();
  for (let shift = 0; shift < 32; shift += 8) {
    const count = new Uint32Array(257);
    for (let i = 0; i < n; i++) count[((a[i] >>> shift) & 0xff) + 1]++;
    for (let i = 0; i < 256; i++) count[i + 1] += count[i];
    for (let i = 0; i < n; i++) b[count[(a[i] >>> shift) & 0xff]++] = a[i];
    const t = a; a = b; b = t;
  }
  return performance.now() - t0;
}
function pickTier() {
  const forced = new URLSearchParams(location.search).get("tier");
  if (forced === "full" || forced === "lite") return forced;
  vc.benchMs = Math.round(benchSortMs());
  return vc.benchMs * 2.4 > 83 ? "lite" : "full";
}
function sceneUrlFor(chunk) {
  return vc.tier === "lite" && chunk.sceneLiteUrl ? chunk.sceneLiteUrl : chunk.sceneUrl;
}

// --- worker plumbing ------------------------------------------------------
// POOL, not a single worker: chunk builds dominate seam latency on mobile
// (user-measured lastLoad ~15 s), and person layers + the prefetched chunk
// decode independently — round-robin them across 2-3 workers.
let reqId = 0;
const pending = new Map();
const POOL_N = Math.min(3, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
const pool = Array.from({ length: POOL_N }, () => {
  const w = new Worker(new URL("./slice_worker.js", import.meta.url), { type: "module" });
  // a worker killed by the OS (mobile OOM) never answers — surface it instead
  // of buffering forever
  w.onerror = (e) => fail("worker", e.message || e);
  w.onmessageerror = () => fail("worker", "message deserialization failed");
  w.onmessage = (e) => {
    const { id, ok, r, t, error } = e.data;
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    ok ? p.resolve({ r, t }) : p.reject(new Error(error));
  };
  return w;
});
function sliceInWorker(kind, descriptor, packUrl) {
  const id = ++reqId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    pool[id % POOL_N].postMessage({ id, kind, descriptor, packUrl });
  });
}

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(1);
renderer.setSize(innerWidth, innerHeight, false);
const spark = new SparkRenderer({ renderer });
scene.add(spark);

function makeFloatTex(data, w, h) {
  const t = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  t.minFilter = THREE.NearestFilter; t.magFilter = THREE.NearestFilter; t.needsUpdate = true;
  return t;
}
function meshFromArrays(sl) {
  const c = new THREE.Vector3(), s = new THREE.Vector3(), q = new THREE.Quaternion(), col = new THREE.Color();
  return new SplatMesh({
    constructSplats: (sp) => {
      for (let i = 0; i < sl.count; i++) {
        c.set(sl.center[i * 3], sl.center[i * 3 + 1], sl.center[i * 3 + 2]);
        s.set(sl.scale[i * 3], sl.scale[i * 3 + 1], sl.scale[i * 3 + 2]);
        q.set(sl.quat[i * 4], sl.quat[i * 4 + 1], sl.quat[i * 4 + 2], sl.quat[i * 4 + 3]);
        col.setRGB(sl.color[i * 3], sl.color[i * 3 + 1], sl.color[i * 3 + 2]);
        sp.pushSplat(c, s, q, sl.opacity[i], col);
      }
    },
  });
}
// streaming objectModifier: time slice + per-chunk alpha (linear, the #1 fix)
function attachStream(mesh, sl, timeDyno, chunkAlphaDyno) {
  const n = sl.count, h = Math.ceil(n / TEX_W);
  const vm = new Float32Array(TEX_W * h * 4), sg = new Float32Array(TEX_W * h * 4);
  for (let i = 0; i < n; i++) {
    vm[i * 4] = sl.vel[i * 3]; vm[i * 4 + 1] = sl.vel[i * 3 + 1]; vm[i * 4 + 2] = sl.vel[i * 3 + 2]; vm[i * 4 + 3] = sl.mut[i];
    sg[i * 4] = Math.max(sl.sigT2[i], 1e-8);
  }
  const dVelMut = dyno.dynoSampler2D(makeFloatTex(vm, TEX_W, h));
  const dSigT2 = dyno.dynoSampler2D(makeFloatTex(sg, TEX_W, h));
  const step = new dyno.Dyno({
    inTypes: { gsplat: dyno.Gsplat, time: "float", chunkAlpha: "float", velMut: "sampler2D", sigT2Tex: "sampler2D" },
    outTypes: { gsplat: dyno.Gsplat },
    globals: () => [dyno.unindent(`const int TEX_W = ${TEX_W};`)],
    statements: ({ inputs, outputs }) => dyno.unindentLines(`
      ${outputs.gsplat} = ${inputs.gsplat};
      int idx = ${inputs.gsplat}.index;
      ivec2 uv = ivec2(idx % TEX_W, idx / TEX_W);
      vec4 vmv = texelFetch(${inputs.velMut}, uv, 0);
      float sigT2 = texelFetch(${inputs.sigT2Tex}, uv, 0).r;
      float dt = ${inputs.time} - vmv.w;
      ${outputs.gsplat}.center = ${inputs.gsplat}.center + vmv.xyz * dt;
      ${outputs.gsplat}.rgba.a = ${inputs.gsplat}.rgba.a * exp(-0.5 * dt * dt / max(sigT2, 1e-8)) * ${inputs.chunkAlpha};
    `),
  });
  mesh.objectModifier = dyno.dynoBlock(
    { gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat },
    ({ gsplat }) => ({ gsplat: step.apply({ gsplat, time: timeDyno, chunkAlpha: chunkAlphaDyno, velMut: dVelMut, sigT2Tex: dSigT2 }).gsplat })
  );
  mesh.updateGenerator();
}

// Residency model (2026-07-26 rework after user-measured seam stalls):
//   BUILD AHEAD  — decode/build the next 2 chunks while the current one plays.
//   PARK         — built chunks stay OUT of the scene (no sort cost) until
//                  ~1s before their fade-in; attaching early at alpha 0 lets
//                  the async sorter digest them invisibly.
//   EVICT LAZILY — dispose only when over the resident budget, farthest from
//                  the playhead first. Scrubbing back re-attaches instantly
//                  instead of rebuilding, and the screen keeps the last ready
//                  content during a jump instead of blanking.
const chunkStates = new Map(); // index -> {status, evicted, attached, meshes, timeDyno, chunkAlphaDyno}
let clip = null;
const PREFETCH_AHEAD = 2;
const RESIDENT_CAP = 4; // built chunks kept in memory (attached or parked)
const ATTACH_AHEAD_FRAMES = 24; // attach this many frames before cover starts

async function loadChunk(chunk) {
  if (chunkStates.has(chunk.index)) return;
  const st = { status: "loading", evicted: false, attached: false, meshes: [], timeDyno: dyno.dynoFloat(0), chunkAlphaDyno: dyno.dynoFloat(0) };
  chunkStates.set(chunk.index, st);
  const t0 = performance.now();
  try {
    const sceneUrl = sceneUrlFor(chunk);
    const index = await (await fetch(sceneUrl)).json();
    const base = sceneUrl.slice(0, sceneUrl.lastIndexOf("/") + 1);
    const persons = index.packs.filter((p) => p.name !== "bg");
    // fire ALL person decodes up front so the pool works them in parallel
    const jobs = persons.map((desc) => sliceInWorker("person", desc, base + desc.file));
    const agg = { fetchMs: 0, decodeMs: 0, sliceMs: 0, buildMs: 0 };
    for (const job of jobs) {
      const { r: arrays, t } = await job;
      if (st.evicted) return; // playhead moved on while we loaded
      if (t) { agg.fetchMs += t.fetchMs; agg.decodeMs += t.decodeMs; agg.sliceMs += t.sliceMs; }
      const tb0 = performance.now();
      const mesh = meshFromArrays(arrays);
      await mesh.initialized;
      attachStream(mesh, arrays, st.timeDyno, st.chunkAlphaDyno);
      agg.buildMs += performance.now() - tb0;
      if (st.evicted) { mesh.dispose?.(); return; }
      st.meshes.push(mesh); // parked: scene.add happens in the attach pass
    }
    st.status = "ready";
    vc.builtChunks++;
    vc.lastLoadMs = Math.round(performance.now() - t0);
    vc.lastTimings = {
      chunk: chunk.index,
      fetchMs: Math.round(agg.fetchMs),
      decodeMs: Math.round(agg.decodeMs),
      sliceMs: Math.round(agg.sliceMs),
      buildMs: Math.round(agg.buildMs),
    };
  } catch (e) {
    fail(`loadChunk[${chunk.index}]`, e);
  }
}
function evictChunk(index) {
  const st = chunkStates.get(index);
  if (!st) return;
  st.evicted = true;
  for (const m of st.meshes) { scene.remove(m); m.dispose?.(); }
  chunkStates.delete(index);
}
// chunks that should be RESIDENT (built or building) at global frame g:
// the covering chunk(s) + the next PREFETCH_AHEAD (modulo for looped clips)
function desiredChunks(g) {
  const cov = crossfadeWeights(clip.chunks, g).map((w) => w.chunk.index);
  const set = new Set(cov);
  const maxCov = cov.length ? Math.max(...cov) : -1;
  for (let k = 1; k <= PREFETCH_AHEAD; k++) set.add((maxCov + k) % clip.chunks.length);
  return set;
}
function coveringReady(g) {
  return crossfadeWeights(clip.chunks, g).every(
    (w) => chunkStates.get(w.chunk.index)?.status === "ready",
  );
}

window.__setFrame = (f) => { vc.frame = ((f % vc.frameCount) + vc.frameCount) % vc.frameCount; };
window.__play = (on) => { vc.playing = on; };

async function main() {
  vc.status = "loading";
  vc.tier = pickTier();
  // absolutize: pack URLs derived from this string reach the worker, which
  // resolves relative URLs against ITS OWN /src/ base, not the page
  vc.clipUrl = new URL(
    // default is PAGE-RELATIVE: deployed pages ship clip.json beside
    // index.html. The old dev-server absolute default (/runs/duo_walk/...)
    // broke every deployed page when this file was synced over the page
    // copies (2026-08-03). Dev use passes ?clip= explicitly.
    new URLSearchParams(location.search).get("clip") || "clip.json",
    location.href,
  ).href;
  hud.textContent = "loading clip…";
  try { clip = await loadClip(vc.clipUrl); } catch (e) { return fail("loadClip", e); }
  vc.frameCount = clip.frame_count; vc.chunks = clip.chunks.length; vc.span = clip.span;
  // playback cadence comes from the CLIP, not a constant: the hardcoded 24
  // played 30 fps scenes slow and 60 fps scenes at 2.5x slow motion
  // (owner report 2026-08-03: "23-25 fps" was this constant, not the device)
  vc.clipFps = Number(clip.fps) || 24;

  // camera framing: from a bounding box, honoring the clip's optional world-up
  let framed = false;
  function frameOn(box) {
    const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const radius = 0.5 * size.length();
    const dist = (radius / Math.tan((camera.fov * Math.PI) / 360)) * 1.15;
    if (clip.up) {
      // world-up hint (clip.json "up"): front-ish eye-level view instead of
      // the legacy +Z offset (a top view for Z-up worlds)
      const up = new THREE.Vector3(...clip.up).normalize();
      const ref = Math.abs(up.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      const h = new THREE.Vector3().crossVectors(up, ref).normalize();
      camera.up.copy(up);
      camera.position.copy(c).addScaledVector(h, dist * 0.92).addScaledVector(up, dist * 0.28);
    } else {
      camera.position.set(c.x + dist * 0.25, c.y + dist * 0.1, c.z + dist);
    }
    camera.lookAt(c); camera.updateProjectionMatrix();
    framed = true;
    window.__setCam = (d, azDeg = 0, elDeg = 0) => {
      const az = (azDeg * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
      camera.position.set(c.x + d * Math.cos(el) * Math.sin(az), c.y + d * Math.sin(el), c.z + d * Math.cos(el) * Math.cos(az));
      camera.lookAt(c); camera.updateProjectionMatrix();
    };
  }

  // static bg: load ONCE from chunk 0 when present, always resident
  hud.textContent = "loading bg…";
  try {
    const scene0 = sceneUrlFor(clip.chunks[0]);
    const idx0 = await (await fetch(scene0)).json();
    const base0 = scene0.slice(0, scene0.lastIndexOf("/") + 1);
    const bgDesc = idx0.packs.find((p) => p.name === "bg");
    if (bgDesc) {
      const { r: bgArrays } = await sliceInWorker("bg", bgDesc, base0 + bgDesc.file);
      const bg = meshFromArrays(bgArrays);
      await bg.initialized;
      scene.add(bg);
      frameOn(bg.getBoundingBox(true));
    }
  } catch (e) { return fail("bg", e); }

  if (!framed) {
    // fg-only clip: build chunk 0 up front and frame on the performers.
    // MUST happen before SparkControls is constructed — the controls capture
    // the camera pose at creation and re-apply their own state every update,
    // so any later reframing is silently overwritten.
    hud.textContent = "loading chunk 0…";
    await loadChunk(clip.chunks[0]);
    const st0 = chunkStates.get(0);
    if (st0 && st0.meshes.length) {
      const box = new THREE.Box3();
      for (const m of st0.meshes) box.union(m.getBoundingBox(true));
      frameOn(box);
    }
  }

  const controls = new SparkControls({ canvas });
  if (playBtn) {
    playBtn.textContent = vc.playing ? "❚❚" : "▶";
    playBtn.addEventListener("click", () => {
      vc.playing = !vc.playing;
      playBtn.textContent = vc.playing ? "❚❚" : "▶";
    });
  }
  if (scrub) {
    scrub.max = String(vc.frameCount - 1);
    scrub.addEventListener("input", () => {
      vc.playing = false;
      if (playBtn) playBtn.textContent = "▶";
      vc.frame = Number(scrub.value);
    });
  }
  window.__setFrame(0);
  let last = performance.now(), acc = 0, accN = 0, fAcc = 0, warmed = false;
  vc.status = "rendering";
  renderer.setAnimationLoop(() => {
    const now = performance.now(), dt = now - last; last = now;
    if (warmed) vc.worstFrameGapMs = Math.max(vc.worstFrameGapMs, dt);
    warmed = true;
    if (vc.playing) {
      // ADAPTIVE cadence (owner report 2026-08-03: black holes during
      // playback that heal on pause = stale depth sort). Content never
      // advances faster than the renderer's own sustained frame time (EMA),
      // which gives the async sorter at least one full render period per
      // content step — slow devices get slower-but-correct playback instead
      // of mis-sorted holes. (Spark's lastSortTime is a TIMESTAMP, not a
      // duration — the first version of this gate misread it and froze
      // playback at one frame per ~30 s.)
      vc.emaDt = vc.emaDt ? vc.emaDt * 0.9 + dt * 0.1 : dt;
      // SORT-COMPLETION gate (2026-08-03, round 2: render-rate pacing was NOT
      // enough — the async sorter takes several render frames per reorder of
      // 1.4M splats, and holes persisted). lastSortTime is a TIMESTAMP; a
      // change means a sort pass finished. Advance at most one content frame
      // per completed sort, floored at 5 fps so an idle sorter (static camera
      // heuristics, field unavailable) can never deadlock playback.
      const sortStamp = Number(spark.lastSortTime ?? 0) || 0;
      const sorted = sortStamp !== vc._prevSortStamp;
      if (sorted) {
        if (vc._prevSortStamp !== undefined && vc._sortT !== undefined) {
          const gap = now - vc._sortT;
          vc.sortHz = Math.round(10000 / Math.max(gap, 1)) / 10;
        }
        vc._prevSortStamp = sortStamp; vc._sortT = now;
      }
      const step = Math.max(1000 / (vc.clipFps || 24), vc.emaDt);
      vc.effFps = Math.round(1000 / step);
      fAcc += dt;
      if (fAcc >= step && (sorted || fAcc >= 200)) {
        fAcc = Math.min(fAcc - step, step);
        const next = (vc.frame + 1) % vc.frameCount;
        // buffering gate: never advance onto a frame whose covering chunk(s)
        // aren't GPU-ready — a free-running clock outruns the loader on slow
        // devices and the playhead then chases unbuilt chunks forever (black)
        if (!coveringReady(next)) { fAcc = 0; }
        else vc.frame = next;
      }
      if (scrub) scrub.value = String(vc.frame);
    }
    const g = vc.frame;
    // buffering is a display state, not a playback state: a paused scrub onto
    // an unbuilt chunk shows it too (the last attached content stays visible)
    vc.buffering = !coveringReady(g);
    const desired = desiredChunks(g);
    for (const idx of desired) if (!chunkStates.has(idx)) loadChunk(clip.chunks[idx]);
    // lazy eviction: only over budget, farthest-from-playhead first
    if (chunkStates.size > RESIDENT_CAP) {
      const cov = crossfadeWeights(clip.chunks, g);
      const curIdx = cov.length ? cov[0].chunk.index : 0;
      const victims = [...chunkStates.keys()]
        .filter((idx) => !desired.has(idx))
        .sort((a, b) => Math.abs(b - curIdx) - Math.abs(a - curIdx));
      while (chunkStates.size > RESIDENT_CAP && victims.length) evictChunk(victims.shift());
    }
    const wByIdx = new Map(crossfadeWeights(clip.chunks, g).map((w) => [w.chunk.index, w.weight]));
    let attached = 0;
    for (const [idx, st] of chunkStates) {
      if (st.status !== "ready") continue;
      const w = wByIdx.get(idx) || 0;
      const framesUntil = clip.chunks[idx].start_frame - g;
      const imminent = w > 0 || (framesUntil > 0 && framesUntil <= ATTACH_AHEAD_FRAMES);
      if (imminent && !st.attached) {
        for (const m of st.meshes) scene.add(m);
        st.attached = true;
      } else if (!imminent && st.attached && w === 0) {
        for (const m of st.meshes) scene.remove(m); // parked, not disposed
        st.attached = false;
      }
      if (st.attached) attached++;
      st.timeDyno.value = chunkLocalTime(clip.chunks[idx], g);
      st.chunkAlphaDyno.value = w;
      for (const m of st.meshes) m.updateVersion?.();
    }
    vc.attached = attached;
    vc.resident = chunkStates.size;
    vc.maxResident = Math.max(vc.maxResident, vc.resident);
    vc.cam = [+camera.position.x.toFixed(2), +camera.position.y.toFixed(2), +camera.position.z.toFixed(2)];
    try { controls.update(camera); renderer.render(scene, camera); }
    catch (e) { return fail("render", e); }
    acc += dt; accN++;
    if (accN >= 8) { vc.fps = Math.round(1000 / (acc / accN)); acc = 0; accN = 0; }
    if (performance.memory) vc.memMB = Math.round(performance.memory.usedJSHeapSize / 1e6);
    hud.textContent =
      `clip ${vc.chunks} chunks  frame ${vc.frame}/${vc.frameCount}  ` +
      `${vc.buffering ? "buffering…" : vc.playing ? "▶" : "❚❚"}\n` +
      `resident ${vc.resident} (attached ${vc.attached ?? 0})  built ${vc.builtChunks}  lastLoad ${vc.lastLoadMs} ms\n` +
      `fps ${vc.fps}  jsHeap ${vc.memMB} MB  worstFrameGap ${Math.round(vc.worstFrameGapMs)} ms  ` +
      `tier ${vc.tier}${vc.benchMs ? ` (bench ${vc.benchMs}ms)` : ""}  ` +
      `play ${vc.effFps ?? "-"}/${vc.clipFps ?? "?"} fps  frameEMA ${vc.emaDt ? Math.round(vc.emaDt) : "-"} ms  sorts ${vc.sortHz ?? "?"}/s`;
  });
}
main().catch((e) => fail("main", e));
