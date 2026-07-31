// vpk1 loader (M7, ADR-004/008): decode native-4D packed splats in the browser.
//
// Byte-exact port of volcap/export/{container,quant,decode}.py — a plane is a
// raw little-endian array sliced from the .vpk by its scene.json descriptor
// (name/dtype/shape/offset/bytes), then dequantized. Outputs match
// decode_planes(): xyz/t linear, rotation/rotation_r unit quats, scaling and
// scaling_t in LOG space, opacity as logits, sh0 as f_dc SH coefficients.
// The parity test in ../test/vpk.test.mjs checks this against the Python decoder.

const U16 = 65535;
const U8 = 255;
const QUAT_BITS = 10;
const QUAT_CODE_MAX = (1 << QUAT_BITS) - 1; // 1023
const QUAT_COMPONENT_MAX = 1 / Math.SQRT2;
const SH_C0 = 0.28209479177387814;

const DTYPE_BYTES = { "<u2": 2, "<u1": 1, "<u4": 4, uint16: 2, uint8: 1, uint32: 4 };

function planeView(buffer, plane) {
  // plane.dtype is a numpy dtype string ("<u2" etc.); shape is [N] or [N, k]
  const [n, k] = plane.shape.length === 1 ? [plane.shape[0], 1] : plane.shape;
  const count = n * k;
  const dt = plane.dtype;
  const off = plane.offset;
  if (dt === "<u2" || dt === "uint16") return { arr: new Uint16Array(buffer, off, count), n, k };
  if (dt === "<u1" || dt === "uint8") return { arr: new Uint8Array(buffer, off, count), n, k };
  if (dt === "<u4" || dt === "uint32") return { arr: new Uint32Array(buffer, off, count), n, k };
  throw new Error(`vpk: unsupported plane dtype ${dt}`);
}

export function decodeGridU16(codes, n, k, lo, hi) {
  // per-axis uniform grid over [lo, hi]; lo/hi are length-k (or scalar)
  const out = new Float32Array(n * k);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      const l = Array.isArray(lo) ? lo[j] : lo;
      const h = Array.isArray(hi) ? hi[j] : hi;
      out[i * k + j] = l + (codes[i * k + j] / U16) * (h - l);
    }
  }
  return out;
}

export function decodeLogU8Log(codes, n, k, lo, hi) {
  // returns LOG-space values (matches decode_planes: np.log(decode_log_u8(...)))
  const logLo = Math.log(lo);
  const logSpan = Math.log(hi) - logLo;
  const out = new Float32Array(n * k);
  for (let i = 0; i < out.length; i++) out[i] = logLo + (codes[i] / U8) * logSpan;
  return out;
}

export function decodeSigmoidU8Logit(codes, n) {
  // inverse of the (2c+1)/512 center quantization -> logits
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = (2 * codes[i] + 1) / 512;
    out[i] = Math.log(a / (1 - a));
  }
  return out;
}

export function decodeColorU8(codes, n) {
  const out = new Float32Array(n * 3);
  for (let i = 0; i < out.length; i++) out[i] = (codes[i] / U8 - 0.5) / SH_C0;
  return out;
}

export function decodeQuatSmallest3(codes, n) {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const p = codes[i] >>> 0;
    const drop = p >>> 30;
    const c = [(p >>> 20) & QUAT_CODE_MAX, (p >>> 10) & QUAT_CODE_MAX, p & QUAT_CODE_MAX];
    const rest = c.map((v) => (v / QUAT_CODE_MAX) * (2 * QUAT_COMPONENT_MAX) - QUAT_COMPONENT_MAX);
    const largest = Math.sqrt(Math.max(0, 1 - (rest[0] ** 2 + rest[1] ** 2 + rest[2] ** 2)));
    const q = [0, 0, 0, 0];
    let ri = 0;
    for (let a = 0; a < 4; a++) q[a] = a === drop ? largest : rest[ri++];
    const norm = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    for (let a = 0; a < 4; a++) out[i * 4 + a] = q[a] / norm;
  }
  return out;
}

// scene.json descriptor -> decoded attribute dict for one pack.
export function decodePack(descriptor, buffer) {
  const planes = Object.fromEntries(descriptor.planes.map((p) => [p.name, p]));
  const dq = descriptor.dequant;
  const attrs = {};
  const get = (name) => planeView(buffer, planes[name]);

  const xyz = get("xyz");
  attrs.xyz = decodeGridU16(xyz.arr, xyz.n, xyz.k, dq.xyz.lo, dq.xyz.hi);
  attrs.count = xyz.n;

  if (planes.t) {
    const t = get("t");
    attrs.t = decodeGridU16(t.arr, t.n, 1, dq.t.lo, dq.t.hi);
  }
  for (const name of ["rotation", "rotation_r"]) {
    if (planes[name]) {
      const q = get(name);
      attrs[name] = decodeQuatSmallest3(q.arr, q.n);
    }
  }
  const s = get("scaling");
  attrs.scaling = decodeLogU8Log(s.arr, s.n, s.k, dq.scaling.lo, dq.scaling.hi);
  if (planes.scaling_t) {
    const st = get("scaling_t");
    attrs.scaling_t = decodeLogU8Log(st.arr, st.n, 1, dq.scaling_t.lo, dq.scaling_t.hi);
  }
  const op = get("opacity");
  attrs.opacity = decodeSigmoidU8Logit(op.arr, op.n);
  const c = get("sh0");
  attrs.sh0 = decodeColorU8(c.arr, c.n);
  return attrs;
}

// Fetch scene.json + every pack (.vpk raw planes or .vpk2 WebP atlases);
// returns {index, packs:{name:attrs}, tMapping}.
export async function loadScene(sceneJsonUrl) {
  const index = await (await fetch(sceneJsonUrl)).json();
  const base = sceneJsonUrl.slice(0, sceneJsonUrl.lastIndexOf("/") + 1);
  const packs = {};
  for (const desc of index.packs) {
    const buf = await (await fetch(base + desc.file)).arrayBuffer();
    if (desc.format === "vpk2") {
      // dynamic import: vpk2.js imports this module's kernels (no static cycle)
      const { decodePackV2 } = await import("./vpk2.js");
      packs[desc.name] = await decodePackV2(desc, buf);
    } else {
      packs[desc.name] = decodePack(desc, buf);
    }
  }
  return { index, packs, tMapping: index.t_mapping };
}
