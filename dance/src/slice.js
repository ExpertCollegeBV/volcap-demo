// 4C4D native-4D gaussian -> 3D conditional slice.
//
// Exact port of third_party/4C4D/utils/general_utils.py (build_rotation_4d,
// build_scaling_rotation_4d) + scene/gaussian_model.py
// (build_covariance_from_scaling_rotation_4d, get_cov_t, get_marginal_t).
//
// The 3D conditional covariance Σ_cond and the velocity v = Σ_xt/σ_t² are
// TIME-INVARIANT, so we precompute them once on the CPU (Σ_cond ->
// scale/quaternion via a symmetric-3x3 eigendecomposition, which is what a
// splat renderer consumes). Playback then only needs, per frame:
//     center(t)  = μ + v·(t − μ_t)
//     opacity(t) = base_opacity · exp(−½ (t − μ_t)² / σ_t²)
// which the GPU objectModifier applies from the per-splat (v, μ_t, σ_t²).
//
// Quaternions are stored (w,x,y,z) — matching build_rotation's q[:,0]=r=w.
// Parity-checked against the vendored torch build_rotation_4d in
// ../test/slice.test.mjs.

const SH_C0 = 0.28209479177387814;

// 4C4D two-quaternion SO(4) rotation: R = flip( M_l(q_l) @ M_r(q_r) ).
export function rot4d(ql, qr) {
  let [a, b, c, d] = ql, [p, q, r, s] = qr;
  const ln = Math.hypot(a, b, c, d) || 1, rn = Math.hypot(p, q, r, s) || 1;
  a /= ln; b /= ln; c /= ln; d /= ln;
  p /= rn; q /= rn; r /= rn; s /= rn;
  // left/right isoclinic multiplication matrices (row-major 4x4)
  const Ml = [[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]];
  const Mr = [[p, q, r, s], [-q, p, -s, r], [-r, s, p, -q], [-s, -r, q, p]];
  const A = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += Ml[i][k] * Mr[k][j];
      A[i][j] = sum;
    }
  // torch .flip(1,2): reverse BOTH axes -> R[i][j] = A[3-i][3-j]
  const R = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) R[i][j] = A[3 - i][3 - j];
  return R;
}

// Full 4D covariance C = R diag(s²) Rᵀ, then the conditional 3D slice.
// s = [sx, sy, sz, st] (linear std devs). Returns {cond3x3, vel3, sigT2}.
export function conditional(ql, qr, s) {
  const R = rot4d(ql, qr);
  const s2 = [s[0] * s[0], s[1] * s[1], s[2] * s[2], s[3] * s[3]];
  const C = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += R[i][k] * s2[k] * R[j][k];
      C[i][j] = sum;
    }
  const sigT2 = C[3][3];
  const c12 = [C[0][3], C[1][3], C[2][3]];
  const vel = [c12[0] / sigT2, c12[1] / sigT2, c12[2] / sigT2];
  const cond = [
    [C[0][0] - c12[0] * vel[0], C[0][1] - c12[0] * vel[1], C[0][2] - c12[0] * vel[2]],
    [C[1][0] - c12[1] * vel[0], C[1][1] - c12[1] * vel[1], C[1][2] - c12[1] * vel[2]],
    [C[2][0] - c12[2] * vel[0], C[2][1] - c12[2] * vel[1], C[2][2] - c12[2] * vel[2]],
  ];
  return { cond, vel, sigT2 };
}

// Cyclic Jacobi eigendecomposition of a symmetric 3x3. Returns eigenvalues
// val[3] and eigenvector columns vec (vec[row][col]).
export function eigSym3(M) {
  const a = [M[0].slice(), M[1].slice(), M[2].slice()];
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 16; sweep++) {
    const off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-18) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      const apq = a[p][q];
      if (Math.abs(apq) < 1e-300) continue;
      // Numerical-Recipes rotation: zero a[p][q]
      const theta = (a[q][q] - a[p][p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const cph = 1 / Math.sqrt(t * t + 1), sph = t * cph;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = cph * akp - sph * akq;
        a[k][q] = sph * akp + cph * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = cph * apk - sph * aqk;
        a[q][k] = sph * apk + cph * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = cph * vkp - sph * vkq;
        V[k][q] = sph * vkp + cph * vkq;
      }
    }
  }
  return { val: [a[0][0], a[1][1], a[2][2]], vec: V };
}

// Orthonormal rotation matrix (columns = eigenvectors, right-handed) -> quat [x,y,z,w].
function matToQuat(vec) {
  // ensure right-handed (det = +1); flip 3rd column if needed
  const det =
    vec[0][0] * (vec[1][1] * vec[2][2] - vec[1][2] * vec[2][1]) -
    vec[0][1] * (vec[1][0] * vec[2][2] - vec[1][2] * vec[2][0]) +
    vec[0][2] * (vec[1][0] * vec[2][1] - vec[1][1] * vec[2][0]);
  const m = [vec[0].slice(), vec[1].slice(), vec[2].slice()];
  if (det < 0) for (let i = 0; i < 3; i++) m[i][2] = -m[i][2];
  const tr = m[0][0] + m[1][1] + m[2][2];
  let x, y, z, w;
  if (tr > 0) {
    let S = Math.sqrt(tr + 1.0) * 2;
    w = 0.25 * S; x = (m[2][1] - m[1][2]) / S; y = (m[0][2] - m[2][0]) / S; z = (m[1][0] - m[0][1]) / S;
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    let S = Math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2;
    w = (m[2][1] - m[1][2]) / S; x = 0.25 * S; y = (m[0][1] + m[1][0]) / S; z = (m[0][2] + m[2][0]) / S;
  } else if (m[1][1] > m[2][2]) {
    let S = Math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2;
    w = (m[0][2] - m[2][0]) / S; x = (m[0][1] + m[1][0]) / S; y = 0.25 * S; z = (m[1][2] + m[2][1]) / S;
  } else {
    let S = Math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2;
    w = (m[1][0] - m[0][1]) / S; x = (m[0][2] + m[2][0]) / S; y = (m[1][2] + m[2][1]) / S; z = 0.25 * S;
  }
  return [x, y, z, w];
}

// Low-level single-splat slice (used by the parity test).
export function slice4D(ql, qr, s) {
  const { cond, vel, sigT2 } = conditional(ql, qr, s);
  const { val, vec } = eigSym3(cond);
  const scale = [Math.sqrt(Math.max(0, val[0])), Math.sqrt(Math.max(0, val[1])), Math.sqrt(Math.max(0, val[2]))];
  return { scale, quat: matToQuat(vec), vel, sigT2, cond };
}

const clamp01 = (v) => Math.min(1, Math.max(0, v));

// Static 3DGS pack (bg: no t / rotation_r / scaling_t) -> render arrays.
// exp(log-scale), stored quat (w,x,y,z) -> THREE (x,y,z,w), sigmoid opacity,
// SH0 -> linear color. No time slice.
export function staticAttrs(attrs) {
  const n = attrs.count;
  const xyz = attrs.xyz, sc = attrs.scaling, op = attrs.opacity, sh = attrs.sh0, q = attrs.rotation;
  const center = new Float32Array(n * 3);
  const scale = new Float32Array(n * 3);
  const quat = new Float32Array(n * 4);
  const color = new Float32Array(n * 3);
  const opacity = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    center[i * 3] = xyz[i * 3]; center[i * 3 + 1] = xyz[i * 3 + 1]; center[i * 3 + 2] = xyz[i * 3 + 2];
    scale[i * 3] = Math.exp(sc[i * 3]); scale[i * 3 + 1] = Math.exp(sc[i * 3 + 1]); scale[i * 3 + 2] = Math.exp(sc[i * 3 + 2]);
    // stored (w,x,y,z) -> THREE (x,y,z,w)
    quat[i * 4] = q[i * 4 + 1]; quat[i * 4 + 1] = q[i * 4 + 2]; quat[i * 4 + 2] = q[i * 4 + 3]; quat[i * 4 + 3] = q[i * 4];
    opacity[i] = 1 / (1 + Math.exp(-op[i]));
    color[i * 3] = clamp01(0.5 + SH_C0 * sh[i * 3]);
    color[i * 3 + 1] = clamp01(0.5 + SH_C0 * sh[i * 3 + 1]);
    color[i * 3 + 2] = clamp01(0.5 + SH_C0 * sh[i * 3 + 2]);
  }
  return { count: n, center, scale, quat, color, opacity };
}

// Batch: decoded person attrs -> render arrays. scaling/scaling_t are LOG-space
// (as decoded by vpk.js); μ_t = attrs.t; color/opacity as in the static path.
export function sliceAttrs(attrs) {
  const n = attrs.count;
  const R = attrs.rotation, Rr = attrs.rotation_r;
  const sc = attrs.scaling, st = attrs.scaling_t, mt = attrs.t;
  const op = attrs.opacity, sh = attrs.sh0, xyz = attrs.xyz;
  const center = new Float32Array(n * 3);
  const scale = new Float32Array(n * 3);
  const quat = new Float32Array(n * 4);
  const color = new Float32Array(n * 3);
  const opacity = new Float32Array(n);
  const vel = new Float32Array(n * 3);
  const mut = new Float32Array(n);
  const sigT2 = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const ql = [R[i * 4], R[i * 4 + 1], R[i * 4 + 2], R[i * 4 + 3]];
    const qr = [Rr[i * 4], Rr[i * 4 + 1], Rr[i * 4 + 2], Rr[i * 4 + 3]];
    const s = [Math.exp(sc[i * 3]), Math.exp(sc[i * 3 + 1]), Math.exp(sc[i * 3 + 2]), Math.exp(st[i])];
    const r = slice4D(ql, qr, s);
    center[i * 3] = xyz[i * 3]; center[i * 3 + 1] = xyz[i * 3 + 1]; center[i * 3 + 2] = xyz[i * 3 + 2];
    scale[i * 3] = r.scale[0]; scale[i * 3 + 1] = r.scale[1]; scale[i * 3 + 2] = r.scale[2];
    quat[i * 4] = r.quat[0]; quat[i * 4 + 1] = r.quat[1]; quat[i * 4 + 2] = r.quat[2]; quat[i * 4 + 3] = r.quat[3];
    vel[i * 3] = r.vel[0]; vel[i * 3 + 1] = r.vel[1]; vel[i * 3 + 2] = r.vel[2];
    mut[i] = mt[i]; sigT2[i] = r.sigT2;
    opacity[i] = 1 / (1 + Math.exp(-op[i]));
    color[i * 3] = clamp01(0.5 + SH_C0 * sh[i * 3]);
    color[i * 3 + 1] = clamp01(0.5 + SH_C0 * sh[i * 3 + 1]);
    color[i * 3 + 2] = clamp01(0.5 + SH_C0 * sh[i * 3 + 2]);
  }
  return { count: n, center, scale, quat, color, opacity, vel, mut, sigT2 };
}
