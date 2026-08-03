// vpk2 loader (ADR-011 lever B): decode the WebP-atlas container in the browser.
//
// Byte-exact port of volcap/export/vpk2.py::read_pack_v2 — parse the VPK2
// binary (magic | u32-LE header len | JSON header | concatenated WebP
// bitstreams), decode each lossless-WebP atlas, slice the first `count`
// pixels row-major, rebuild the vpk1 integer code planes (pos/t hi|lo joins,
// rotation hi+aux bitfield join, sh0 palette gather for lever-L1 packs), then
// run the SAME dequant kernels as vpk.js. Splat order is the Morton atlas
// order — a free permutation (the 4D slice pipeline is order-agnostic, P6).
//
// WebP decode: ImageDecoder (WebCodecs) when available, else
// createImageBitmap + OffscreenCanvas with colorSpaceConversion/premultiply
// disabled. Lossless RGB WebP has no alpha, so both paths are bit-faithful.
// Parity: ../test/vpk2.browser.html against gen_vpk2_reference.py.

import {
  decodeColorU8,
  decodeGridU16,
  decodeLogU8Log,
  decodeQuatSmallest3,
  decodeSigmoidU8Logit,
} from "./vpk.js";

const MAGIC = "VPK2";

export function parseHeader(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(0, 4));
  if (magic !== MAGIC) throw new Error(`vpk2: bad magic ${magic}`);
  const hlen = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + hlen)));
  return { header, base: 8 + hlen };
}

async function decodeWebpRGBA(blobBytes, width, height) {
  if (typeof ImageDecoder !== "undefined" && (await ImageDecoder.isTypeSupported("image/webp"))) {
    const dec = new ImageDecoder({ data: blobBytes, type: "image/webp" });
    const { image } = await dec.decode();
    const out = new Uint8Array(width * height * 4);
    await image.copyTo(out, { format: "RGBA" });
    image.close();
    dec.close();
    return out;
  }
  const bitmap = await createImageBitmap(new Blob([blobBytes], { type: "image/webp" }), {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return new Uint8Array(ctx.getImageData(0, 0, width, height).data.buffer);
}

// RGBA pixels -> tightly-packed u8 (count, channels), row-major, tail dropped.
function packChannels(rgba, count, channels) {
  const out = new Uint8Array(count * channels);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < channels; c++) out[i * channels + c] = rgba[i * 4 + c];
  }
  return out;
}

// Decode all atlas images of a .vpk2 buffer into {name: u8 (count*channels)}.
export async function decodeImages(buffer) {
  const { header, base } = parseHeader(buffer);
  const images = {};
  for (const entry of header.images) {
    const blob = new Uint8Array(buffer, base + entry.offset, entry.bytes);
    const rgba = await decodeWebpRGBA(blob, header.atlas_width, entry.height);
    images[entry.name] = packChannels(rgba, header.count, entry.channels);
  }
  return { header, images };
}

function joinU16(hi, lo) {
  const out = new Uint16Array(hi.length);
  for (let i = 0; i < hi.length; i++) out[i] = (hi[i] << 8) | lo[i];
  return out;
}

function joinRotation(hi, aux) {
  // aux = (idx<<6)|(lo0<<4)|(lo1<<2)|lo2 ; comps = hi<<2 | lo
  const n = aux.length;
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const a = aux[i];
    const idx = (a >>> 6) & 0x3;
    const c0 = (hi[i * 3] << 2) | ((a >>> 4) & 0x3);
    const c1 = (hi[i * 3 + 1] << 2) | ((a >>> 2) & 0x3);
    const c2 = (hi[i * 3 + 2] << 2) | (a & 0x3);
    out[i] = ((idx << 30) | (c0 << 20) | (c1 << 10) | c2) >>> 0;
  }
  return out;
}

// Rebuild the vpk1 integer code planes from decoded atlas images (byte-exact
// counterpart of _restore_planes; sorted Morton order).
export function restorePlanes(header, images) {
  const planes = {};
  for (const name of header.source_planes) {
    if (name === "xyz") planes.xyz = joinU16(images.pos_hi, images.pos_lo);
    else if (name === "t") planes.t = joinU16(images.t_hi, images.t_lo);
    else if (name === "rotation" || name === "rotation_r")
      planes[name] = joinRotation(images[`${name}_hi`], images[`${name}_aux`]);
    else if (name === "sh0" && images.sh0_labels) {
      const palette = header.sh0_palette;
      if (!palette) throw new Error("vpk2: sh0_labels without sh0_palette in header");
      const labels = images.sh0_labels;
      const sh0 = new Uint8Array(labels.length * 3);
      for (let i = 0; i < labels.length; i++) {
        const p = palette[labels[i]];
        sh0[i * 3] = p[0];
        sh0[i * 3 + 1] = p[1];
        sh0[i * 3 + 2] = p[2];
      }
      planes.sh0 = sh0;
    } else planes[name] = images[name];
  }
  return planes;
}

// .vpk2 ArrayBuffer -> decoded attribute dict (same shape as vpk.js decodePack).
export async function decodePackV2(descriptor, buffer) {
  const { header, images } = await decodeImages(buffer);
  const planes = restorePlanes(header, images);
  const dq = header.dequant;
  const n = header.count;
  const attrs = { count: n };
  attrs.xyz = decodeGridU16(planes.xyz, n, 3, dq.xyz.lo, dq.xyz.hi);
  if (planes.t) attrs.t = decodeGridU16(planes.t, n, 1, dq.t.lo, dq.t.hi);
  for (const name of ["rotation", "rotation_r"]) {
    if (planes[name]) attrs[name] = decodeQuatSmallest3(planes[name], n);
  }
  attrs.scaling = decodeLogU8Log(planes.scaling, n, 3, dq.scaling.lo, dq.scaling.hi);
  if (planes.scaling_t)
    attrs.scaling_t = decodeLogU8Log(planes.scaling_t, n, 1, dq.scaling_t.lo, dq.scaling_t.hi);
  attrs.opacity = decodeSigmoidU8Logit(planes.opacity, n);
  attrs.sh0 = decodeColorU8(planes.sh0, n);
  return attrs;
}
