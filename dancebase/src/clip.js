// vpkclip1 — the M8 multi-chunk manifest (a thin index over per-chunk vpk1
// scene.json files). Each chunk is a self-contained 120-frame clip trained
// chunk-local; this manifest carries the GLOBAL frame window so the viewer can
// map a global playback frame to each chunk's local time and blend the overlap.
//
// A plain vpk1 scene.json is accepted as a degenerate single-chunk clip (the
// M7 path), so the streaming viewer is a strict superset of the single-clip one.

// clipUrl -> { fps, frame_count(global), span, chunks:[{index, start_frame,
// end_frame, overlap_frames, frame_count(local), sceneUrl}] }
export async function loadClip(clipUrl) {
  const idx = await (await fetch(clipUrl)).json();
  const base = clipUrl.slice(0, clipUrl.lastIndexOf("/") + 1);

  if (idx.version === "vpk1") {
    // single-chunk fallback: a scene.json IS a 1-chunk clip
    const span = idx.t_mapping?.span ?? 10;
    const n = idx.frame_count;
    return {
      fps: idx.fps,
      frame_count: n,
      span,
      chunks: [
        { index: 0, start_frame: 0, end_frame: n, overlap_frames: 0, frame_count: n, span, sceneUrl: clipUrl },
      ],
    };
  }
  if (idx.version !== "vpkclip1") throw new Error(`clip: unsupported version ${idx.version}`);

  const span = idx.span ?? 10;
  const up = idx.up ?? null; // optional world-up hint for viewer boot cameras
  const chunks = idx.chunks.map((c) => ({
    index: c.index,
    start_frame: c.start_frame,
    end_frame: c.end_frame,
    overlap_frames: c.overlap_frames ?? 0,
    frame_count: c.end_frame - c.start_frame,
    span,
    sceneUrl: base + c.scene,
    sceneLiteUrl: c.scene_lite ? base + c.scene_lite : null,
  }));
  // invariants the viewer relies on: sorted, contiguous coverage, <=2 overlap
  for (let i = 1; i < chunks.length; i++) {
    if (chunks[i].start_frame <= chunks[i - 1].start_frame)
      throw new Error("clip: chunks must be sorted by start_frame");
    if (chunks[i].start_frame > chunks[i - 1].end_frame)
      throw new Error(`clip: gap between chunk ${i - 1} and ${i}`);
  }
  return { fps: idx.fps, frame_count: idx.frame_count, span, up, chunks };
}

// per-chunk local playback time for a global frame: the chunk's frame_count
// local frames map to [0, span] (t = span * local / frame_count), matching the
// frozen per-clip mapping (container.py). Clamped to the chunk's range.
export function chunkLocalTime(chunk, globalFrame) {
  const local = Math.min(Math.max(globalFrame - chunk.start_frame, 0), chunk.frame_count);
  return (chunk.span ?? 10) * (local / chunk.frame_count);
}

// chunks covering a global frame (1 normally; 2 inside an overlap band)
export function chunksAt(chunks, globalFrame) {
  return chunks.filter((c) => globalFrame >= c.start_frame && globalFrame < c.end_frame);
}

// Partition-of-unity crossfade over the covering chunks at a global frame.
// Returns [{chunk, weight}] with weights summing to 1. In an overlap band the
// weight ramps cosine from the ending chunk to the starting one; the viewer
// scales each chunk's LINEAR alpha by its weight (NOT logit — the seam-#1 fix),
// so two premultiplied splat sums blend exactly with no density notch.
export function crossfadeWeights(chunks, globalFrame) {
  const cov = chunksAt(chunks, globalFrame);
  if (cov.length === 0) return [];
  if (cov.length === 1) return [{ chunk: cov[0], weight: 1 }];
  if (cov.length > 2) throw new Error(`clip: >2 chunks overlap frame ${globalFrame}`);
  const [a, b] = cov[0].start_frame < cov[1].start_frame ? cov : [cov[1], cov[0]];
  const bandLen = a.end_frame - b.start_frame; // overlap width in frames
  const p = (globalFrame - b.start_frame + 0.5) / bandLen; // (0,1) across the band
  const wB = 0.5 * (1 - Math.cos(Math.PI * Math.min(Math.max(p, 0), 1)));
  return [
    { chunk: a, weight: 1 - wB },
    { chunk: b, weight: wB },
  ];
}
