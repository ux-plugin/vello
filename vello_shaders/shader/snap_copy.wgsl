// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Copies scattered rects of the packed (r32uint) accumulator into the snapshot — the compute-pass
// form of the per-window encoder blits, so a batched pass can refresh the backdrop between window
// dispatches without closing the pass. One dispatch covers every rect of a refresh: workgroup i
// reads its 16x16 tile from `tiles[i]` = (tile_x, tile_y, clip_x1, clip_y1) in device pixels.

@group(0) @binding(0) var<storage, read> tiles: array<vec4<u32>>;
@group(0) @binding(1) var src: texture_2d<u32>;
@group(0) @binding(2) var dst: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let t = tiles[wid.x];
    let x = t.x + lid.x;
    let y = t.y + lid.y;
    if (x < t.z && y < t.w) {
        let p = vec2<i32>(i32(x), i32(y));
        textureStore(dst, p, textureLoad(src, p, 0));
    }
}
