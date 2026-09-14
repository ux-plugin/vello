// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT OR Unlicense

// This must be kept in sync with `ConfigUniform` in `vello_encoding/src/config.rs`
struct Config {
    width_in_tiles: u32,
    height_in_tiles: u32,

    target_width: u32,
    target_height: u32,

    // The initial color applied to the pixels in a tile during the fine stage.
    // The format is packed RGBA8 in MSB order.
    base_color: u32,

    n_drawobj: u32,
    n_path: u32,
    n_clip: u32,

    // To reduce the number of bindings, info and bin data are combined
    // into one buffer.
    bin_data_start: u32,

    // offsets within scene buffer (in u32 units)
    pathtag_base: u32,
    pathdata_base: u32,

    drawtag_base: u32,
    drawdata_base: u32,

    transform_base: u32,
    style_base: u32,

    // Sizes of bump allocated buffers (in element size units)
    lines_size: u32,
    binning_size: u32,
    tiles_size: u32,
    seg_counts_size: u32,
    segments_size: u32,
    blend_size: u32,
    ptcl_size: u32,

    // Whole-viewport phased render: the half-open draw-object index range [draw_start, draw_end) this
    // phase includes. Draws outside it are forced to an empty bbox in binning, so they land in no bin
    // and appear in no PTCL this phase. Defaults to the full range (0, n_drawobj), leaving a normal
    // single-phase render unchanged.
    draw_start: u32,
    draw_end: u32,

    // Whole-viewport windowed fine: the tile-round window `[seg_lo, seg_target)` this dispatch
    // renders. Each CMD_EFFECT marker carries its effect's ROUND (reach-disjoint effects share a
    // round); a command's round is that of the last marker before it on ITS tile, so untouched
    // tiles render early and total passes scale with effect stack depth, not effect count.
    // `seg_target == SEG_ALL` removes the upper bound; with `seg_lo == 0` (the defaults) that is
    // the normal, non-windowed render. the sparse fields keep the uniform a multiple of 16 bytes (a
    // WebGPU requirement). Must be kept in sync with `ConfigUniform` in `vello_encoding/src/config.rs`.
    seg_target: u32,
    seg_lo: u32,
    // Sparse window dispatch: when `sparse_n != 0`, fine's grid is `(min(n, 65535), ceil(n / 65535), 1)`
    // workgroups and workgroup `(x, y)` reads tile word `y * 65535 + x` from
    // `effect_params[sparse_base + ..]` (`y<<16 | x`, biased by 0x40000000 — see fine.wgsl `main`).
    // Zero = the full grid.
    sparse_base: u32,
    sparse_n: u32,
    // The FRAME extent in pixels: the accumulator/backdrop rows. The target extent covers the whole
    // tile grid — frame rows plus any interest-region rows rented below them — so backdrop reads and
    // their edge-extend clamps must bound against the frame, never the grid. Equal to target_* when
    // no regions are rented, leaving every clamp byte-identical.
    frame_width: u32,
    frame_height: u32,
    // Pad to a 16-byte multiple (WebGPU uniform requirement).
    frame_pad0: u32,
    frame_pad1: u32,
}

// Sentinel `seg_target` value meaning "render all segments in one pass" (the non-segmented default).
const SEG_ALL = 0xffffffffu;

// Geometry of tiles and bins

const TILE_WIDTH = 16u;
const TILE_HEIGHT = 16u;
// Number of tiles per bin
const N_TILE_X = 16u;
const N_TILE_Y = 16u;
const N_TILE = N_TILE_X * N_TILE_Y;

// Not currently supporting non-square tiles
const TILE_SCALE = 0.0625;

// The "split" point between using local memory in fine for the blend stack and spilling to the blend_spill buffer.
// A higher value will increase vgpr ("register") pressure in fine, but decrease required dynamic memory allocation.
// If changing, also change in vello_shaders/src/cpu/coarse.rs.
const BLEND_STACK_SPLIT = 4u;

// The following are computed in draw_leaf from the generic gradient parameters
// encoded in the scene, and stored in the gradient's info struct, for
// consumption during fine rasterization.

// Radial gradient kinds
const RAD_GRAD_KIND_CIRCULAR = 1u;
const RAD_GRAD_KIND_STRIP = 2u;
const RAD_GRAD_KIND_FOCAL_ON_CIRCLE = 3u;
const RAD_GRAD_KIND_CONE = 4u;

// Radial gradient flags
const RAD_GRAD_SWAPPED = 1u;
