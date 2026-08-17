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

    // Whole-viewport segmented fine: the segment index this fine dispatch renders. Segments are the
    // command ranges between CMD_EFFECT boundaries in one shared PTCL. fine paints only the commands
    // whose running segment index equals `seg_target`, and stops once it is past that segment.
    // `SEG_ALL` (the default) renders every segment in a single pass — the normal, non-segmented
    // behavior. `_pad_seg*` keep the uniform a multiple of 16 bytes (a WebGPU requirement). Must be
    // kept in sync with `ConfigUniform` in `vello_encoding/src/config.rs`.
    seg_target: u32,
    _pad_seg0: u32,
    _pad_seg1: u32,
    _pad_seg2: u32,
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
