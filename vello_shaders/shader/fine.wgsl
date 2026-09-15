
struct Tile {
    backdrop: i32,
    segments: u32,
}

#import segment
#import config

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage> segments: array<Segment>;

#import blend
#import ptcl

const GRADIENT_WIDTH = 512;

const IMAGE_QUALITY_LOW = 0u;
const IMAGE_QUALITY_MEDIUM = 1u;
const IMAGE_QUALITY_HIGH = 2u;

const LUMINANCE_MASK_LAYER = 0x10000u;

@group(0) @binding(2)
var<storage> ptcl: array<u32>;

@group(0) @binding(3)
var<storage> info: array<u32>;

@group(0) @binding(4)
var<storage, read_write> blend_spill: array<u32>;

#ifdef packed
@group(0) @binding(5)
var output: texture_storage_2d_array<r32uint, read_write>;
#else
@group(0) @binding(5)
var output: texture_storage_2d_array<rgba8unorm, write>;
#endif

const STG_LAYER_PX: i32 = 8192;

fn stg_layer(p: vec2<i32>) -> i32 {
    return p.y / STG_LAYER_PX;
}

fn stg_local(p: vec2<i32>) -> vec2<i32> {
    return vec2(p.x, p.y % STG_LAYER_PX);
}

@group(0) @binding(6)
var gradients: texture_2d<f32>;

@group(0) @binding(7)
var image_atlas: texture_2d<f32>;

@group(0) @binding(8)
var<storage> effect_params: array<f32>;

#ifdef packed
const SRC_NONE: f32 = 0.0;
const SRC_STORE: f32 = 1.0;
const SRC_REGS: f32 = 2.0;
const SRC_AREA: f32 = 3.0;
const SPARSE_ROW: u32 = 65535u;

struct FxDesc {
    bits: u32,
    program: u32,
    u: array<vec4<f32>, 6>,
    rec: array<vec4<f32>, 12>,
}

/// One operand record: `[source, x0, y0, x1]` then `[y1, dx, dy, decode]`. `lo..hi` is a store
/// rect whose rows carry the page offset; `shift` displaces the read in frame space.
struct Rec {
    source: f32,
    lo: vec2<i32>,
    hi: vec2<i32>,
    shift: vec2<f32>,
    decode: f32,
}

fn rec_of(d: FxDesc, k: u32) -> Rec {
    let a = d.rec[2u * k];
    let b = d.rec[2u * k + 1u];
    var r: Rec;
    r.source = a.x;
    r.lo = vec2<i32>(i32(a.y), i32(a.z));
    r.hi = vec2<i32>(i32(a.w), i32(b.x));
    r.shift = b.yz;
    r.decode = b.w;
    return r;
}

/// The first row of the page holding store row `y`; `config.frame_height` is the page pitch.
fn page_rows(y: i32) -> i32 {
    return (y / i32(config.frame_height)) * i32(config.frame_height);
}

/// The frame position of store pixel `p` for an arm writing record 4's rect: the rect's page and
/// placement (its shift) undone.
fn fx_frame_pos(d: FxDesc, p: vec2<f32>) -> vec2<f32> {
    let out = rec_of(d, 4u);
    return p - out.shift - vec2<f32>(0.0, f32(page_rows(out.lo.y)));
}

/// Frame position `fp` on record `r`'s page, displaced by the record's shift.
fn rec_pos(r: Rec, fp: vec2<f32>) -> vec2<f32> {
    return fp - r.shift + vec2<f32>(0.0, f32(page_rows(r.lo.y)));
}

fn rec_ipos(r: Rec, fp: vec2<f32>) -> vec2<i32> {
    let q = rec_pos(r, fp);
    return vec2<i32>(i32(floor(q.x)), i32(floor(q.y)));
}

fn st_ld(q: vec2<i32>) -> vec4<f32> {
    return unpack4x8unorm(textureLoad(output, stg_local(q), stg_layer(q)).x);
}

fn st_ld_f32(q: vec2<i32>) -> f32 {
    return bitcast<f32>(textureLoad(output, stg_local(q), stg_layer(q)).x);
}

/// Nearest read of `r` at store pixel `q`; a pixel outside the rect reads its clamped edge, or
/// zero when `transparent`.
fn rec_ld(r: Rec, q: vec2<i32>, transparent: bool) -> vec4<f32> {
    let c = clamp(q, r.lo, r.hi - vec2<i32>(1, 1));
    if (transparent && any(c != q)) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    return st_ld(c);
}

/// Bilinear read of `r` at continuous store position `q`; taps outside the rect read its clamped
/// edge, or zero when `transparent`.
fn rec_bilin(r: Rec, q: vec2<f32>, transparent: bool) -> vec4<f32> {
    let ihi = r.hi - vec2<i32>(1, 1);
    let cp = select(clamp(q, vec2<f32>(r.lo), vec2<f32>(ihi)), q, transparent);
    let fl = floor(cp);
    let i0 = vec2<i32>(i32(fl.x), i32(fl.y));
    let f = cp - fl;
    let s00 = rec_ld(r, i0, transparent);
    let s10 = rec_ld(r, i0 + vec2<i32>(1, 0), transparent);
    let s01 = rec_ld(r, i0 + vec2<i32>(0, 1), transparent);
    let s11 = rec_ld(r, i0 + vec2<i32>(1, 1), transparent);
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

/// The sampled distance of record 3 at frame position `fc`: f32 bits, bilinear, decoded.
fn fx_sdf_at(d: FxDesc, fc: vec2<f32>) -> f32 {
    let r = rec_of(d, 3u);
    let q = rec_pos(r, fc - vec2<f32>(0.5, 0.5));
    let ihi = r.hi - vec2<i32>(1, 1);
    let cp = clamp(q, vec2<f32>(r.lo), vec2<f32>(ihi));
    let fl = floor(cp);
    let i0 = vec2<i32>(i32(fl.x), i32(fl.y));
    let f = cp - fl;
    let s00 = st_ld_f32(i0);
    let s10 = st_ld_f32(min(i0 + vec2<i32>(1, 0), ihi));
    let s01 = st_ld_f32(min(i0 + vec2<i32>(0, 1), ihi));
    let s11 = st_ld_f32(min(i0 + vec2<i32>(1, 1), ihi));
    let texel = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
    return (texel - 0.5) * r.decode;
}

fn fx_warp_sample(v: Rec, fp: vec2<f32>, disp: vec2<f32>, ca_scale: f32, ca_amount: f32, transparent: bool) -> vec4<f32> {
    let bp = fp + disp;
    let dlen = length(disp);
    let castr = smoothstep(0.0, 5.0 * ca_scale, dlen);
    var cadir = vec2<f32>(0.0, 0.0);
    if (dlen > 0.01 * ca_scale) { cadir = disp / dlen; }
    let cashift = cadir * ca_amount * castr;
    let cr = rec_bilin(v, rec_pos(v, bp - cashift), transparent);
    let cg = rec_bilin(v, rec_pos(v, bp), transparent);
    let cb = rec_bilin(v, rec_pos(v, bp + cashift), transparent);
    return vec4<f32>(cr.r, cg.g, cb.b, cg.a);
}
#endif

var<private> win_lo: u32 = 0u;
var<private> win_hi: u32 = 0u;

#ifdef msaa

const MASK_LUT_INDEX: u32 = 9;

#ifdef msaa8
const MASK_WIDTH = 32u;
const MASK_HEIGHT = 32u;
const SH_SAMPLES_SIZE = 512u;
const SAMPLE_WORDS_PER_PIXEL = 2u;
@group(0) @binding(MASK_LUT_INDEX)
var<storage> mask_lut: array<u32, 256u>;
#endif

#ifdef msaa16
const MASK_WIDTH = 64u;
const MASK_HEIGHT = 64u;
const SH_SAMPLES_SIZE = 1024u;
const SAMPLE_WORDS_PER_PIXEL = 4u;
@group(0) @binding(MASK_LUT_INDEX)
var<storage> mask_lut: array<u32, 2048u>;
#endif

const WG_SIZE = 64u;
var<workgroup> sh_count: array<u32, WG_SIZE>;

var<workgroup> sh_winding_y: array<atomic<u32>, 4u>;
var<workgroup> sh_winding_y_prefix: array<atomic<u32>, 4u>;
var<workgroup> sh_winding: array<atomic<u32>, 64u>;
var<workgroup> sh_samples: array<atomic<u32>, SH_SAMPLES_SIZE>;

fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

const SEG_SIZE = 5u;

const ONE_MINUS_ULP: f32 = 0.99999994;
const ROBUST_EPSILON: f32 = 2e-7;

fn fill_path_ms(fill: CmdFill, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let even_odd = (fill.size_and_rule & 1u) != 0u;
    if even_odd {
        fill_path_ms_evenodd(fill, local_id, result);
        return;
    }
    let n_segs = fill.size_and_rule >> 1u;
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    if th_ix < 64u {
        if th_ix < 4u {
            atomicStore(&sh_winding_y[th_ix], 0x80808080u);
        }
        atomicStore(&sh_winding[th_ix], 0x80808080u);
    }
    let sample_count = PIXELS_PER_THREAD * SAMPLE_WORDS_PER_PIXEL;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0x80808080u);
    }
    workgroupBarrier();
    let n_batch = (n_segs + (WG_SIZE - 1u)) / WG_SIZE;
    for (var batch = 0u; batch < n_batch; batch++) {
        let seg_ix = batch * WG_SIZE + th_ix;
        let seg_off = fill.seg_data + seg_ix;
        var count = 0u;
        let slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);
        if th_ix < slice_size {
            let segment = segments[seg_off];
            let xy0 = segment.point0;
            let xy1 = segment.point1;
            var y_edge_f = f32(TILE_HEIGHT);
            var delta = select(-1, 1, xy1.x <= xy0.x);
            if xy0.x == 0.0 {
                y_edge_f = xy0.y;
            } else if xy1.x == 0.0 {
                y_edge_f = xy1.y;
            }
            if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicAdd(&sh_winding_y[y_edge >> 2u], u32(delta) << ((y_edge & 3u) << 3u));
            }
        }
        sh_count[th_ix] = count;
        let lg_n = firstLeadingBit(slice_size * 2u - 1u);
        for (var i = 0u; i < lg_n; i++) {
            workgroupBarrier();
            if th_ix >= 1u << i {
                count += sh_count[th_ix - (1u << i)];
            }
            workgroupBarrier();
            sh_count[th_ix] = count;
        }
        let total = workgroupUniformLoad(&sh_count[slice_size - 1u]);
        for (var i = th_ix; i < total; i += WG_SIZE) {
            var lo = 0u;
            var hi = slice_size;
            let goal = i;
            while hi > lo + 1u {
                let mid = (lo + hi) >> 1u;
                if goal >= sh_count[mid - 1u] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let el_ix = lo;
            let last_pixel = i + 1u == sh_count[el_ix];
            let sub_ix = i - select(0u, sh_count[el_ix - 1u], el_ix > 0u);
            let seg_off = fill.seg_data + batch * WG_SIZE + el_ix;
            let segment = segments[seg_off];
            let xy0_in = segment.point0;
            let xy1_in = segment.point1;
            let is_down = xy1_in.y >= xy0_in.y;
            let xy0 = select(xy1_in, xy0_in, is_down);
            let xy1 = select(xy0_in, xy1_in, is_down);

            let dx = abs(xy1.x - xy0.x);
            let dy = xy1.y - xy0.y;
            let idxdy = 1.0 / (dx + dy);
            var a = dx * idxdy;
            let is_positive_slope = xy1.x >= xy0.x;
            let x_sign = select(-1.0, 1.0, is_positive_slope);
            let xt0 = floor(xy0.x * x_sign);
            let c = xy0.x * x_sign - xt0;
            let y0i = floor(xy0.y);
            let ytop = y0i + 1.0;
            let b = min((dy * c + dx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
            let count_x = span(xy0.x, xy1.x) - 1u;
            let count = count_x + span(xy0.y, xy1.y);
            let robust_err = floor(a * (f32(count) - 1.0) + b) - f32(count_x);
            if robust_err != 0.0 {
                a -= ROBUST_EPSILON * sign(robust_err);
            }
            let x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));

            let zf = a * f32(sub_ix) + b;
            let z = floor(zf);
            let x = x0i + i32(x_sign * z);
            let y = i32(y0i) + i32(sub_ix) - i32(z);
            var is_delta: bool;
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                is_delta = y0i == xy0.y;
                is_bump = xy0.x == 0.0 && y0i != xy0.y;
            } else {
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                let delta_pix = pix_ix + 1u;
                if is_delta {
                    let delta = select(u32(-1i), 1u, is_down) << ((delta_pix & 3u) << 3u);
                    atomicAdd(&sh_winding[delta_pix >> 2u], delta);
                }
            }
            let mask_block = u32(is_positive_slope) * (MASK_WIDTH * MASK_HEIGHT / 2u);
            let half_height = f32(MASK_HEIGHT / 2u);
            let mask_row = floor(min(a * half_height, half_height - 1.0)) * f32(MASK_WIDTH);
            let mask_col = floor((zf - z) * f32(MASK_WIDTH));
            let mask_ix = mask_block + u32(mask_row + mask_col);
#ifdef msaa8
            var mask = mask_lut[mask_ix / 4u] >> ((mask_ix % 4u) * 8u);
            mask &= 0xffu;
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(8.0 * (xy0.y - f32(y))));
                mask &= 0xffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(8.0 * (xy1.y - f32(y))));
                mask &= ~(0xffu << mask_shift);
            }
            let mask_a = mask ^ (mask << 7u);
            let mask_b = mask_a ^ (mask_a << 14u);
            let mask0_exp = mask_b & 0x1010101u;
            var mask0_signed = select(mask0_exp, u32(-i32(mask0_exp)), is_down);
            let mask1_exp = (mask_b >> 4u) & 0x1010101u;
            var mask1_signed = select(mask1_exp, u32(-i32(mask1_exp)), is_down);
            if is_bump {
                let bump_delta = select(u32(-0x1010101i), 0x1010101u, is_down);
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
            }
            atomicAdd(&sh_samples[pix_ix * 2u], mask0_signed);
            atomicAdd(&sh_samples[pix_ix * 2u + 1u], mask1_signed);
#endif
#ifdef msaa16
            var mask = mask_lut[mask_ix / 2u] >> ((mask_ix % 2u) * 16u);
            mask &= 0xffffu;
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(16.0 * (xy0.y - f32(y))));
                mask &= 0xffffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(16.0 * (xy1.y - f32(y))));
                mask &= ~(0xffffu << mask_shift);
            }
            let mask0 = mask & 0xffu;
            let mask0_a = mask0 ^ (mask0 << 7u);
            let mask0_b = mask0_a ^ (mask0_a << 14u);
            let mask0_exp = mask0_b & 0x1010101u;
            var mask0_signed = select(mask0_exp, u32(-i32(mask0_exp)), is_down);
            let mask1_exp = (mask0_b >> 4u) & 0x1010101u;
            var mask1_signed = select(mask1_exp, u32(-i32(mask1_exp)), is_down);
            let mask1 = (mask >> 8u) & 0xffu;
            let mask1_a = mask1 ^ (mask1 << 7u);
            let mask1_b = mask1_a ^ (mask1_a << 14u);
            let mask2_exp = mask1_b & 0x1010101u;
            var mask2_signed = select(mask2_exp, u32(-i32(mask2_exp)), is_down);
            let mask3_exp = (mask1_b >> 4u) & 0x1010101u;
            var mask3_signed = select(mask3_exp, u32(-i32(mask3_exp)), is_down);
            if is_bump {
                let bump_delta = select(u32(-0x1010101i), 0x1010101u, is_down);
                mask0_signed += bump_delta;
                mask1_signed += bump_delta;
                mask2_signed += bump_delta;
                mask3_signed += bump_delta;
            }
            atomicAdd(&sh_samples[pix_ix * 4u], mask0_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 1u], mask1_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 2u], mask2_signed);
            atomicAdd(&sh_samples[pix_ix * 4u + 3u], mask3_signed);
#endif
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    let major = (th_ix * PIXELS_PER_THREAD) >> 2u;
    var packed_w = atomicLoad(&sh_winding[major]);
    packed_w += (packed_w - 0x808080u) << 8u;
    packed_w += (packed_w - 0x8080u) << 16u;
    var packed_y = atomicLoad(&sh_winding_y[local_id.y >> 2u]);
    packed_y += (packed_y - 0x808080u) << 8u;
    packed_y += (packed_y - 0x8080u) << 16u;
    var wind_y = (packed_y >> ((local_id.y & 3u) << 3u)) - 0x80u;
    if (local_id.y & 3u) == 3u && local_id.x == 0u {
        let prefix_y = wind_y;
        atomicStore(&sh_winding_y_prefix[local_id.y >> 2u], prefix_y);
    }
    let prefix_x = ((packed_w >> 24u) - 0x80u) * 0x1010101u;
    atomicStore(&sh_winding[major], prefix_x);
    workgroupBarrier();
    for (var i = (major & ~3u); i < major; i++) {
        packed_w += atomicLoad(&sh_winding[i]);
    }
    for (var i = 0u; i < (local_id.y >> 2u); i++) {
        wind_y += atomicLoad(&sh_winding_y_prefix[i]);
    }

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let minor = i;
        let expected_zero = (((packed_w >> (minor * 8u)) + wind_y) & 0xffu) - u32(fill.backdrop);
        if expected_zero >= 256u {
            area[i] = 1.0;
        } else {
#ifdef msaa8
            let samples0 = atomicLoad(&sh_samples[pix_ix * 2u]);
            let samples1 = atomicLoad(&sh_samples[pix_ix * 2u + 1u]);
            let xored0 = (expected_zero * 0x1010101u) ^ samples0;
            let xored0_2 = xored0 | (xored0 * 2u);
            let xored1 = (expected_zero * 0x1010101u) ^ samples1;
            let xored1_2 = xored1 | (xored1 >> 1u);
            let xored2 = (xored0_2 & 0xAAAAAAAAu) | (xored1_2 & 0x55555555u);
            let xored4 = xored2 | (xored2 * 4u);
            let xored8 = xored4 | (xored4 * 16u);
            area[i] = f32(countOneBits(xored8 & 0xC0C0C0C0u)) * 0.125;
#endif
#ifdef msaa16
            let samples0 = atomicLoad(&sh_samples[pix_ix * 4u]);
            let samples1 = atomicLoad(&sh_samples[pix_ix * 4u + 1u]);
            let samples2 = atomicLoad(&sh_samples[pix_ix * 4u + 2u]);
            let samples3 = atomicLoad(&sh_samples[pix_ix * 4u + 3u]);
            let xored0 = (expected_zero * 0x1010101u) ^ samples0;
            let xored0_2 = xored0 | (xored0 * 2u);
            let xored1 = (expected_zero * 0x1010101u) ^ samples1;
            let xored1_2 = xored1 | (xored1 >> 1u);
            let xored01 = (xored0_2 & 0xAAAAAAAAu) | (xored1_2 & 0x55555555u);
            let xored01_4 = xored01 | (xored01 * 4u);
            let xored2 = (expected_zero * 0x1010101u) ^ samples2;
            let xored2_2 = xored2 | (xored2 * 2u);
            let xored3 = (expected_zero * 0x1010101u) ^ samples3;
            let xored3_2 = xored3 | (xored3 >> 1u);
            let xored23 = (xored2_2 & 0xAAAAAAAAu) | (xored3_2 & 0x55555555u);
            let xored23_4 = xored23 | (xored23 >> 2u);
            let xored4 = (xored01_4 & 0xCCCCCCCCu) | (xored23_4 & 0x33333333u);
            let xored8 = xored4 | (xored4 * 16u);
            area[i] = f32(countOneBits(xored8 & 0xF0F0F0F0u)) * 0.0625;
#endif
        }
    }
    *result = area;
}

fn fill_path_ms_evenodd(fill: CmdFill, local_id: vec2<u32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let n_segs = fill.size_and_rule >> 1u;
    let th_ix = local_id.y * (TILE_WIDTH / PIXELS_PER_THREAD) + local_id.x;
    if th_ix < TILE_HEIGHT {
        if th_ix == 0u {
            atomicStore(&sh_winding_y[th_ix], 0u);
        }
        atomicStore(&sh_winding[th_ix], 0u);
    }
    let sample_count = PIXELS_PER_THREAD;
    for (var i = 0u; i < sample_count; i++) {
        atomicStore(&sh_samples[th_ix * sample_count + i], 0u);
    }
    workgroupBarrier();
    let n_batch = (n_segs + (WG_SIZE - 1u)) / WG_SIZE;
    for (var batch = 0u; batch < n_batch; batch++) {
        let seg_ix = batch * WG_SIZE + th_ix;
        let seg_off = fill.seg_data + seg_ix;
        var count = 0u;
        let slice_size = min(n_segs - batch * WG_SIZE, WG_SIZE);
        if th_ix < slice_size {
            let segment = segments[seg_off];
            let xy0 = segment.point0;
            let xy1 = segment.point1;
            var y_edge_f = f32(TILE_HEIGHT);
            if xy0.x == 0.0 {
                y_edge_f = xy0.y;
            } else if xy1.x == 0.0 {
                y_edge_f = xy1.y;
            }
            if !(xy0.y == xy1.y && xy0.y == floor(xy0.y)) {
                count = span(xy0.x, xy1.x) + span(xy0.y, xy1.y) - 1u;
            }
            let y_edge = u32(ceil(y_edge_f));
            if y_edge < TILE_HEIGHT {
                atomicXor(&sh_winding_y[0], 1u << y_edge);
            }
        }
        sh_count[th_ix] = count;
        let lg_n = firstLeadingBit(slice_size * 2u - 1u);
        for (var i = 0u; i < lg_n; i++) {
            workgroupBarrier();
            if th_ix >= 1u << i {
                count += sh_count[th_ix - (1u << i)];
            }
            workgroupBarrier();
            sh_count[th_ix] = count;
        }
        let total = workgroupUniformLoad(&sh_count[slice_size - 1u]);
        for (var i = th_ix; i < total; i += WG_SIZE) {
            var lo = 0u;
            var hi = slice_size;
            let goal = i;
            while hi > lo + 1u {
                let mid = (lo + hi) >> 1u;
                if goal >= sh_count[mid - 1u] {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let el_ix = lo;
            let last_pixel = i + 1u == sh_count[el_ix];
            let sub_ix = i - select(0u, sh_count[el_ix - 1u], el_ix > 0u);
            let seg_off = fill.seg_data + batch * WG_SIZE + el_ix;
            let segment = segments[seg_off];
            let xy0_in = segment.point0;
            let xy1_in = segment.point1;
            let is_down = xy1_in.y >= xy0_in.y;
            let xy0 = select(xy1_in, xy0_in, is_down);
            let xy1 = select(xy0_in, xy1_in, is_down);

            let dx = abs(xy1.x - xy0.x);
            let dy = xy1.y - xy0.y;
            let idxdy = 1.0 / (dx + dy);
            var a = dx * idxdy;
            let is_positive_slope = xy1.x >= xy0.x;
            let x_sign = select(-1.0, 1.0, is_positive_slope);
            let xt0 = floor(xy0.x * x_sign);
            let c = xy0.x * x_sign - xt0;
            let y0i = floor(xy0.y);
            let ytop = y0i + 1.0;
            let b = min((dy * c + dx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
            let count_x = span(xy0.x, xy1.x) - 1u;
            let count = count_x + span(xy0.y, xy1.y);
            let robust_err = floor(a * (f32(count) - 1.0) + b) - f32(count_x);
            if robust_err != 0.0 {
                a -= ROBUST_EPSILON * sign(robust_err);
            }
            let x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));

            let zf = a * f32(sub_ix) + b;
            let z = floor(zf);
            let x = x0i + i32(x_sign * z);
            let y = i32(y0i) + i32(sub_ix) - i32(z);
            var is_delta: bool;
            var is_bump = false;
            let zp = floor(a * f32(sub_ix - 1u) + b);
            if sub_ix == 0u {
                is_delta = y0i == xy0.y;
                is_bump = xy0.x == 0.0;
            } else {
                is_delta = z == zp;
                is_bump = is_positive_slope && !is_delta;
            }
            if u32(x) < TILE_WIDTH - 1u && u32(y) < TILE_HEIGHT {
                if is_delta {
                    atomicXor(&sh_winding[y], 2u << u32(x));
                }
            }
            let mask_block = u32(is_positive_slope) * (MASK_WIDTH * MASK_HEIGHT / 2u);
            let half_height = f32(MASK_HEIGHT / 2u);
            let mask_row = floor(min(a * half_height, half_height - 1.0)) * f32(MASK_WIDTH);
            let mask_col = floor((zf - z) * f32(MASK_WIDTH));
            let mask_ix = mask_block + u32(mask_row + mask_col);
            let pix_ix = u32(y) * TILE_WIDTH + u32(x);
#ifdef msaa8
            var mask = mask_lut[mask_ix / 4u] >> ((mask_ix % 4u) * 8u);
            mask &= 0xffu;
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(8.0 * (xy0.y - f32(y))));
                mask &= 0xffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(8.0 * (xy1.y - f32(y))));
                mask &= ~(0xffu << mask_shift);
            }
            if is_bump {
                mask ^= 0xffu;
            }
            atomicXor(&sh_samples[pix_ix], mask);
#endif
#ifdef msaa16
            var mask = mask_lut[mask_ix / 2u] >> ((mask_ix % 2u) * 16u);
            mask &= 0xffffu;
            if sub_ix == 0u && !is_bump {
                let mask_shift = u32(round(16.0 * (xy0.y - f32(y))));
                mask &= 0xffffu << mask_shift;
            }
            if last_pixel && xy1.x != 0.0 {
                let mask_shift = u32(round(16.0 * (xy1.y - f32(y))));
                mask &= ~(0xffffu << mask_shift);
            }
            if is_bump {
                mask ^= 0xffffu;
            }
            atomicXor(&sh_samples[pix_ix], mask);
#endif
        }
        workgroupBarrier();
    }
    var area: array<f32, PIXELS_PER_THREAD>;
    var scan_x = atomicLoad(&sh_winding[local_id.y]);
    scan_x ^= scan_x << 1u;
    scan_x ^= scan_x << 2u;
    scan_x ^= scan_x << 4u;
    scan_x ^= scan_x << 8u;
    var scan_y = atomicLoad(&sh_winding_y[0]);
    scan_y ^= scan_y << 1u;
    scan_y ^= scan_y << 2u;
    scan_y ^= scan_y << 4u;
    scan_y ^= scan_y << 8u;
    let row_parity = (scan_y >> local_id.y) ^ u32(fill.backdrop);

    for (var i = 0u; i < PIXELS_PER_THREAD; i++) {
        let pix_ix = th_ix * PIXELS_PER_THREAD + i;
        let samples = atomicLoad(&sh_samples[pix_ix]);
        let pix_parity = row_parity ^ (scan_x >> (pix_ix % TILE_WIDTH));
        let pix_mask = u32(-i32(pix_parity & 1u));
#ifdef msaa8
        area[i] = f32(countOneBits((samples ^ pix_mask) & 0xffu)) * 0.125;
#endif
#ifdef msaa16
        area[i] = f32(countOneBits((samples ^ pix_mask) & 0xffffu)) * 0.0625;
#endif
    }
    *result = area;
}
#endif

fn erf7(x: f32) -> f32 {
    let y = clamp(x * 1.1283791671, -100.0, 100.0);
    let yy = y * y;
    let z = y + (0.24295 + (0.03395 + 0.0104 * yy) * yy) * (y * yy);
    return z / sqrt(1.0 + z * z);
}

fn hypot(a: f32, b: f32) -> f32 {
    return sqrt(a * a + b * b);
}

fn read_fill(cmd_ix: u32) -> CmdFill {
    let size_and_rule = ptcl[cmd_ix + 1u];
    let seg_data = ptcl[cmd_ix + 2u];
    let backdrop = i32(ptcl[cmd_ix + 3u]);
    return CmdFill(size_and_rule, seg_data, backdrop);
}

fn read_color(cmd_ix: u32) -> CmdColor {
    let rgba_color = ptcl[cmd_ix + 1u];
    return CmdColor(rgba_color);
}

fn read_blur_rect(cmd_ix: u32) -> CmdBlurRect {
    let info_offset = ptcl[cmd_ix + 1u];
    let rgba_color = ptcl[cmd_ix + 2u];

    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let width = bitcast<f32>(info[info_offset + 6u]);
    let height = bitcast<f32>(info[info_offset + 7u]);
    let radius = bitcast<f32>(info[info_offset + 8u]);
    let std_dev = bitcast<f32>(info[info_offset + 9u]);

    return CmdBlurRect(rgba_color, matrx, xlat, width, height, radius, std_dev);
}

fn read_lin_grad(cmd_ix: u32) -> CmdLinGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let line_x = bitcast<f32>(info[info_offset]);
    let line_y = bitcast<f32>(info[info_offset + 1u]);
    let line_c = bitcast<f32>(info[info_offset + 2u]);
    return CmdLinGrad(index, extend_mode, line_x, line_y, line_c);
}

fn read_rad_grad(cmd_ix: u32) -> CmdRadGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let focal_x = bitcast<f32>(info[info_offset + 6u]);
    let radius = bitcast<f32>(info[info_offset + 7u]);
    let flags_kind = info[info_offset + 8u];
    let flags = flags_kind >> 3u;
    let kind = flags_kind & 0x7u;
    return CmdRadGrad(index, extend_mode, matrx, xlat, focal_x, radius, kind, flags);
}

fn read_sweep_grad(cmd_ix: u32) -> CmdSweepGrad {
    let index_mode = ptcl[cmd_ix + 1u];
    let index = index_mode >> 2u;
    let extend_mode = index_mode & 0x3u;
    let info_offset = ptcl[cmd_ix + 2u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let t0 = bitcast<f32>(info[info_offset + 6u]);
    let t1 = bitcast<f32>(info[info_offset + 7u]);
    return CmdSweepGrad(index, extend_mode, matrx, xlat, t0, t1);
}

fn read_image(cmd_ix: u32) -> CmdImage {
    let info_offset = ptcl[cmd_ix + 1u];
    let m0 = bitcast<f32>(info[info_offset]);
    let m1 = bitcast<f32>(info[info_offset + 1u]);
    let m2 = bitcast<f32>(info[info_offset + 2u]);
    let m3 = bitcast<f32>(info[info_offset + 3u]);
    let matrx = vec4(m0, m1, m2, m3);
    let xlat = vec2(bitcast<f32>(info[info_offset + 4u]), bitcast<f32>(info[info_offset + 5u]));
    let xy = info[info_offset + 6u];
    let width_height = info[info_offset + 7u];
    let sample_alpha = info[info_offset + 8u];
    let alpha = f32(sample_alpha & 0xFFu) / 255.0;
    let format = sample_alpha >> 15u;
    let alpha_type = (sample_alpha >> 14u) & 0x1u;
    let quality = (sample_alpha >> 12u) & 0x3u;
    let x_extend = (sample_alpha >> 10u) & 0x3u;
    let y_extend = (sample_alpha >> 8u) & 0x3u;
    let x = f32(xy >> 16u);
    let y = f32(xy & 0xffffu);
    let width = f32(width_height >> 16u);
    let height = f32(width_height & 0xffffu);
    return CmdImage(matrx, xlat, vec2(x, y), vec2(width, height), format, x_extend, y_extend, quality, alpha, alpha_type);
}

fn read_end_clip(cmd_ix: u32) -> CmdEndClip {
    let blend = ptcl[cmd_ix + 1u];
    let alpha = bitcast<f32>(ptcl[cmd_ix + 2u]);
    return CmdEndClip(blend, alpha);
}

const PIXEL_FORMAT_RGBA: u32 = 0u;
const PIXEL_FORMAT_BGRA: u32 = 1u;
fn pixel_format(pixel: vec4f, format: u32) -> vec4f {
    switch format {
        case PIXEL_FORMAT_BGRA: {
            return pixel.bgra;
        }
        case PIXEL_FORMAT_RGBA, default: {
            return pixel;
        }
    }
}

const ALPHA: u32 = 0u;
const PREMULTIPLIED_ALPHA: u32 = 1u;
fn maybe_premul_alpha(pixel: vec4f, alpha_type: u32) -> vec4f {
    switch alpha_type {
        case PREMULTIPLIED_ALPHA: {
            return pixel;
        }
        case ALPHA, default: {
            return premul_alpha(pixel);
        }
    }
}

const EXTEND_PAD: u32 = 0u;
const EXTEND_REPEAT: u32 = 1u;
const EXTEND_REFLECT: u32 = 2u;
fn extend_mode_normalized(t: f32, mode: u32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, 1.0);
        }
        case EXTEND_REPEAT: {
            return fract(t);
        }
        case EXTEND_REFLECT, default: {
            return abs(t - 2.0 * round(0.5 * t));
        }
    }
}

fn extend_mode(t: f32, mode: u32, max: f32) -> f32 {
    switch mode {
        case EXTEND_PAD: {
            return clamp(t, 0.0, max);
        }
        case EXTEND_REPEAT: {
            return extend_mode_normalized(t / max, mode) * max;
        }
        case EXTEND_REFLECT, default: {
            return extend_mode_normalized(t / max, mode) * max;
        }
    }
}

const MF: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        -(3.0 / 6.0) / 3.0 - 1.0 / 3.0,
        (3.0 / 6.0) / 3.0 + 2.0 * 1.0 / 3.0,
        -(1.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        1.0 - (2.0 / 6.0) / 3.0,
        0.0,
        -3.0 + (12.0 / 6.0) / 3.0 + 1.0 / 3.0,
        2.0 - (9.0 / 6.0) / 3.0 - 1.0 / 3.0
    ),
    vec4<f32>(
        (1.0 / 6.0) / 3.0,
        (3.0 / 6.0) / 3.0 + 1.0 / 3.0,
        3.0 - (15.0 / 6.0) / 3.0 - 2.0 * 1.0 / 3.0,
        -2.0 + (9.0 / 6.0) / 3.0 + 1.0 / 3.0
    ),
    vec4<f32>(
        0.0,
        0.0,
        -1.0 / 3.0,
        (1.0 / 6.0) / 3.0 + 1.0 / 3.0
    )
);

fn cubic_weights(fract: f32) -> vec4<f32> {
    return vec4<f32>(
        single_weight(fract, MF[0][0], MF[0][1], MF[0][2], MF[0][3]),
        single_weight(fract, MF[1][0], MF[1][1], MF[1][2], MF[1][3]),
        single_weight(fract, MF[2][0], MF[2][1], MF[2][2], MF[2][3]),
        single_weight(fract, MF[3][0], MF[3][1], MF[3][2], MF[3][3])
    );
}

fn single_weight(t: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return t * (t * (t * d + c) + b) + a;
}

fn bicubic_sample(
    coords: vec2<f32>,
    atlas_offset: vec2<f32>,
    atlas_max: vec2<f32>,
    alpha_type: u32,
) -> vec4<f32> {
    let frac_coords = fract(coords + vec2(0.5));
    let cx = cubic_weights(frac_coords.x);
    let cy = cubic_weights(frac_coords.y);

    let s00 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-1.5, -1.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s10 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-0.5, -1.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s20 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(0.5, -1.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s30 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(1.5, -1.5), atlas_offset, atlas_max)), 0), alpha_type);

    let s01 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-1.5, -0.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s11 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-0.5, -0.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s21 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(0.5, -0.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s31 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(1.5, -0.5), atlas_offset, atlas_max)), 0), alpha_type);

    let s02 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-1.5, 0.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s12 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-0.5, 0.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s22 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(0.5, 0.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s32 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(1.5, 0.5), atlas_offset, atlas_max)), 0), alpha_type);

    let s03 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-1.5, 1.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s13 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(-0.5, 1.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s23 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(0.5, 1.5), atlas_offset, atlas_max)), 0), alpha_type);
    let s33 = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(clamp(coords + vec2(1.5, 1.5), atlas_offset, atlas_max)), 0), alpha_type);

    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;

    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

const PIXELS_PER_THREAD = 4u;

#ifndef msaa

fn fill_path(fill: CmdFill, xy: vec2<f32>, result: ptr<function, array<f32, PIXELS_PER_THREAD>>) {
    let n_segs = fill.size_and_rule >> 1u;
    let even_odd = (fill.size_and_rule & 1u) != 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    let backdrop_f = f32(fill.backdrop);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        area[i] = backdrop_f;
    }
    for (var i = 0u; i < n_segs; i++) {
        let seg_off = fill.seg_data + i;
        let segment = segments[seg_off];
        let y = segment.point0.y - xy.y;
        let delta = segment.point1 - segment.point0;
        let y0 = clamp(y, 0.0, 1.0);
        let y1 = clamp(y + delta.y, 0.0, 1.0);
        let dy = y0 - y1;
        if dy != 0.0 {
            let vec_y_recip = 1.0 / delta.y;
            let t0 = (y0 - y) * vec_y_recip;
            let t1 = (y1 - y) * vec_y_recip;
            let startx = segment.point0.x - xy.x;
            let x0 = startx + t0 * delta.x;
            let x1 = startx + t1 * delta.x;
            let xmin0 = min(x0, x1);
            let xmax0 = max(x0, x1);
            for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                let i_f = f32(i);
                let xmin = min(xmin0 - i_f, 1.0) - 1.0e-6;
                let xmax = xmax0 - i_f;
                let b = min(xmax, 1.0);
                let c = max(b, 0.0);
                let d = max(xmin, 0.0);
                let a = (b + 0.5 * (d * d - c * c) - xmin) / (xmax - xmin);
                area[i] += a * dy;
            }
        }
        let y_edge = sign(delta.x) * clamp(xy.y - segment.y_edge + 1.0, 0.0, 1.0);
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] += y_edge;
        }
    }
    if even_odd {
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            let a = area[i];
            area[i] = abs(a - 2.0 * round(0.5 * a));
        }
    } else {
        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
            area[i] = min(abs(area[i]), 1.0);
        }
    }
    *result = area;
}

#endif

#ifdef packed
const EFFECT_INLINE_BASE: u32 = 100u;

fn fx_srgb_to_lin(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}
fn fx_lin_to_srgb(c: f32) -> f32 {
    if (c <= 0.0031308) { return c * 12.92; }
    return 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
fn fx_premul_srgb_to_lin(s: vec4<f32>) -> vec4<f32> {
    let a = max(s.a, 1e-5);
    let st = s.rgb / a;
    return vec4<f32>(vec3<f32>(fx_srgb_to_lin(st.r), fx_srgb_to_lin(st.g), fx_srgb_to_lin(st.b)) * a, s.a);
}
fn fx_premul_lin_to_srgb(s: vec4<f32>) -> vec4<f32> {
    let a = max(s.a, 1e-5);
    let st = s.rgb / a;
    return vec4<f32>(vec3<f32>(fx_lin_to_srgb(st.r), fx_lin_to_srgb(st.g), fx_lin_to_srgb(st.b)) * a, s.a);
}

fn fx_scatter_hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
fn fx_scatter_hash2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(fx_scatter_hash(p), fx_scatter_hash(p + vec2<f32>(73.7, 157.3))) * 2.0 - 1.0;
}

// ==== BEGIN GENERATED: field programs ====

fn fx_sdfRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, r: f32) -> f32 {
    let d = abs(p) - halfSize + vec2<f32>(r);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0))) - r;
}

fn fx_ramp(d: f32, edge: f32) -> f32 {
    return clamp(-d / edge, 0.0, 1.0);
}

fn fx_profile(x: f32, kind: i32) -> f32 {
    let t = 1.0 - x;
    if (kind == 0) { return sqrt(max(0.0, 1.0 - t * t)); }
    let t4 = t * t * t * t;
    if (kind == 1) { return pow(max(0.0, 1.0 - t4), 0.25); }
    if (kind == 2) { return 1.0 - pow(max(0.0, 1.0 - t4), 0.25); }
    let c = pow(max(0.0, 1.0 - t4), 0.25);
    let sx = clamp(x, 0.0, 1.0);
    let ss = sx * sx * sx * (sx * (sx * 6.0 - 15.0) + 10.0);
    return mix(c, 1.0 - c, ss);
}
fn fx_profileSlope(x: f32, kind: i32) -> f32 {
    let delta = 0.001;
    return (fx_profile(min(1.0, x + delta), kind) - fx_profile(max(0.0, x - delta), kind)) / (2.0 * delta);
}

fn fx_coverage(d: f32, softness: f32) -> f32 {
    return smoothstep(0.0, softness, -d);
}

fn fx_radialDirection(localPos: vec2<f32>, halfSize: vec2<f32>, splay: f32, tilt: f32) -> vec2<f32> {
    let radialDir = normalize(localPos / max(vec2<f32>(1.0), halfSize));
    let flatDir = vec2<f32>(cos(tilt), sin(tilt));
    let blended = mix(flatDir, radialDir, splay);
    let l = length(blended);
    if (l > 0.001) { return blended / l; }
    return vec2<f32>(0.0);
}

fn fx_snell(theta1: f32, n1: f32, n2: f32) -> f32 {
    let s = (n1 / n2) * sin(theta1);
    if (abs(s) > 1.0) { return -1.0; }
    return asin(s);
}
fn fx_refract(t: f32, thick: f32, n2: f32, kind: i32) -> f32 {
    if (t <= 0.0 || t >= 1.0) { return 0.0; }
    let h = fx_profile(t, kind) * thick;
    let dh = fx_profileSlope(t, kind) * thick;
    let sA = atan(dh);
    let tI = abs(sA);
    let tR = fx_snell(tI, 1.0, n2);
    if (tR < 0.0) { return 0.0; }
    return (h * tan(tR) - h * tan(tI)) * sign(dh);
}

fn fx_band(x: f32, centre: f32, width: f32) -> f32 {
    return exp(-0.5 * pow((x - centre) / max(width, 1e-4), 2.0));
}

fn fx_specular(t: f32, bezel: f32, lightAngle: f32, dir: vec2<f32>, scale: f32) -> f32 {
    if (t <= 0.0 || t >= 1.0) { return 0.0; }
    let band = fx_band(t * bezel, 2.0 * scale, scale);
    let ld = vec2<f32>(cos(lightAngle), sin(lightAngle));
    var f = abs(dot(dir, ld));
    f = pow(f, 2.0);
    return band * f;
}

fn _hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}
fn _vnoise(p: vec2<f32>, seed: f32) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let s = vec2<f32>(seed, 0.0);
    let a = _hash(i + vec2<f32>(0.0, 0.0) + s);
    let b = _hash(i + vec2<f32>(1.0, 0.0) + s);
    let c = _hash(i + vec2<f32>(0.0, 1.0) + s);
    let d = _hash(i + vec2<f32>(1.0, 1.0) + s);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}
fn _fbm(p: vec2<f32>, seed: f32) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var o = 0; o < 4; o = o + 1) {
        v = v + amp * _vnoise(p * freq, seed);
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return v;
}
fn fractalNoise(p: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(_fbm(p, 0.0), _fbm(p, 37.0), _fbm(p, 71.0), _fbm(p, 113.0));
}

fn fx_fieldDistance_lens(u: array<vec4<f32>, 6>, p: vec2<f32>) -> f32 {
    return fx_sdfRoundedBox(p, u[1].xy, min(u[1].z, min(u[1].xy.x, u[1].xy.y)));
}
fn fx_computeField_lens(u: array<vec4<f32>, 6>, fc: vec2<f32>, anchor: vec2<f32>, sampled: bool, sampled_d: f32) -> vec4<f32> {
    let scale = u[4].x;
    let localPos = fc - anchor;
    var n0 = fx_fieldDistance_lens(u, localPos);
    if (sampled) {
        n0 = sampled_d;
    }
    if (n0 > 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    let n1 = fx_ramp(n0, min(u[2].x, min(u[1].xy.x, u[1].xy.y)));
    let n2 = fx_radialDirection(localPos, u[1].xy, u[3].x, u[3].y);
    let n3 = fx_refract(n1, u[2].y, u[2].z, i32(u[1].w));
    let n4 = fx_coverage(n0, 1.5 * u[4].x);
    let dist = n0;
    let edgeT = n1;
    let dir = n2;
    let refracted = n3;
    let mask = n4;
    let bezel = min(u[2].x, min(u[1].x, u[1].y));
    var disp = refracted * scale;
    let edgeFade = pow(1.0 - edgeT, 1.5);
    disp = disp * (1.0 + u[3].z * edgeFade);
    var dpx = dir * disp;
    let zoomFactor = 1.0 / max(u[3].w, 0.1) - 1.0;
    dpx = dpx + localPos * zoomFactor;
    let specular = fx_specular(edgeT, bezel, u[2].w, dir, scale);
    return vec4<f32>(dpx.x, dpx.y, specular, mask);
}
fn fx_computeField_texture(u: array<vec4<f32>, 6>, fc: vec2<f32>, anchor: vec2<f32>, sampled: bool, sampled_d: f32) -> vec4<f32> {
    let scale = u[4].x;
    let localPos = fc - anchor;
    let n0 = fractalNoise(localPos / max(u[0].w, 1.0));
    let n1 = ((n0.rg - vec2<f32>(0.5, 0.5)) * u[0].z);
    let displacement = n1;
    return vec4<f32>(displacement.x, displacement.y, 0.0, 1.0);
}

fn fx_fieldDistance_radial(u: array<vec4<f32>, 6>, p: vec2<f32>) -> f32 {
    return length(p) - max(u[1].x, 1.0);
}
fn fx_computeField_radial(u: array<vec4<f32>, 6>, fc: vec2<f32>, anchor: vec2<f32>, sampled: bool, sampled_d: f32) -> vec4<f32> {
    let scale = u[4].x;
    let localPos = fc - anchor;
    var n0 = fx_fieldDistance_radial(u, localPos);
    if (sampled) {
        n0 = sampled_d;
    }
    let n1 = fx_ramp(n0, u[1].x);
    let specular = n1;
    let mask = n1;
    return vec4<f32>(vec2<f32>(0.0).x, vec2<f32>(0.0).y, specular, mask);
}
fn fx_computeField(d: FxDesc, fc: vec2<f32>) -> vec4<f32> {
    let anchor = d.rec[10].xy;
    let sampled = d.rec[6].x == SRC_STORE;
    let sampled_d = select(0.0, fx_sdf_at(d, fc), sampled);
    if (d.program == 1u) { return fx_computeField_lens(d.u, fc, anchor, sampled, sampled_d); }
    if (d.program == 2u) { return fx_computeField_texture(d.u, fc, anchor, sampled, sampled_d); }
    if (d.program == 3u) { return fx_computeField_radial(d.u, fc, anchor, sampled, sampled_d); }
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
// ==== END GENERATED: field programs ====
fn fx_applyPointwise(bits: u32, shade: bool, maskmix: bool, value0: vec4<f32>, orig: vec4<f32>, field: vec4<f32>, u: array<vec4<f32>, 6>) -> vec4<f32> {
    var value = value0;
    if (shade) {
        let specular = field.b;
        let specularOpacity = u[4].w;
        let specularSaturation = u[5].x;
        let specLuma = dot(value.rgb, vec3<f32>(0.299, 0.587, 0.114));
        var saturated = mix(vec3<f32>(specLuma), value.rgb, 1.0 + specularSaturation);
        saturated = max(saturated, vec3<f32>(0.0));
        let highlightColor = mix(vec3<f32>(1.0, 0.98, 0.95), saturated, min(specularSaturation / 9.0, 1.0));
        value = vec4<f32>(value.rgb + specular * specularOpacity * highlightColor * value.a, value.a);
    }
    if ((bits & 1u) != 0u) {
        value = mix(value, value * orig.a, u[5].y);
    }
    if ((bits & 4u) != 0u) {
        let tintColor = u[3];
        value = vec4<f32>(tintColor.rgb * tintColor.a, tintColor.a) * value.a
            + value * (1.0 - tintColor.a);
    }
    if (maskmix) {
        let mask = field.a;
        value = vec4<f32>(mix(orig.rgb, value.rgb, mask), mix(orig.a, value.a, mask));
    }
    return value;
}

fn fx_load_desc(base: u32) -> FxDesc {
    var d: FxDesc;
    d.bits = u32(effect_params[base]);
    d.program = u32(effect_params[base + 1u]);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let o = base + 2u + k * 4u;
        d.u[k] = vec4<f32>(effect_params[o], effect_params[o + 1u], effect_params[o + 2u], effect_params[o + 3u]);
    }
    for (var k = 0u; k < 12u; k = k + 1u) {
        let o = base + 26u + k * 4u;
        d.rec[k] = vec4<f32>(effect_params[o], effect_params[o + 1u], effect_params[o + 2u], effect_params[o + 3u]);
    }
    return d;
}

fn fx_in_window(round: u32) -> bool {
    return round >= win_lo && (win_hi == SEG_ALL || round < win_hi);
}

/// One separable Gaussian axis over the value record at frame position `fp`.
fn fx_blur_value(d: FxDesc, v: Rec, fp: vec2<f32>) -> vec4<f32> {
    let u = d.u;
    let sigma = max(u[0].z, 0.5);
    let stride = max(i32(u[0].w), 1);
    var radius = i32(ceil(3.0 * sigma));
    radius = radius - (radius % stride);
    let axis = vec2<f32>(u[0].x, u[0].y);
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let srgb_blur = u[2].z != 0.0;
    let transparent = u[2].w != 0.0;
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var wsum = 0.0;
    for (var tt = -radius; tt <= radius; tt = tt + stride) {
        let w = exp(-f32(tt * tt) * inv2s2);
        let rawtap = rec_ld(v, rec_ipos(v, fp + axis * f32(tt)), transparent);
        let tap = select(fx_premul_srgb_to_lin(rawtap), rawtap, srgb_blur);
        acc = acc + w * tap;
        wsum = wsum + w;
    }
    return select(fx_premul_lin_to_srgb(acc / wsum), acc / wsum, srgb_blur);
}

/// Twelve jittered reads of the value record around `fp`, confined to the lens box.
fn fx_scatter_value(d: FxDesc, v: Rec, fp: vec2<f32>) -> vec4<f32> {
    let frost = d.u[4].z;
    let scl = d.u[4].x;
    let fc = fp + vec2<f32>(0.5, 0.5) - (d.u[0].zw - d.u[1].xy);
    if (frost <= 0.01) {
        return rec_bilin(v, rec_pos(v, fp), false);
    }
    let lens_c = d.u[0].zw;
    let lens_h = d.u[1].xy + vec2<f32>(16.0, 16.0);
    var sacc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var t = 0u; t < 12u; t = t + 1u) {
        let n = fx_scatter_hash2(fc + vec2<f32>(f32(t) * 7.3, f32(t) * 13.1));
        let off = n * frost * 6.0 * scl;
        let pb = clamp(fp + off, lens_c - lens_h, lens_c + lens_h);
        sacc = sacc + rec_bilin(v, rec_pos(v, pb), false);
    }
    return sacc / 12.0;
}

/// One arm at one pixel: the value through its head, the reference and coverage as their records
/// say, the pointwise tail, and the landing the compose bits select.
fn fx_arm_value(d: FxDesc, fp: vec2<f32>, acc: vec4<f32>, area: f32) -> vec4<f32> {
    let fld = fx_computeField(d, fp + vec2<f32>(0.5, 0.5));
    let v = rec_of(d, 0u);
    let o = rec_of(d, 1u);
    let c = rec_of(d, 2u);
    var value = acc;
    if (v.source == SRC_AREA) {
        value = vec4<f32>(0.0, 0.0, 0.0, area);
    } else if (v.source == SRC_STORE) {
        if ((d.bits & 32u) != 0u) {
            value = fx_warp_sample(v, fp, fld.xy, d.u[4].x, d.u[4].y, d.u[5].w != 0.0);
        } else if ((d.bits & 64u) != 0u) {
            value = fx_blur_value(d, v, fp);
        } else if ((d.bits & 256u) != 0u) {
            value = fx_scatter_value(d, v, fp);
        } else {
            value = rec_ld(v, rec_ipos(v, fp), true);
        }
    }
    var orig = acc;
    if (o.source == SRC_AREA) {
        orig = vec4<f32>(0.0, 0.0, 0.0, area);
    } else if (o.source == SRC_STORE) {
        orig = rec_ld(o, rec_ipos(o, fp), true);
    }
    var cov = 1.0;
    if (c.source == SRC_AREA) {
        cov = area;
    } else if (c.source == SRC_STORE) {
        cov = rec_ld(c, rec_ipos(c, fp), true).a;
    }
    let eff = fx_applyPointwise(d.bits, (d.bits & 8u) != 0u, (d.bits & 16u) != 0u, value, orig, fld, d.u);
    if ((d.bits & 512u) != 0u) {
        return eff;
    }
    if ((d.bits & 16384u) != 0u) {
        return eff + acc * (1.0 - eff.a);
    }
    if ((d.bits & 128u) != 0u) {
        var scov = value.a;
        if ((d.bits & 2u) != 0u) {
            scov = value.a * (1.0 - d.u[3].a * orig.a);
        }
        let a = d.u[3].a * scov;
        return vec4<f32>(d.u[3].xyz * a, a) + acc * (1.0 - a);
    }
    return mix(acc, eff, cov);
}

fn fx_run_mark(
    cmd_ix: u32,
    xy: vec2<f32>,
    rgba: ptr<function, array<vec4<f32>, PIXELS_PER_THREAD>>,
    area: ptr<function, array<f32, PIXELS_PER_THREAD>>,
) {
    let round = ptcl[cmd_ix + 3u];
    let effect_id = ptcl[cmd_ix + 1u];
    if (effect_id < EFFECT_INLINE_BASE || !fx_in_window(round)) {
        return;
    }
    let d = fx_load_desc(ptcl[cmd_ix + 4u]);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let p = xy + vec2<f32>(f32(i), 0.0);
        (*rgba)[i] = fx_arm_value(d, fx_frame_pos(d, p), (*rgba)[i], (*area)[i]);
    }
}

/// Whether this tile has anything to paint or run in the window: a mark of the window's rounds,
/// or a command after a marker whose round is in the window.
fn fx_window_has_work(tile_ix: u32) -> bool {
    var scan_ix = tile_ix * PTCL_INITIAL_ALLOC + 3u;
    if win_lo > 0u {
        let hm = ptcl[tile_ix * PTCL_INITIAL_ALLOC + 2u];
        if hm != 0u {
            scan_ix = hm;
        }
    }
    var scan_seg = 0u;
    while true {
        let t = ptcl[scan_ix];
        if t == CMD_END {
            break;
        }
        if t == CMD_EFFECT {
            let round = ptcl[scan_ix + 3u];
            if win_hi != SEG_ALL && round >= win_hi {
                break;
            }
            if ptcl[scan_ix + 1u] >= EFFECT_INLINE_BASE && fx_in_window(round) {
                return true;
            }
            scan_seg = round;
            let hm = ptcl[scan_ix + 7u];
            scan_ix += 8u;
            if scan_seg < win_lo && hm != 0u {
                scan_ix = hm;
            }
            continue;
        }
        if t == CMD_JUMP {
            scan_ix = ptcl[scan_ix + 1u];
            continue;
        }
        if fx_in_window(scan_seg) && t != CMD_FILL && t != CMD_SOLID {
            return true;
        }
        switch t {
            case CMD_FILL: {
                scan_ix += 4u;
            }
            case CMD_SOLID, CMD_BEGIN_CLIP: {
                scan_ix += 1u;
            }
            case CMD_COLOR, CMD_IMAGE: {
                scan_ix += 2u;
            }
            case CMD_END_CLIP, CMD_BLUR_RECT, CMD_LIN_GRAD, CMD_RAD_GRAD, CMD_SWEEP_GRAD: {
                scan_ix += 3u;
            }
            default: {
                scan_ix += 1u;
            }
        }
    }
    return false;
}
#endif

fn fx_store_tile(xy: vec2<f32>, rgba: ptr<function, array<vec4<f32>, PIXELS_PER_THREAD>>) {
    let xy_uint = vec2<u32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        let coords = xy_uint + vec2(i, 0u);
        if coords.x < config.target_width && coords.y < config.target_height {
#ifdef packed
            textureStore(output, stg_local(vec2<i32>(coords)), stg_layer(vec2<i32>(coords)), vec4<u32>(pack4x8unorm((*rgba)[i]), 0u, 0u, 0u));
#else
            textureStore(output, stg_local(vec2<i32>(coords)), stg_layer(vec2<i32>(coords)), (*rgba)[i]);
#endif
        }
    }
}

@compute @workgroup_size(4, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    if ptcl[0] == ~0u {
        return;
    }
    win_lo = config.seg_lo;
    win_hi = config.seg_target;
    var tile_xy = wg_id.xy;
#ifdef packed
    if (config.sparse_n != 0u) {
        let wg_lin = wg_id.y * SPARSE_ROW + wg_id.x;
        if (wg_lin >= config.sparse_n) {
            return;
        }
        let raw = bitcast<u32>(effect_params[config.sparse_base + wg_lin]);
        tile_xy = vec2(raw & 0xffffu, (raw >> 16u) & 0x1fffu);
    }
#endif
    let tile_ix = tile_xy.y * config.width_in_tiles + tile_xy.x;
    let xy = vec2(f32(tile_xy.x * TILE_WIDTH + local_id.x * PIXELS_PER_THREAD), f32(tile_xy.y * TILE_HEIGHT + local_id.y));
    let local_xy = vec2(f32(local_id.x * PIXELS_PER_THREAD), f32(local_id.y));
    var rgba: array<vec4<f32>, PIXELS_PER_THREAD>;
#ifdef packed
    if !fx_window_has_work(tile_ix) {
        return;
    }
    let base_xy = vec2<i32>(xy);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        rgba[i] = st_ld(base_xy + vec2(i32(i), 0));
    }
#else
    let base_color = unpack4x8unorm(config.base_color);
    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
        rgba[i] = base_color;
    }
#endif
    var blend_stack: array<array<u32, PIXELS_PER_THREAD>, BLEND_STACK_SPLIT>;
    var clip_depth = 0u;
    var area: array<f32, PIXELS_PER_THREAD>;
    var cmd_ix = tile_ix * PTCL_INITIAL_ALLOC;
    let blend_offset = ptcl[cmd_ix];
    let fx_head_group = ptcl[cmd_ix + 1u];
    let fx_head_marker = ptcl[cmd_ix + 2u];
    cmd_ix += 3u;
    if win_lo > 0u && fx_head_group != 0u {
        var pm = fx_head_marker;
        if ptcl[pm] == CMD_JUMP {
            pm = ptcl[pm + 1u];
        }
        cmd_ix = select(fx_head_marker, fx_head_group, ptcl[pm + 3u] >= win_lo);
    }
    var seg_current = 0u;
    while true {
        let tag = ptcl[cmd_ix];
        if tag == CMD_END {
            break;
        }
        if tag == CMD_EFFECT {
            let round = ptcl[cmd_ix + 3u];
#ifdef packed
            fx_run_mark(cmd_ix, xy, &rgba, &area);
#endif
            if win_hi != SEG_ALL && round >= win_hi {
                break;
            }
            seg_current = round;
            let fx_link_group = ptcl[cmd_ix + 6u];
            let fx_link_marker = ptcl[cmd_ix + 7u];
            cmd_ix += 8u;
            if seg_current < win_lo && fx_link_group != 0u {
                var pm = fx_link_marker;
                if ptcl[pm] == CMD_JUMP {
                    pm = ptcl[pm + 1u];
                }
                cmd_ix = select(fx_link_marker, fx_link_group, ptcl[pm + 3u] >= win_lo);
            }
            continue;
        }
        let seg_active = seg_current >= win_lo
            && (win_hi == SEG_ALL || seg_current < win_hi);
        switch tag {
            case CMD_FILL: {
                let fill = read_fill(cmd_ix);
#ifdef msaa
                fill_path_ms(fill, local_id.xy, &area);
#else
                fill_path(fill, local_xy, &area);
#endif
                cmd_ix += 4u;
            }
            case CMD_SOLID: {
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    area[i] = 1.0;
                }
                cmd_ix += 1u;
            }
            case CMD_COLOR: {
                if seg_active {
                    let color = read_color(cmd_ix);
                    let fg = unpack4x8unorm(color.rgba_color);
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        let fg_i = fg * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                cmd_ix += 2u;
            }
            case CMD_BEGIN_CLIP: {
                if seg_active {
                if clip_depth < BLEND_STACK_SPLIT {
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_stack[clip_depth][i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                } else {
                    let blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                    let local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                    let local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                    for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                        blend_spill[local_blend_start + i] = pack4x8unorm(rgba[i]);
                        rgba[i] = vec4(0.0);
                    }
                }
                clip_depth += 1u;
                }
                cmd_ix += 1u;
            }
            case CMD_END_CLIP: {
                if seg_active {
                let end_clip = read_end_clip(cmd_ix);
                clip_depth -= 1u;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    var bg_rgba: u32;
                    if clip_depth < BLEND_STACK_SPLIT {
                        bg_rgba = blend_stack[clip_depth][i];
                    } else {
                        let blend_in_scratch = clip_depth - BLEND_STACK_SPLIT;
                        let local_tile_ix = local_id.x * PIXELS_PER_THREAD + local_id.y * TILE_WIDTH;
                        let local_blend_start = blend_offset + blend_in_scratch * TILE_WIDTH * TILE_HEIGHT + local_tile_ix;
                        bg_rgba = blend_spill[local_blend_start + i];
                    }
                    let bg = unpack4x8unorm(bg_rgba);
                    let fg = rgba[i] * area[i] * end_clip.alpha;
                    if end_clip.blend == LUMINANCE_MASK_LAYER {
                        if area[i] == 0f {
                            rgba[i] = bg;
                            continue;
                        }
                        let luminance = clamp(svg_lum(unpremultiply(fg)) * fg.a, 0.0, 1.0);
                        rgba[i] = bg * luminance;
                    } else {
                        rgba[i] = blend_mix_compose(bg, fg, end_clip.blend);
                    }
                }
                }
                cmd_ix += 3u;
            }
            case CMD_JUMP: {
                cmd_ix = ptcl[cmd_ix + 1u];
            }
            case CMD_BLUR_RECT: {
                if seg_active {

                let blur = read_blur_rect(cmd_ix);

                let std_dev = max(blur.std_dev, 1e-5);
                let inv_std_dev = 1.0 / std_dev;
                
                let min_edge = min(blur.width, blur.height);
                let radius_max = 0.5 * min_edge;
                let r0 = min(hypot(blur.radius, std_dev * 1.15), radius_max);
                let r1 = min(hypot(blur.radius, std_dev * 2.0), radius_max);

                let exponent = 2.0 * r1 / r0;
                let inv_exponent = 1.0 / exponent;
                
                let delta = 1.25 * std_dev * (exp(-pow(0.5 * inv_std_dev * blur.width, 2.0)) - exp(-pow(0.5 * inv_std_dev * blur.height, 2.0)));
                let width = blur.width + min(delta, 0.0);
                let height = blur.height - max(delta, 0.0);

                let scale = 0.5 * erf7(inv_std_dev * 0.5 * (max(width, height) - 0.5 * blur.radius));

                let blur_rgba = unpack4x8unorm(blur.rgba_color);

                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = blur.matrx.xy * my_xy.x + blur.matrx.zw * my_xy.y + blur.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;

                    let y0 = abs(y) - (height * 0.5 - r1);
                    let y1 = max(y0, 0.0);

                    let x0 = abs(x) - (width * 0.5 - r1);
                    let x1 = max(x0, 0.0);

                    let d_pos = pow(pow(x1, exponent) + pow(y1, exponent), inv_exponent);
                    let d_neg = min(max(x0, y0), 0.0);
                    let d = d_pos + d_neg - r1;
                    let alpha = scale * (erf7(inv_std_dev * (min_edge + d)) - erf7(inv_std_dev * d));

                    let fg_rgba = blur_rgba * alpha;
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                }
                cmd_ix += 3u;
            }
            case CMD_LIN_GRAD: {
                if seg_active {
                let lin = read_lin_grad(cmd_ix);
                let d = lin.line_x * xy.x + lin.line_y * xy.y + lin.line_c;
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_d = d + lin.line_x * f32(i);
                    let x = i32(round(extend_mode_normalized(my_d, lin.extend_mode) * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(x, i32(lin.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                }
                cmd_ix += 3u;
            }
            case CMD_RAD_GRAD: {
                if seg_active {
                let rad = read_rad_grad(cmd_ix);
                let focal_x = rad.focal_x;
                let radius = rad.radius;
                let is_strip = rad.kind == RAD_GRAD_KIND_STRIP;
                let is_circular = rad.kind == RAD_GRAD_KIND_CIRCULAR;
                let is_focal_on_circle = rad.kind == RAD_GRAD_KIND_FOCAL_ON_CIRCLE;
                let is_swapped = (rad.flags & RAD_GRAD_SWAPPED) != 0u;
                let r1_recip = select(1.0 / radius, 0.0, is_circular);
                let less_scale = select(1.0, -1.0, is_swapped || (1.0 - focal_x) < 0.0);
                let t_sign = sign(1.0 - focal_x);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = rad.matrx.xy * my_xy.x + rad.matrx.zw * my_xy.y + rad.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    let xx = x * x;
                    let yy = y * y;
                    var t = 0.0;
                    var is_valid = true;
                    if is_strip {
                        let a = radius - yy;
                        t = sqrt(a) + x;
                        is_valid = a >= 0.0;
                    } else if is_focal_on_circle {
                        t = (xx + yy) / x;
                        is_valid = t >= 0.0 && x != 0.0;
                    } else if radius > 1.0 {
                        t = sqrt(xx + yy) - x * r1_recip;
                    } else {
                        let a = xx - yy;
                        t = less_scale * sqrt(a) - x * r1_recip;
                        is_valid = a >= 0.0 && t >= 0.0;
                    }
                    if is_valid {
                        t = extend_mode_normalized(focal_x + t_sign * t, rad.extend_mode);
                        t = select(t, 1.0 - t, is_swapped);
                        let x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                        let fg_rgba = textureLoad(gradients, vec2(x, i32(rad.index)), 0);
                        let fg_i = fg_rgba * area[i];
                        rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                    }
                }
                }
                cmd_ix += 3u;
            }
            case CMD_SWEEP_GRAD: {
                if seg_active {
                let sweep = read_sweep_grad(cmd_ix);
                let scale = 1.0 / (sweep.t1 - sweep.t0);
                for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                    let my_xy = vec2(xy.x + f32(i), xy.y);
                    let local_xy = sweep.matrx.xy * my_xy.x + sweep.matrx.zw * my_xy.y + sweep.xlat;
                    let x = local_xy.x;
                    let y = local_xy.y;
                    let xabs = abs(x);
                    let yabs = abs(y);
                    let slope = min(xabs, yabs) / max(xabs, yabs);
                    let s = slope * slope;
                    var phi = slope * (0.15912117063999176025390625f + s * (-5.185396969318389892578125e-2f + s * (2.476101927459239959716796875e-2f + s * (-7.0547382347285747528076171875e-3f))));
                    phi = select(phi, 1.0 / 4.0 - phi, xabs < yabs);
                    phi = select(phi, 1.0 / 2.0 - phi, x < 0.0);
                    phi = select(phi, 1.0 - phi, y < 0.0);
                    phi = select(phi, 0.0, phi != phi);
                    phi = (phi - sweep.t0) * scale;
                    let t = extend_mode_normalized(phi, sweep.extend_mode);
                    let ramp_x = i32(round(t * f32(GRADIENT_WIDTH - 1)));
                    let fg_rgba = textureLoad(gradients, vec2(ramp_x, i32(sweep.index)), 0);
                    let fg_i = fg_rgba * area[i];
                    rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                }
                }
                cmd_ix += 3u;
            }
            case CMD_IMAGE: {
                if seg_active {
                let image = read_image(cmd_ix);
                let atlas_max = image.atlas_offset + image.extents - vec2(1.0);
                switch image.quality {
                    case IMAGE_QUALITY_LOW: {
                        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                            if area[i] != 0.0 {
                                let my_xy = vec2(xy.x + f32(i) + 0.5, xy.y + 0.5);
                                var atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
                                atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                                atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                                atlas_uv = atlas_uv + image.atlas_offset;
                                let atlas_uv_clamped = clamp(atlas_uv, image.atlas_offset, atlas_max);
                                let fg_rgba = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(atlas_uv_clamped), 0), image.alpha_type);
                                let fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                            }
                        }
                    }
                    case IMAGE_QUALITY_MEDIUM, default: {
                        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                            if area[i] != 0.0 {
                                let my_xy = vec2(xy.x + f32(i) + 0.5, xy.y + 0.5);
                                var atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
                                atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                                atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                                atlas_uv = atlas_uv + image.atlas_offset - vec2(0.5);
                                let atlas_uv_clamped = clamp(atlas_uv, image.atlas_offset, atlas_max);
                                let uv_quad = vec4(floor(atlas_uv_clamped), ceil(atlas_uv_clamped));
                                let uv_frac = fract(atlas_uv);
                                let a = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xy), 0), image.alpha_type);
                                let b = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.xw), 0), image.alpha_type);
                                let c = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zy), 0), image.alpha_type);
                                let d = maybe_premul_alpha(textureLoad(image_atlas, vec2<i32>(uv_quad.zw), 0), image.alpha_type);
                                let fg_rgba = mix(mix(a, b, uv_frac.y), mix(c, d, uv_frac.y), uv_frac.x);
                                let fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                            }
                        }
                    }
                    case IMAGE_QUALITY_HIGH: {
                        for (var i = 0u; i < PIXELS_PER_THREAD; i += 1u) {
                            if area[i] != 0.0 {
                                let my_xy = vec2(xy.x + f32(i) + 0.5, xy.y + 0.5);
                                var atlas_uv = image.matrx.xy * my_xy.x + image.matrx.zw * my_xy.y + image.xlat;
                                atlas_uv.x = extend_mode(atlas_uv.x, image.x_extend_mode, image.extents.x);
                                atlas_uv.y = extend_mode(atlas_uv.y, image.y_extend_mode, image.extents.y);
                                atlas_uv = atlas_uv + image.atlas_offset;
                                let fg_rgba = bicubic_sample(atlas_uv, image.atlas_offset, atlas_max, image.alpha_type);
                                let fg_i = pixel_format(fg_rgba * area[i] * image.alpha, image.format);
                                rgba[i] = rgba[i] * (1.0 - fg_i.a) + fg_i;
                            }
                        }
                    }
                }
                }
                cmd_ix += 2u;
            }
            default: {}
        }
    }
    fx_store_tile(xy, &rgba);
}

fn premul_alpha(rgba: vec4<f32>) -> vec4<f32> {
    return vec4(rgba.rgb * rgba.a, rgba.a);
}
