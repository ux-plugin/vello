// new_fine — the effects-in-fine CMD_EFFECT arm, rewritten as FIELD-SOURCE × UNIT.
//
// STATUS: consumer half of the `bake` contract (see `render_core::vello::bake`). It reads the SAME
// 26-float descriptor `bake` produces — `[bits, program, 6×vec4 u]` — so swapping `fine.wgsl`'s
// CMD_EFFECT block (currently ~250 lines of `#ifdef have_input/have_draft/load_base` permutations
// braided with WARP/BLUR/SCATTER/SPREAD/MATERIALIZE bit tests) for the functions below is a pure
// STRUCTURAL change: no descriptor changes, no round changes. It is therefore gated the honest way —
// pixel-diffed against the current `fine.wgsl` output on the A/B oracle (`wv_effect_ab`), fixture by
// fixture — NOT by any CPU check, because its correctness IS its pixels. Until that gate passes this
// file is spliced into `fine.wgsl` in place of the old arm behind a build flag; it is not standalone
// WGSL (it references `config`, the source textures, and the shared `fx_*` helpers already in fine).
//
// THE FACTORING. The old arm had no seam between "which field" and "which unit": a lens warp, a
// shadow blur and a frost scatter were three hand-written branches that each re-derived their own
// sampling, masking and compositing. Here every effect is the SAME three stages, and the ONLY thing
// that varies is a value in the descriptor:
//
//   1. FIELD SOURCE  — `fx_computeField(program, u, px)` → (displacement.xy, specular, mask).
//      `program` selects the source: analytic lens (1), sampled-SDF lens (4), texture (2), radial (3).
//      A source is a distinct PERF CLASS, not a branch to duplicate: an analytic field is closed-form
//      and pointwise; a sampled field is one texture read; NEITHER changes the units below it. This is
//      the whole promise of `field.rs` — arbitrary geometry plugs in at the source and every unit
//      downstream is unchanged (proved on the CPU side by `units::sampled_lens_differs_only_in_field_distance`).
//
//   2. HEAD  — `fx_head(...)` turns the input texel(s) into `value` using the field: a plain read, a
//      masked displaced read (WARP, with chromatic aberration), a jittered read (SCATTER), or one axis
//      of a separable blur (BLUR). A head is the one stage that reads OTHER pixels, so it is the fusion
//      barrier; the descriptor's `bits` names which head, and there is exactly one per arm.
//
//   3. TAIL + COMPOSITE  — `fx_applyPointwise(bits, ...)` (already clean in fine: shade, clip, tint,
//      erase, mask-mix) then `fx_composite(...)` lands the result: a masked mix into the accumulator,
//      or a straight SPREAD of the shadow colour over it. MATERIALIZE writes the head's result to a
//      scratch UNMASKED so the next arm reads the full field.
//
// WHAT DISSOLVES. The old arm bound a DIFFERENT texture per permutation (`base_in` / `draft_in` /
// `input_in`) and selected it with `#ifdef`, which is why one gather needed several compiled variants.
// Here the three logical sources are bind slots chosen by `bits` at runtime (uniform per marker, so
// still uniform control flow — WGSL's `textureSample`-under-branch rule is respected), so ONE arm
// serves every effect and the permutation `#ifdef`s are gone.

// ---------------------------------------------------------------------------------------------------
// bits vocabulary — mirrors `render_core::vello::bake::bits`. One flag per unit/mode.
const FX_ERASE:       u32 = 2u;
const FX_TINT:        u32 = 4u;
const FX_SHADE:       u32 = 8u;
const FX_MASKMIX:     u32 = 16u;
const FX_WARP:        u32 = 32u;
const FX_BLUR:        u32 = 64u;
const FX_SPREAD:      u32 = 128u;
const FX_SCATTER:     u32 = 256u;
const FX_MATERIALIZE: u32 = 512u;
const FX_SRGB:        u32 = 1024u;
const FX_SHADOW_EDGE: u32 = 2048u;
// (bit 4096 FLOOD_ERASE is gone: the Text inner flood is a plain Rasterize node in the DAG now, read
//  by the EraseBy arm like any other coverage — no back-sampling special case.)
const FX_SCRATCH_COV: u32 = 8192u;

// The three logical source surfaces an arm may read are the module-scope bindings already declared in
// `fine.wgsl` — `base_in` (the materialized backdrop a WARP/blur-H reads), `draft_in` (the H-pass
// result a blur-V reads), `input_in` (the previous chained link's output a scatter/tail reads). WGSL
// forbids a texture in a struct or a function parameter freely, so — like the rest of fine — the head
// functions reference these bindings directly, and `bits` (uniform per marker) selects between them in
// uniform control flow. A plain pointwise arm reads none of them (it works over the accumulator pixel
// already in registers). `fx_bilin` reads `base_in`; `fx_bilin_input` reads `input_in` (both already
// in fine); a blur-V reads `draft_in` directly below.

// ---------------------------------------------------------------------------------------------------
// HEAD — produce (value, orig, cov) for one pixel from the field + the source surfaces.
//
// `value` is the head's result the tail refines; `orig` is the untouched backdrop the mask-mix/erase
// confine against; `cov` is the coverage the composite uses (1.0 when the arm materializes UNMASKED,
// else the rasterized silhouette the caller passes in `area`). A head is uniform-branched on `bits`,
// so every `textureSample`/`textureLoad` sits in uniform control flow.
struct FxHead { value: vec4<f32>, orig: vec4<f32>, cov: f32 }

// A masked displaced read of `base` at the field displacement, with chromatic aberration growing
// toward the rim — the lens refraction. Byte-for-byte the old `warp` branch. `scale` = u[4].x,
// `ca` = u[4].y (chromatic-aberration amount).
fn fx_head_warp(field: vec4<f32>, u: array<vec4<f32>, 6>, px: vec2<f32>) -> FxHead {
    let ipx = vec2<i32>(i32(px.x), i32(px.y));
    let bp = px + field.xy;
    let dlen = length(field.xy);
    let castr = smoothstep(0.0, 5.0 * u[4].x, dlen);
    var cadir = vec2<f32>(0.0, 0.0);
    if (dlen > 0.01 * u[4].x) { cadir = field.xy / dlen; }
    let cashift = cadir * u[4].y * castr;
    // fx_bilin reads `base_in` (the materialized backdrop) bilinearly — the accessor the old arm used.
    let cr = fx_bilin(bp - cashift);
    let cg = fx_bilin(bp);
    let cb = fx_bilin(bp + cashift);
    let value = vec4<f32>(cr.r, cg.g, cb.b, cg.a);
    let orig = textureLoad(base_in, ipx, 0);
    // The refraction mask (`field.a`) is applied ONCE by the mask-mix tail; the composite keeps the
    // rasterized silhouette `cov`, so leave `cov` untouched here (masking twice rings the rim).
    return FxHead(value, orig, -1.0);   // cov = -1 sentinel: "caller keeps its `area` coverage"
}

// A jittered read of `input_in` (the previous link's blurred-warp surface) — the frost scatter. This
// is the HEAD of the final frost arm: shade + mask-mix run pointwise after it in the SAME arm (`fuse()`
// returns `[scatter, shade, maskmix]` as one run), so `orig` is the ORIGINAL backdrop (`base_in`) the
// mask-mix confines over, and `cov` is the sentinel -1 → the arm composites MASKED against the shape's
// silhouette. No separate headless tail, and so no flag: the scatter head already names `input_in`.
// `frost` = u[4].z, `scale` = u[4].x.
fn fx_head_scatter(u: array<vec4<f32>, 6>, px: vec2<f32>) -> FxHead {
    let frost = u[4].z;
    let scl = u[4].x;
    let fc = px + vec2<f32>(0.5, 0.5);
    var value = vec4<f32>(0.0);
    if (frost > 0.01) {
        for (var t = 0u; t < 12u; t = t + 1u) {
            let n = fx_scatter_hash2(fc + vec2<f32>(f32(t) * 7.3, f32(t) * 13.1));
            value = value + fx_bilin_input(px + n * frost * 6.0 * scl);
        }
        value = value / 12.0;
    } else {
        value = fx_bilin_input(px);
    }
    let orig = textureLoad(base_in, vec2<i32>(i32(px.x), i32(px.y)), 0);
    return FxHead(value, orig, -1.0);
}

// One axis of a separable Gaussian of `sigma` = u[0].z along `axis` = u[0].xy. Reads `draft` on the V
// pass (a materialized H result) else the H source (`input` for a frosted warp, `base` for a backdrop
// or shadow silhouette). Writes UNMASKED (cov = 1). `is_v` picks the source, `srgb`/`shadow_edge`
// mirror the old bits 1024/2048 (sRGB mixing, transparent OOB for a shadow silhouette).
fn fx_head_blur(u: array<vec4<f32>, 6>, bits: u32, px: vec2<f32>, is_v: bool, base_color: u32) -> FxHead {
    let sigma = max(u[0].z, 0.5);
    let radius = i32(ceil(3.0 * sigma));
    let ipx = vec2<i32>(i32(px.x), i32(px.y));
    let axis = vec2<i32>(i32(u[0].x), i32(u[0].y));
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let srgb = (bits & FX_SRGB) != 0u;
    let shadow_edge = (bits & FX_SHADOW_EDGE) != 0u;
    let bgraw = unpack4x8unorm(base_color);
    let bg = select(
        select(fx_premul_srgb_to_lin(bgraw), bgraw, srgb),
        vec4<f32>(0.0),
        shadow_edge,
    );
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var tt = -radius; tt <= radius; tt = tt + 1) {
        let w = exp(-f32(tt * tt) * inv2s2);
        let sp = ipx + axis * tt;
        var dims: vec2<i32>;
        var rawtap: vec4<f32>;
        if (is_v) {
            dims = vec2<i32>(textureDimensions(draft_in));
            rawtap = textureLoad(draft_in, sp, 0);
        } else {
            dims = vec2<i32>(textureDimensions(base_in));
            rawtap = textureLoad(base_in, sp, 0);
        }
        let inb = sp.x >= 0 && sp.y >= 0 && sp.x < dims.x && sp.y < dims.y;
        let tapc = select(fx_premul_srgb_to_lin(rawtap), rawtap, srgb);
        acc = acc + w * select(bg, tapc, inb);
        wsum = wsum + w;
    }
    let value = select(fx_premul_lin_to_srgb(acc / wsum), acc / wsum, srgb);
    return FxHead(value, value, 1.0);
}

// ---------------------------------------------------------------------------------------------------
// COMPOSITE — land the finished `eff` onto the accumulator pixel `dst`.
//
// Two modes, chosen by `bits`, replacing the old arm's `if spread { … } else { … }`:
//   * SPREAD — a shadow's straight colour (u[3]) source-OVER the accumulator at coverage `scov`. The
//     coverage source is itself derived from bits (blurred silhouette alpha, precomputed scratch, or
//     the erase of a flood by a punch), exactly as the old arm did.
//   * masked — `mix(dst, eff, cov)`: the backdrop effect confined to the rasterized silhouette.
fn fx_composite(bits: u32, dst: vec4<f32>, eff: vec4<f32>, headv: vec4<f32>, cov: f32, u: array<vec4<f32>, 6>) -> vec4<f32> {
    if ((bits & FX_SPREAD) != 0u) {
        let is_blur = (bits & FX_BLUR) != 0u;
        let is_erase = (bits & FX_ERASE) != 0u;
        let scratch_cov = (bits & FX_SCRATCH_COV) != 0u;
        var scov = select(cov, headv.a, is_blur);
        if (scratch_cov) { scov = headv.a; }
        else if (is_erase) { scov = cov * (1.0 - u[3].a * headv.a); }
        let a = u[3].a * scov;
        return vec4<f32>(u[3].xyz * a, a) + dst * (1.0 - a);
    }
    return mix(dst, eff, cov);
}

// ---------------------------------------------------------------------------------------------------
// THE ARM — one call replacing the whole CMD_EFFECT interpreter body. `dst` is the accumulator pixel;
// `area_cov` is the rasterized silhouette coverage the CmdFill left; the descriptor is (`bits`,
// `program`, `u`). Returns the new accumulator pixel. The head is picked by `bits`; the field is
// evaluated once (only the units that read it — shade/mask-mix, and the warp displacement — need it);
// the tail and composite finish. MATERIALIZE forces `cov = 1` so an intermediate link writes unmasked.
fn fx_effect_arm(bits: u32, program: u32, u: array<vec4<f32>, 6>, px: vec2<f32>, dst: vec4<f32>, area_cov: f32, base_color: u32) -> vec4<f32> {
    let field = fx_computeField(program, u, px + vec2<f32>(0.5, 0.5));
    let shade = (bits & FX_SHADE) != 0u;
    let maskmix = (bits & FX_MASKMIX) != 0u;

    var head: FxHead;
    if ((bits & FX_WARP) != 0u) {
        head = fx_head_warp(field, u, px);
    } else if ((bits & FX_SCATTER) != 0u) {
        head = fx_head_scatter(u, px);
    } else if ((bits & FX_BLUR) != 0u) {
        let is_v = u[0].y > 0.5;   // axis (0,1) = vertical pass reads the draft
        head = fx_head_blur(u, bits, px, is_v, base_color);
    } else {
        // Plain pointwise (tint / field tint): work over the accumulator pixel, confine against it.
        head = FxHead(dst, dst, -1.0);
    }

    // cov: the head's own if it materialized (>= 0), else the rasterized silhouette the caller holds.
    var cov = select(area_cov, head.cov, head.cov >= 0.0);
    if ((bits & FX_MATERIALIZE) != 0u) { cov = 1.0; }

    let eff = fx_applyPointwise(bits, shade, maskmix, head.value, head.orig, field, u);
    return fx_composite(bits, dst, eff, head.value, cov, u);
}
