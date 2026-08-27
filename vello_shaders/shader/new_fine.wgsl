// new_fine — the effects-in-fine CMD_EFFECT arm as FIELD-SOURCE × UNIT.
//
// STATUS: the clean consumer half of the `bake` contract (see `render_core::vello::bake`). It reads the
// SAME 26-float descriptor `bake` produces — `[bits, program, 6×vec4 u]` — driven by the DAG scheduler
// (WV_DAG). It splices into `fine.wgsl` in place of the old ~250-line `#ifdef` × bit-test arm; the only
// externals it needs are the texture-sampling helpers already in fine (`fx_bilin`/`fx_bilin_input`/
// `fx_scatter_hash2`/`fx_premul_*`) and the source-texture bindings (`base_in`/`draft_in`/`input_in`,
// plus `fieldTex`/`fieldSamp` for the SDF source). Verified structurally by naga; pixel-gated against
// the old arm via `wv_effect_ab`, brought up one unit at a time.
//
// THE FACTORING. There is NO effect-specific path — a lens, a shadow, a blur are the SAME three stages,
// and only descriptor values vary:
//
//   1. FIELD SOURCE — `fieldDistance(u, p)` is the one place geometry enters, and it has exactly TWO
//      paths behind ONE function: the closed-form rounded box (analytical) and the baked signed-distance
//      texture (SDF/sampled), selected by the source flag `u[5].w`. Everything downstream of the distance
//      — ramp, refraction, specular, coverage, the whole `computeField` — is byte-identical for both
//      (proved on the CPU side by `units::sampled_lens_differs_only_in_field_distance`). So arbitrary
//      geometry plugs in at the source and no unit changes. `computeField` takes the arm's `u` directly
//      (no global field buffer), which is what lets one text serve fine's per-tile compute arm.
//
//   2. HEAD — `fx_head_*` turns the input texel(s) into `value` using the field: a plain read, a masked
//      displaced read (WARP + chromatic aberration), a jittered read (SCATTER), or one axis of a
//      separable blur (BLUR). The head is the only stage that reads other pixels — the fusion barrier;
//      `bits` names which, exactly one per arm.
//
//   3. TAIL + COMPOSITE — `fx_tail` runs the pointwise units (shade, clip, tint, erase, mask-mix) and
//      `fx_composite` lands the result (masked mix, or a SPREAD of a straight colour). MATERIALIZE writes
//      the head's result to a scratch UNMASKED so the next arm reads the full field.

// ---------------------------------------------------------------------------------------------------
// bits vocabulary — mirrors `render_core::vello::bake::bits`.
const FX_CLIP:        u32 = 1u;
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
const FX_SCRATCH_COV: u32 = 8192u;

// ===================================================================================================
// FIELD SOURCE — the rounded-box lens field, `u`-parameterised. Helpers below are source-agnostic and
// verbatim from the generated `field_prelude`; only `fieldDistance` chooses analytical vs SDF.
// ===================================================================================================

fn sdfRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, r: f32) -> f32 {
    let d = abs(p) - halfSize + vec2<f32>(r);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0))) - r;
}

// ONE distance, TWO sources. `u[5].w <= 0.5` → analytical closed-form rounded box (`u[1].z` is the
// corner radius). `u[5].w > 0.5` → the baked SDF texture (`u[1].z` is the decode scale; a sampled lens
// has no corner, so the slot is reused). Everything that reads the distance is identical either way.
fn fieldDistance(u: array<vec4<f32>, 6>, p: vec2<f32>) -> f32 {
    if (u[5].w > 0.5) {
        let uv = p / (2.0 * u[1].xy) + vec2<f32>(0.5, 0.5);
        return (textureSampleLevel(fieldTex, fieldSamp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r - 0.5) * u[1].z;
    }
    return sdfRoundedBox(p, u[1].xy, min(u[1].z, min(u[1].x, u[1].y)));
}

fn fieldRamp(d: f32, edge: f32) -> f32 {
    return clamp(-d / edge, 0.0, 1.0);
}
fn fieldProfile(x: f32, kind: i32) -> f32 {
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
fn fieldProfileSlope(x: f32, kind: i32) -> f32 {
    let delta = 0.001;
    return (fieldProfile(min(1.0, x + delta), kind) - fieldProfile(max(0.0, x - delta), kind)) / (2.0 * delta);
}
fn fieldCoverage(d: f32, softness: f32) -> f32 {
    return smoothstep(0.0, softness, -d);
}
fn fieldRadialDirection(localPos: vec2<f32>, halfSize: vec2<f32>, splay: f32, tilt: f32) -> vec2<f32> {
    let radialDir = normalize(localPos / max(vec2<f32>(1.0), halfSize));
    let flatDir = vec2<f32>(cos(tilt), sin(tilt));
    let blended = mix(flatDir, radialDir, splay);
    let l = length(blended);
    if (l > 0.001) { return blended / l; }
    return vec2<f32>(0.0);
}
fn fieldSnell(theta1: f32, n1: f32, n2: f32) -> f32 {
    let s = (n1 / n2) * sin(theta1);
    if (abs(s) > 1.0) { return -1.0; }
    return asin(s);
}
fn fieldRefract(t: f32, thick: f32, n2: f32, kind: i32) -> f32 {
    if (t <= 0.0 || t >= 1.0) { return 0.0; }
    let h = fieldProfile(t, kind) * thick;
    let dh = fieldProfileSlope(t, kind) * thick;
    let sA = atan(dh);
    let tI = abs(sA);
    let tR = fieldSnell(tI, 1.0, n2);
    if (tR < 0.0) { return 0.0; }
    return (h * tan(tR) - h * tan(tI)) * sign(dh);
}
fn fieldBand(x: f32, centre: f32, width: f32) -> f32 {
    return exp(-0.5 * pow((x - centre) / max(width, 1e-4), 2.0));
}
fn unitSpecular(t: f32, bezel: f32, lightAngle: f32, dir: vec2<f32>, scale: f32) -> f32 {
    if (t <= 0.0 || t >= 1.0) { return 0.0; }
    let band = fieldBand(t * bezel, 2.0 * scale, scale);
    let ld = vec2<f32>(cos(lightAngle), sin(lightAngle));
    var f = abs(dot(dir, ld));
    f = pow(f, 2.0);
    return band * f;
}

// The field at device pixel `fc`, packed as (displacement.x, displacement.y, specular, mask). One body
// for both sources — the distance is the only thing that changed.
fn computeField(u: array<vec4<f32>, 6>, fc: vec2<f32>) -> vec4<f32> {
    let scale = u[4].x;
    let localPos = fc - u[0].zw;
    let n0 = fieldDistance(u, localPos);
    if (n0 > 0.0) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    let edgeT = fieldRamp(n0, min(u[2].x, min(u[1].x, u[1].y)));
    let dir = fieldRadialDirection(localPos, u[1].xy, u[3].x, u[3].y);
    let refracted = fieldRefract(edgeT, u[2].y, u[2].z, i32(u[1].w));
    let mask = fieldCoverage(n0, 1.5 * u[4].x);
    let bezel = min(u[2].x, min(u[1].x, u[1].y));
    var disp = refracted * scale;
    let edgeFade = pow(1.0 - edgeT, 1.5);
    disp = disp * (1.0 + u[3].z * edgeFade);
    var dpx = dir * disp;
    let zoomFactor = 1.0 / max(u[3].w, 0.1) - 1.0;
    dpx = dpx + localPos * zoomFactor;
    let specular = unitSpecular(edgeT, bezel, u[2].w, dir, scale);
    return vec4<f32>(dpx.x, dpx.y, specular, mask);
}

// ===================================================================================================
// UNITS — heads (read other pixels), the pointwise tail, and the composite. Source surfaces are the
// module bindings fine already declares (`base_in`/`draft_in`/`input_in`), selected by `bits` in
// uniform control flow. `fx_bilin` reads `base_in`; `fx_bilin_input` reads `input_in`.
// ===================================================================================================

struct FxHead { value: vec4<f32>, orig: vec4<f32>, cov: f32 }

// WARP: masked displaced read of the materialized backdrop with chromatic aberration — lens refraction.
fn fx_head_warp(field: vec4<f32>, u: array<vec4<f32>, 6>, px: vec2<f32>) -> FxHead {
    let ipx = vec2<i32>(i32(px.x), i32(px.y));
    let bp = px + field.xy;
    let dlen = length(field.xy);
    let castr = smoothstep(0.0, 5.0 * u[4].x, dlen);
    var cadir = vec2<f32>(0.0, 0.0);
    if (dlen > 0.01 * u[4].x) { cadir = field.xy / dlen; }
    let cashift = cadir * u[4].y * castr;
    let cr = fx_bilin(bp - cashift);
    let cg = fx_bilin(bp);
    let cb = fx_bilin(bp + cashift);
    let value = vec4<f32>(cr.r, cg.g, cb.b, cg.a);
    let orig = textureLoad(base_in, ipx, 0);
    return FxHead(value, orig, -1.0); // cov = -1: keep the caller's rasterized silhouette coverage
}

// SCATTER: jittered read of the previous link's blurred surface — the frost head of the final frost arm
// (shade + mask-mix run pointwise after it, so orig = the original backdrop for the mask-mix).
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

// BLUR: one axis of a separable Gaussian (axis u[0].xy, sigma u[0].z). V reads the draft, H reads the
// source; writes UNMASKED (cov = 1). `srgb`/`shadow_edge` mirror bits 1024/2048.
fn fx_head_blur(u: array<vec4<f32>, 6>, bits: u32, px: vec2<f32>, is_v: bool, base_color: u32) -> FxHead {
    let sigma = max(u[0].z, 0.5);
    let radius = i32(ceil(3.0 * sigma));
    let ipx = vec2<i32>(i32(px.x), i32(px.y));
    let axis = vec2<i32>(i32(u[0].x), i32(u[0].y));
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let srgb = (bits & FX_SRGB) != 0u;
    let shadow_edge = (bits & FX_SHADOW_EDGE) != 0u;
    let bgraw = unpack4x8unorm(base_color);
    let bg = select(select(fx_premul_srgb_to_lin(bgraw), bgraw, srgb), vec4<f32>(0.0), shadow_edge);
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

// The pointwise tail — the field-measuring and colour units, in order. `value` is the head's result,
// `orig` the confine-against surface, `field` the computeField for this pixel.
fn fx_tail(bits: u32, value0: vec4<f32>, orig: vec4<f32>, field: vec4<f32>, u: array<vec4<f32>, 6>) -> vec4<f32> {
    var value = value0;
    if ((bits & FX_SHADE) != 0u) {
        let specular = field.b;
        let specularOpacity = u[4].w;
        let specularSaturation = u[5].x;
        let specLuma = dot(value.rgb, vec3<f32>(0.299, 0.587, 0.114));
        var saturated = mix(vec3<f32>(specLuma), value.rgb, 1.0 + specularSaturation);
        saturated = max(saturated, vec3<f32>(0.0));
        let highlightColor = mix(vec3<f32>(1.0, 0.98, 0.95), saturated, min(specularSaturation / 9.0, 1.0));
        value = vec4<f32>(value.rgb + specular * specularOpacity * highlightColor * value.a, value.a);
    }
    if ((bits & FX_CLIP) != 0u) {
        value = mix(value, value * value.a, u[5].y);
    }
    if ((bits & FX_TINT) != 0u) {
        let tintColor = u[3];
        let tinted = vec4<f32>(tintColor.rgb * tintColor.a, tintColor.a) * value.a;
        value = mix(value, tinted, select(0.0, 1.0, tintColor.a >= 0.0));
    }
    if ((bits & FX_ERASE) != 0u) {
        value = value * (1.0 - orig.a * u[3].w);
    }
    if ((bits & FX_MASKMIX) != 0u) {
        let mask = field.a;
        value = vec4<f32>(mix(orig.rgb, value.rgb, mask), mix(orig.a, value.a, mask));
    }
    return value;
}

// COMPOSITE — SPREAD (a straight colour source-over) or a masked mix into the accumulator pixel `dst`.
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

// THE ARM — one call replacing the whole CMD_EFFECT interpreter body. The field is evaluated once
// (only the units that read it use it); the head is picked by `bits`; the tail + composite finish.
fn fx_effect_arm(bits: u32, u: array<vec4<f32>, 6>, px: vec2<f32>, dst: vec4<f32>, area_cov: f32, base_color: u32) -> vec4<f32> {
    let field = computeField(u, px + vec2<f32>(0.5, 0.5));

    var head: FxHead;
    if ((bits & FX_WARP) != 0u) {
        head = fx_head_warp(field, u, px);
    } else if ((bits & FX_SCATTER) != 0u) {
        head = fx_head_scatter(u, px);
    } else if ((bits & FX_BLUR) != 0u) {
        let is_v = u[0].y > 0.5;
        head = fx_head_blur(u, bits, px, is_v, base_color);
    } else {
        head = FxHead(dst, dst, -1.0);
    }

    var cov = select(area_cov, head.cov, head.cov >= 0.0);
    if ((bits & FX_MATERIALIZE) != 0u) { cov = 1.0; }

    let eff = fx_tail(bits, head.value, head.orig, field, u);
    return fx_composite(bits, dst, eff, head.value, cov, u);
}
