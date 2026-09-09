// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Load rendering shaders.

#[cfg(feature = "wgpu")]
use wgpu::Device;

use crate::ShaderId;

#[cfg(feature = "wgpu")]
use crate::{
    Error, RendererOptions,
    recording::{BindType, ImageFormat},
    wgpu_engine::WgpuEngine,
};

// Shaders for the full pipeline
pub struct FullShaders {
    pub pathtag_reduce: ShaderId,
    pub pathtag_reduce2: ShaderId,
    pub pathtag_scan1: ShaderId,
    pub pathtag_scan: ShaderId,
    pub pathtag_scan_large: ShaderId,
    pub bbox_clear: ShaderId,
    pub flatten: ShaderId,
    pub draw_reduce: ShaderId,
    pub draw_leaf: ShaderId,
    pub clip_reduce: ShaderId,
    pub clip_leaf: ShaderId,
    pub binning: ShaderId,
    pub tile_alloc: ShaderId,
    pub backdrop: ShaderId,
    pub path_count_setup: ShaderId,
    pub path_count: ShaderId,
    pub coarse: ShaderId,
    pub path_tiling_setup: ShaderId,
    pub path_tiling: ShaderId,
    pub fine_area: Option<ShaderId>,
    /// `fine_area` with `load_base`: composites over a `base_in` texture (the previous phase's output)
    /// instead of clearing — the whole-viewport gather phasing.
    pub fine_area_load: Option<ShaderId>,
    /// `fine_area_load` with `have_draft`: a second sampled input `draft_in` at binding 10 — a
    /// separable blur's V pass samples the H pass's unmasked result here while `base_in` holds the
    /// original backdrop, so the silhouette mask applies exactly once.
    pub fine_area_load_draft: Option<ShaderId>,
    /// `fine_area_load` with `have_input`: a second sampled input `input_in` at binding 10 (sharing the
    /// slot with `draft_in`, since the two are never bound together) — a chained gather (frosted glass:
    /// warp → blur → scatter → shade) routes the previous link's materialised surface here while
    /// `base_in` holds the original backdrop.
    pub fine_area_load_input: Option<ShaderId>,
    /// `fine_area` with `rw_accum`: the output is bound READ-WRITE and updated in place — one
    /// accumulator, no ping-pong; a tile with no work in the dispatch window returns untouched.
    /// Only valid on devices with rgba8unorm read-write storage.
    pub fine_area_rw: Option<ShaderId>,
    /// `fine_area` with `acc_u32`: output is the PACKED accumulator (`r32uint`, one `pack4x8unorm`
    /// texel per pixel) bound read-write — the seed window (clears to the base color, only writes).
    /// r32uint read-write storage is core WebGPU: no adapter-specific features on any platform.
    pub fine_area_u: Option<ShaderId>,
    /// `fine_area` with `draft_clear`: a RASTERIZE window — transparent init, rgba8 draft output,
    /// no base or slot-10 bindings; the window's fenced draws paint a silhouette into its lease.
    pub fine_area_draft: Option<ShaderId>,
    /// `fine_area_u` + `rw_accum load_base base_u32`: in-place composite over the packed accumulator
    /// (own-pixel read-modify-write), with `base_in` a packed SNAPSHOT of the accumulator for the
    /// arms' backdrop reads (orig, escaped taps, fused warps).
    /// Rect copy accumulator -> snapshot inside a compute pass (the batched blit).
    pub snap_copy: Option<ShaderId>,
    pub fine_area_rwu: Option<ShaderId>,
    /// `fine_area_rwu` + `have_input`: a chained gather's composite (binding 10 = the previous
    /// link's materialised surface).
    pub fine_area_rwu_input: Option<ShaderId>,
    /// `fine_area_rwu` + `have_draft`: a separable blur's V-at-composite (binding 10 = the H draft).
    pub fine_area_rwu_draft: Option<ShaderId>,
    /// `fine_area_rwu_input` + `input_u32`: the chained input at binding 10 is a packed r32uint
    /// staging lease instead of an rgba8 draft.
    pub fine_area_rwu_input_pk: Option<ShaderId>,
    /// `fine_area_rwu_draft` + `input_u32`.
    pub fine_area_rwu_draft_pk: Option<ShaderId>,
    /// `fine_area_load` with `base_u32`: a materialize window (rgba8 draft output) whose backdrop
    /// is the packed snapshot.
    pub fine_area_loadu: Option<ShaderId>,
    /// `fine_area_loadu` without `region_reads`: the region STORE window — reads the packed
    /// staging store as `base_in` and writes the region atlas, so the atlas must not also be
    /// bound as the route atlas.
    pub fine_area_loadu_store: Option<ShaderId>,
    /// `fine_area_loadu` + `have_input`.
    pub fine_area_loadu_input: Option<ShaderId>,
    /// `fine_area_loadu` + `have_draft`.
    pub fine_area_loadu_draft: Option<ShaderId>,
    /// `staging_rw` + `draft_clear`: a rasterize window writing the packed staging store.
    pub fine_area_stg: Option<ShaderId>,
    /// `staging_rw` + `load_base base_u32`: a materialize window over the packed accumulator
    /// backdrop, output and value reads through the packed staging store.
    pub fine_area_stg_load: Option<ShaderId>,
    /// `fine_area_stg_load` + `have_input`: slot 10 binds a sampled (SDF) texture.
    pub fine_area_stg_load_sdf: Option<ShaderId>,
    /// `staging_rw` + `base_u32 stg_taps`: a chain materialize — taps and value ride the packed
    /// staging store, the packed accumulator is the read-only base for backdrop-edge taps and orig.
    pub fine_area_stg_chain: Option<ShaderId>,
    /// `fine_area_stg_chain` + `have_draft` (the separable-blur cov semantics).
    pub fine_area_stg_chain_draft: Option<ShaderId>,
    pub fine_msaa8: Option<ShaderId>,
    pub fine_msaa16: Option<ShaderId>,
    // 2-level dispatch works for CPU pathtag scan even for large
    // inputs, 3-level is not yet implemented.
    pub pathtag_is_cpu: bool,
}

#[cfg(feature = "wgpu")]
pub(crate) fn full_shaders(
    device: &Device,
    engine: &mut WgpuEngine,
    options: &RendererOptions,
) -> Result<FullShaders, Error> {
    use crate::wgpu_engine::CpuShaderType;
    use BindType::*;

    let mut force_gpu = false;
    let force_gpu_from: Option<&str> = None;
    // Uncomment this to force use of GPU shaders from the specified shader and later even
    // if `engine.use_cpu` is specified.
    //let force_gpu_from = Some("binning");

    #[cfg(feature = "hot_reload")]
    let mut shaders = vello_shaders::compile::ShaderInfo::from_default()?;
    #[cfg(not(feature = "hot_reload"))]
    let shaders = vello_shaders::SHADERS;

    macro_rules! add_shader {
        ($name:ident, $label:expr, $bindings:expr, $cpu:expr) => {{
            if force_gpu_from == Some(stringify!($name)) {
                force_gpu = true;
            }
            #[cfg(feature = "hot_reload")]
            let source = shaders
                .remove(stringify!($name))
                .expect(stringify!($name))
                .source
                .into();
            #[cfg(not(feature = "hot_reload"))]
            let source = shaders.$name.wgsl.code;
            engine.add_compute_shader(
                device,
                concat!("vello.", $label),
                source,
                &$bindings,
                if force_gpu {
                    CpuShaderType::Missing
                } else {
                    $cpu
                },
            )
        }};
        ($name:ident, $bindings:expr, $cpu:expr) => {{ add_shader!($name, stringify!($name), $bindings, $cpu) }};
        ($name:ident, $bindings:expr) => {
            add_shader!(
                $name,
                $bindings,
                CpuShaderType::Present(vello_shaders::cpu::$name)
            )
        };
    }

    let pathtag_reduce = add_shader!(pathtag_reduce, [Uniform, BufReadOnly, Buffer]);
    let pathtag_reduce2 = add_shader!(
        pathtag_reduce2,
        [BufReadOnly, Buffer],
        CpuShaderType::Skipped
    );
    let pathtag_scan1 = add_shader!(
        pathtag_scan1,
        [BufReadOnly, BufReadOnly, Buffer],
        CpuShaderType::Skipped
    );
    let pathtag_scan = add_shader!(
        pathtag_scan_small,
        [Uniform, BufReadOnly, BufReadOnly, Buffer],
        CpuShaderType::Present(vello_shaders::cpu::pathtag_scan)
    );
    let pathtag_scan_large = add_shader!(
        pathtag_scan_large,
        [Uniform, BufReadOnly, BufReadOnly, Buffer],
        CpuShaderType::Skipped
    );
    let bbox_clear = add_shader!(bbox_clear, [Uniform, Buffer]);
    let flatten = add_shader!(
        flatten,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let draw_reduce = add_shader!(draw_reduce, [Uniform, BufReadOnly, Buffer]);
    let draw_leaf = add_shader!(
        draw_leaf,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ]
    );
    let clip_reduce = add_shader!(clip_reduce, [BufReadOnly, BufReadOnly, Buffer, Buffer]);
    let clip_leaf = add_shader!(
        clip_leaf,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
        ]
    );
    let binning = add_shader!(
        binning,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
            Buffer,
        ]
    );
    let tile_alloc = add_shader!(
        tile_alloc,
        [Uniform, BufReadOnly, BufReadOnly, Buffer, Buffer, Buffer]
    );
    let path_count_setup = add_shader!(path_count_setup, [Buffer, Buffer]);
    let path_count = add_shader!(
        path_count,
        [Uniform, Buffer, BufReadOnly, BufReadOnly, Buffer, Buffer]
    );
    let backdrop = add_shader!(
        backdrop_dyn,
        [Uniform, Buffer, BufReadOnly, Buffer],
        CpuShaderType::Present(vello_shaders::cpu::backdrop)
    );
    let coarse = add_shader!(
        coarse,
        [
            Uniform,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
            Buffer,
            Buffer,
        ]
    );
    let path_tiling_setup = add_shader!(path_tiling_setup, [Buffer, Buffer, Buffer]);
    let path_tiling = add_shader!(
        path_tiling,
        [
            Buffer,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            BufReadOnly,
            Buffer,
        ]
    );
    let fine_resources = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        // Effects-in-fine chain descriptors (binding 8, every permutation).
        BufReadOnly,
        // Mask LUT buffer, used only when MSAA is enabled.
        BufReadOnly,
    ];
    // `fine_area_load`: the area bindings (no mask LUT) plus `base_in` at binding 8 — the previous
    // phase's output that this fine phase composites over (whole-viewport gather phasing).
    let fine_resources_load = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        // Effects-in-fine chain descriptors (binding 8); base_in follows at binding 9.
        BufReadOnly,
        ImageRead(ImageFormat::Rgba8),
    ];
    // `fine_area_load_draft`: `fine_area_load` plus a second sampled input `draft_in` at binding 10 —
    // a separable blur's V pass reads its H pass's UNMASKED result from here while `base_in` still
    // holds the original backdrop, so the silhouette mask applies exactly once. Binding 11 is the
    // region atlas (`region_reads`), so escaped taps route on this permutation too — a layer
    // blur's chain rides it.
    let fine_resources_load_draft = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
    ];

    let aa_support = &options.antialiasing_support;
    let fine_area = if aa_support.area {
        Some(add_shader!(
            fine_area,
            fine_resources[..fine_resources.len() - 1],
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_area_load = if aa_support.area {
        Some(add_shader!(
            fine_area_load,
            fine_resources_load,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_area_load_draft = if aa_support.area {
        Some(add_shader!(
            fine_area_load_draft,
            fine_resources_load_draft,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    // `fine_area_load_input`: same bindings as `fine_area_load_draft` (base_in at 9, a second sampled
    // input at 10) — the shader interprets binding 10 as `input_in` under the `have_input` define.
    let fine_area_load_input = if aa_support.area {
        Some(add_shader!(
            fine_area_load_input,
            fine_resources_load_draft,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    // `fine_area_rw`: the area bindings with the output image bound read-write (no `base_in`, no
    // mask LUT) — the single-accumulator whole-viewport path.
    let fine_resources_rw = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        // Effects-in-fine chain descriptors (binding 8).
        BufReadOnly,
    ];
    // Building the pipeline eagerly creates its bind group layout, and a ReadWrite rgba8unorm
    // storage entry is a validation error on a device without adapter-specific format features —
    // so the permutation only exists when the device creator requested them (which implies the
    // adapter supports rgba8unorm read-write on every platform this renderer targets).
    let fine_area_rw = if aa_support.area
        && device
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
    {
        Some(add_shader!(
            fine_area_rw,
            fine_resources_rw,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    // The packed-accumulator permutation family: r32uint read-write output (core WebGPU — no
    // feature gate) and/or an r32uint snapshot backdrop. Draft inputs stay rgba8.
    let fine_resources_u = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
    ];
    let fine_resources_rwu = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_resources_rwu_two = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_resources_loadu = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_resources_loadu_two = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_resources_draft = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
    ];
    let area = aa_support.area;
    let fine_area_u = area.then(|| add_shader!(fine_area_u, fine_resources_u, CpuShaderType::Missing));
    let fine_area_draft =
        area.then(|| add_shader!(fine_area_draft, fine_resources_draft, CpuShaderType::Missing));
    let snap_copy = area.then(|| {
        add_shader!(
            snap_copy,
            [BufReadOnly, ImageRead(ImageFormat::R32Uint), Image(ImageFormat::R32Uint)],
            CpuShaderType::Missing
        )
    });
    let fine_area_rwu = area.then(|| add_shader!(fine_area_rwu, fine_resources_rwu, CpuShaderType::Missing));
    let fine_area_rwu_input =
        area.then(|| add_shader!(fine_area_rwu_input, fine_resources_rwu_two, CpuShaderType::Missing));
    let fine_area_rwu_draft =
        area.then(|| add_shader!(fine_area_rwu_draft, fine_resources_rwu_two, CpuShaderType::Missing));
    let fine_resources_rwu_two_pk = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_area_rwu_input_pk = area
        .then(|| add_shader!(fine_area_rwu_input_pk, fine_resources_rwu_two_pk, CpuShaderType::Missing));
    let fine_area_rwu_draft_pk = area
        .then(|| add_shader!(fine_area_rwu_draft_pk, fine_resources_rwu_two_pk, CpuShaderType::Missing));
    let fine_area_loadu = area.then(|| add_shader!(fine_area_loadu, fine_resources_loadu, CpuShaderType::Missing));
    let fine_resources_loadu_store = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        Image(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
    ];
    let fine_area_loadu_store = area
        .then(|| add_shader!(fine_area_loadu_store, fine_resources_loadu_store, CpuShaderType::Missing));
    let fine_area_loadu_input =
        area.then(|| add_shader!(fine_area_loadu_input, fine_resources_loadu_two, CpuShaderType::Missing));
    let fine_area_loadu_draft =
        area.then(|| add_shader!(fine_area_loadu_draft, fine_resources_loadu_two, CpuShaderType::Missing));
    let fine_resources_stg = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
    ];
    let fine_resources_stg_load = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_resources_stg_load_sdf = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_resources_stg_chain = [
        Uniform,
        BufReadOnly,
        BufReadOnly,
        BufReadOnly,
        Buffer,
        ImageReadWrite(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
        ImageRead(ImageFormat::Rgba8),
        BufReadOnly,
        ImageRead(ImageFormat::R32Uint),
        ImageRead(ImageFormat::Rgba8),
    ];
    let fine_area_stg =
        area.then(|| add_shader!(fine_area_stg, fine_resources_stg, CpuShaderType::Missing));
    let fine_area_stg_load =
        area.then(|| add_shader!(fine_area_stg_load, fine_resources_stg_load, CpuShaderType::Missing));
    let fine_area_stg_load_sdf =
        area.then(|| add_shader!(fine_area_stg_load_sdf, fine_resources_stg_load_sdf, CpuShaderType::Missing));
    let fine_area_stg_chain =
        area.then(|| add_shader!(fine_area_stg_chain, fine_resources_stg_chain, CpuShaderType::Missing));
    let fine_area_stg_chain_draft =
        area.then(|| add_shader!(fine_area_stg_chain_draft, fine_resources_stg_chain, CpuShaderType::Missing));
    let fine_msaa8 = if aa_support.msaa8 {
        Some(add_shader!(
            fine_msaa8,
            fine_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };
    let fine_msaa16 = if aa_support.msaa16 {
        Some(add_shader!(
            fine_msaa16,
            fine_resources,
            CpuShaderType::Missing
        ))
    } else {
        None
    };

    Ok(FullShaders {
        pathtag_reduce,
        pathtag_reduce2,
        pathtag_scan,
        pathtag_scan1,
        pathtag_scan_large,
        bbox_clear,
        flatten,
        draw_reduce,
        draw_leaf,
        clip_reduce,
        clip_leaf,
        binning,
        tile_alloc,
        path_count_setup,
        path_count,
        backdrop,
        coarse,
        path_tiling_setup,
        path_tiling,
        fine_area,
        fine_area_load,
        fine_area_load_draft,
        fine_area_load_input,
        fine_area_rw,
        fine_area_u,
        fine_area_draft,
        snap_copy,
        fine_area_rwu,
        fine_area_rwu_input,
        fine_area_rwu_draft,
        fine_area_rwu_input_pk,
        fine_area_rwu_draft_pk,
        fine_area_loadu,
        fine_area_loadu_store,
        fine_area_loadu_input,
        fine_area_loadu_draft,
        fine_area_stg,
        fine_area_stg_load,
        fine_area_stg_load_sdf,
        fine_area_stg_chain,
        fine_area_stg_chain_draft,
        fine_msaa8,
        fine_msaa16,
        pathtag_is_cpu: options.use_cpu,
    })
}
