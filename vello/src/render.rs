// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Take an encoded scene and create a graph to render it

use crate::recording::{BufferProxy, ImageFormat, ImageProxy, Recording, ResourceProxy};
use crate::shaders::FullShaders;
use crate::{AaConfig, RenderParams};

#[cfg(feature = "wgpu")]
use crate::Scene;

use vello_encoding::{Encoding, Resolver, WorkgroupSize, make_mask_lut, make_mask_lut_16};

#[derive(Clone, Copy, Debug)]
enum AtlasProxyAction {
    Created,
    Reused,
    Resized,
}

/// State for a render in progress.
pub struct Render {
    fine_wg_count: Option<WorkgroupSize>,
    fine_resources: Option<FineResources>,
    mask_buf: Option<ResourceProxy>,

    #[cfg(feature = "debug_layers")]
    captured_buffers: Option<CapturedBuffers>,
}

#[cfg(feature = "debug_layers")]
impl Drop for Render {
    fn drop(&mut self) {
        if self.captured_buffers.is_some() {
            unreachable!("Render captured buffers without freeing them");
        }
    }
}

/// Resources produced by pipeline, needed for fine rasterization.
struct FineResources {
    aa_config: AaConfig,

    config_buf: ResourceProxy,
    bump_buf: ResourceProxy,
    tile_buf: ResourceProxy,
    segments_buf: ResourceProxy,
    ptcl_buf: ResourceProxy,
    gradient_image: ResourceProxy,
    info_bin_data_buf: ResourceProxy,
    image_atlas: ResourceProxy,
    blend_spill_buf: ResourceProxy,
    effect_params_buf: ResourceProxy,

    out_image: ImageProxy,
}

/// A collection of internal buffers that are used for debug visualization when the
/// `debug_layers` feature is enabled. The contents of these buffers remain GPU resident
/// and must be freed directly by the caller.
///
/// Some of these buffers are also scheduled for a download to allow their contents to be
/// processed for CPU-side validation. These buffers are documented as such.
#[cfg(feature = "debug_layers")]
pub struct CapturedBuffers {
    pub sizes: vello_encoding::BufferSizes,

    /// Buffers that remain GPU-only
    pub path_bboxes: BufferProxy,

    /// Buffers scheduled for download
    pub lines: BufferProxy,
}

#[cfg(feature = "debug_layers")]
impl CapturedBuffers {
    pub fn release_buffers(self, recording: &mut Recording) {
        recording.free_buffer(self.path_bboxes);
        recording.free_buffer(self.lines);
    }
}

#[cfg(feature = "wgpu")]
pub(crate) fn render_full(
    scene: &Scene,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    image_atlas: &mut Option<ImageProxy>,
    params: &RenderParams,
) -> (Recording, ResourceProxy) {
    render_encoding_full(scene.encoding(), resolver, shaders, image_atlas, params)
}

/// A **persistent** phased render, driven one phase at a time by the caller so it can interleave its
/// own work (a gather's blur/glass effect) *between* phases while still sharing the single setup.
///
/// The caller gets a recording per step, so it can run the front-end, then each fine window, into
/// the *same* encoder, recording its own passes in the gaps. vello's engine keeps the shared buffer proxies live
/// in its `bind_map` across those `run_recording_into` calls (nothing frees them until
/// [`record_phased_frees`]), so the geometry front-end and every buffer allocation are paid ONCE.
///
/// Area AA only.
#[cfg(feature = "wgpu")]
pub struct PhasedSession {
    cpu_config: vello_encoding::RenderConfig,
    scene_buf: ResourceProxy,
    config_full: ResourceProxy,
    gradient_image: ResourceProxy,
    image_atlas: ResourceProxy,
    info_bin_data_buf: ResourceProxy,
    tile_buf: ResourceProxy,
    segments_buf: ResourceProxy,
    ptcl_buf: ResourceProxy,
    tagmonoid_buf: ResourceProxy,
    path_bbox_buf: ResourceProxy,
    bump_buf: ResourceProxy,
    lines_buf: ResourceProxy,
    draw_reduced_buf: ResourceProxy,
    draw_monoid_buf: ResourceProxy,
    clip_inp_buf: ResourceProxy,
    clip_el_buf: ResourceProxy,
    clip_bic_buf: ResourceProxy,
    clip_bbox_buf: ResourceProxy,
    draw_bbox_buf: ResourceProxy,
    bin_header_buf: ResourceProxy,
    path_buf: ResourceProxy,
    seg_counts_buf: ResourceProxy,
    blend_spill_buf: ResourceProxy,
    effect_params_buf: ResourceProxy,
    width: u32,
    height: u32,
    /// `draw_reduce`/`draw_leaf`/clip read `path_bbox` (produced by the per-phase `flatten`), so they
    /// are recorded once inside the FIRST phase, after its flatten. This tracks whether that happened.
    did_draw_frontend: bool,
}

#[cfg(feature = "wgpu")]
impl PhasedSession {
    /// Set the reach-crop origins for the NEXT fine dispatch: `scratch_out` shifts where the producer
    /// writes `output`, `scratch_in` where the consumer samples its scratch (`draft`/`input`). Each
    /// `record_fine_packed` consumes them into its per-dispatch config and resets to `[0, 0]`, so a
    /// full-viewport call needs no set. `[0, 0]` for a slot means that slot is not cropped.
    pub fn set_scratch_origins(&mut self, scratch_out: [u32; 2], scratch_in: [u32; 2]) {
        self.cpu_config.gpu.scratch_out = scratch_out;
        self.cpu_config.gpu.scratch_in = scratch_in;
    }

    /// Set the sparse tile list for the NEXT fine dispatch: the grid becomes `(n, 1, 1)` workgroups
    /// and workgroup `i` reads its tile coordinate from `effect_params[base + i]` (packed
    /// `y<<16 | x`, biased by `0x40000000`). Each `record_fine_packed` consumes it into its
    /// per-dispatch config and resets to zero, so a full-viewport call needs no set.
    pub fn set_sparse(&mut self, base: u32, n: u32) {
        self.cpu_config.gpu.sparse_base = base;
        self.cpu_config.gpu.sparse_n = n;
    }

    /// Place the `base` slot for the NEXT fine dispatch inside the store: the rect at `at` holds
    /// the frame-space region `[org, org + ext)`. Consumed by that dispatch and reset.
    pub fn set_base_rect(&mut self, org: [u32; 2], ext: [u32; 2], at: [u32; 2]) {
        self.cpu_config.gpu.base_org = org;
        self.cpu_config.gpu.base_ext = ext;
        self.cpu_config.gpu.base_at = at;
    }

    /// A fresh `R32Uint` (packed-rgba8) image proxy sized to the frame — the in-place accumulator
    /// or its round-boundary snapshot.
    pub(crate) fn new_packed_image(&self) -> ImageProxy {
        ImageProxy::new(self.width, self.height, ImageFormat::R32Uint)
    }

    /// DEBUG: the bump buffer's resource id, for the post-frame diagnostics readback.
    pub fn debug_bump_proxy_id(&self) -> crate::recording::ResourceId {
        self.bump_buf.as_buf().unwrap().id
    }
}

/// Resolve the scene, allocate every shared buffer, and record the geometry front-end (pathtag
/// reduce/scan) that is draw-range-independent. Returns the session (holding the shared proxies) and
/// the front-end recording — the caller runs it into its encoder, then [`record_frontend_full`]
/// once and [`record_fine_packed`] per window.
#[cfg(feature = "wgpu")]
pub(crate) fn begin_phased(
    encoding: &Encoding,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    persistent_image_atlas: &mut Option<ImageProxy>,
    params: &RenderParams,
    effect_params: &[u8],
) -> (PhasedSession, Recording) {
    use vello_encoding::RenderConfig;
    assert!(
        matches!(params.antialiasing_method, AaConfig::Area),
        "begin_phased supports Area AA only"
    );

    let mut recording = Recording::default();
    let mut packed = vec![];
    let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
    let gradient_image = if ramps.height == 0 {
        ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
    } else {
        let data: &[u8] = bytemuck::cast_slice(ramps.data);
        ResourceProxy::Image(recording.upload_image(ramps.width, ramps.height, ImageFormat::Rgba8, data))
    };
    let atlas_width = images.width.max(1);
    let atlas_height = images.height.max(1);
    let image_atlas = match persistent_image_atlas {
        Some(proxy) if proxy.width == atlas_width && proxy.height == atlas_height => *proxy,
        Some(proxy) => {
            recording.free_image(*proxy);
            let new_proxy = ImageProxy::new(atlas_width, atlas_height, ImageFormat::Rgba8);
            *persistent_image_atlas = Some(new_proxy);
            new_proxy
        }
        None => {
            let proxy = ImageProxy::new(atlas_width, atlas_height, ImageFormat::Rgba8);
            *persistent_image_atlas = Some(proxy);
            proxy
        }
    };
    for image in images.images {
        recording.write_image(image_atlas, image.1, image.2, image.0.clone());
    }
    let image_atlas = ResourceProxy::Image(image_atlas);

    let mut cpu_config = RenderConfig::new(&layout, params.width, params.height, &params.base_color);
    crate::apply_frame_extent(&mut cpu_config.gpu);
    let buffer_sizes = &cpu_config.buffer_sizes;
    let wg_counts = &cpu_config.workgroup_counts;

    if packed.is_empty() {
        packed.resize(size_of::<u32>(), u8::MAX);
    }
    let scene_buf = ResourceProxy::Buffer(recording.upload("vello.scene", packed));
    let config_full =
        ResourceProxy::Buffer(recording.upload_uniform("vello.config", bytemuck::bytes_of(&cpu_config.gpu)));

    // Shared buffers — allocated ONCE, re-dispatched into by every phase.
    let info_bin_data_buf =
        ResourceProxy::new_buf(buffer_sizes.bin_data.size_in_bytes() as u64, "vello.info_bin_data_buf");
    let tile_buf = ResourceProxy::new_buf(buffer_sizes.tiles.size_in_bytes().into(), "vello.tile_buf");
    let segments_buf =
        ResourceProxy::new_buf(buffer_sizes.segments.size_in_bytes().into(), "vello.segments_buf");
    let ptcl_buf = ResourceProxy::new_buf(buffer_sizes.ptcl.size_in_bytes().into(), "vello.ptcl_buf");
    let tagmonoid_buf =
        ResourceProxy::new_buf(buffer_sizes.path_monoids.size_in_bytes().into(), "vello.tagmonoid_buf");
    let path_bbox_buf =
        ResourceProxy::new_buf(buffer_sizes.path_bboxes.size_in_bytes().into(), "vello.path_bbox_buf");
    let bump_buf =
        ResourceProxy::Buffer(BufferProxy::new(buffer_sizes.bump_alloc.size_in_bytes().into(), "vello.bump_buf"));
    let lines_buf = ResourceProxy::new_buf(buffer_sizes.lines.size_in_bytes().into(), "vello.lines_buf");
    let draw_reduced_buf =
        ResourceProxy::new_buf(buffer_sizes.draw_reduced.size_in_bytes().into(), "vello.draw_reduced_buf");
    let draw_monoid_buf =
        ResourceProxy::new_buf(buffer_sizes.draw_monoids.size_in_bytes().into(), "vello.draw_monoid_buf");
    let clip_inp_buf =
        ResourceProxy::new_buf(buffer_sizes.clip_inps.size_in_bytes().into(), "vello.clip_inp_buf");
    let clip_el_buf =
        ResourceProxy::new_buf(buffer_sizes.clip_els.size_in_bytes().into(), "vello.clip_el_buf");
    let clip_bic_buf =
        ResourceProxy::new_buf(buffer_sizes.clip_bics.size_in_bytes().into(), "vello.clip_bic_buf");
    let clip_bbox_buf =
        ResourceProxy::new_buf(buffer_sizes.clip_bboxes.size_in_bytes().into(), "vello.clip_bbox_buf");
    let draw_bbox_buf =
        ResourceProxy::new_buf(buffer_sizes.draw_bboxes.size_in_bytes().into(), "vello.draw_bbox_buf");
    let bin_header_buf =
        ResourceProxy::new_buf(buffer_sizes.bin_headers.size_in_bytes().into(), "vello.bin_header_buf");
    let path_buf = ResourceProxy::new_buf(buffer_sizes.paths.size_in_bytes().into(), "vello.path_buf");
    let seg_counts_buf =
        ResourceProxy::new_buf(buffer_sizes.seg_counts.size_in_bytes().into(), "vello.seg_counts_buf");
    let blend_spill_buf =
        ResourceProxy::Buffer(BufferProxy::new(buffer_sizes.blend_spill.size_in_bytes().into(), "vello.blend_spill"));
    // Effects-in-fine: upload this frame's per-effect chain descriptors (empty → a zeroed dummy so the
    // binding is always valid). fine reads them at each inline CMD_EFFECT marker via its `p2` offset.
    let ep_bytes: Vec<u8> = if effect_params.is_empty() { vec![0u8; 256] } else { effect_params.to_vec() };
    let effect_params_buf: ResourceProxy = recording.upload("vello.effect_params", ep_bytes).into();

    // Shared front-end (once): pathtag reduce/scan. draw/clip are deferred into the first phase
    // (they read `path_bbox`, produced by that phase's flatten).
    let reduced_buf =
        ResourceProxy::new_buf(buffer_sizes.path_reduced.size_in_bytes().into(), "vello.reduced_buf");
    recording.dispatch(shaders.pathtag_reduce, wg_counts.path_reduce, [config_full, scene_buf, reduced_buf]);
    let mut pathtag_parent = reduced_buf;
    let mut large_pathtag_bufs = None;
    let use_large_path_scan = wg_counts.use_large_path_scan && !shaders.pathtag_is_cpu;
    if use_large_path_scan {
        let reduced2_buf =
            ResourceProxy::new_buf(buffer_sizes.path_reduced2.size_in_bytes().into(), "vello.reduced2_buf");
        recording.dispatch(shaders.pathtag_reduce2, wg_counts.path_reduce2, [reduced_buf, reduced2_buf]);
        let reduced_scan_buf =
            ResourceProxy::new_buf(buffer_sizes.path_reduced_scan.size_in_bytes().into(), "reduced_scan_buf");
        recording.dispatch(shaders.pathtag_scan1, wg_counts.path_scan1, [reduced_buf, reduced2_buf, reduced_scan_buf]);
        pathtag_parent = reduced_scan_buf;
        large_pathtag_bufs = Some((reduced2_buf, reduced_scan_buf));
    }
    let pathtag_scan = if use_large_path_scan { shaders.pathtag_scan_large } else { shaders.pathtag_scan };
    recording.dispatch(pathtag_scan, wg_counts.path_scan, [config_full, scene_buf, pathtag_parent, tagmonoid_buf]);
    recording.free_resource(reduced_buf);
    if let Some((reduced2, reduced_scan)) = large_pathtag_bufs {
        recording.free_resource(reduced2);
        recording.free_resource(reduced_scan);
    }

    let session = PhasedSession {
        cpu_config,
        scene_buf,
        config_full,
        gradient_image,
        image_atlas,
        info_bin_data_buf,
        tile_buf,
        segments_buf,
        ptcl_buf,
        tagmonoid_buf,
        path_bbox_buf,
        bump_buf,
        lines_buf,
        draw_reduced_buf,
        draw_monoid_buf,
        clip_inp_buf,
        clip_el_buf,
        clip_bic_buf,
        clip_bbox_buf,
        draw_bbox_buf,
        bin_header_buf,
        path_buf,
        seg_counts_buf,
        blend_spill_buf,
        effect_params_buf,
        width: params.width,
        height: params.height,
        did_draw_frontend: false,
    };
    (session, recording)
}

/// Record the whole scene's geometry front-end + tiling + coarse **once** over the full draw range,
/// building one complete PTCL with every `CMD_EFFECT` marker in place. This is the front-end-once
/// counterpart to a per-phase front-end: instead of re-running flatten/binning/coarse per phase narrowed to
/// a draw window, it runs them a single time (full range, via [`PhasedSession::config_full`]), then the
/// caller dispatches [`record_fine_packed`] once per window over this shared PTCL.
#[cfg(feature = "wgpu")]
pub(crate) fn record_frontend_full(session: &mut PhasedSession, shaders: &FullShaders) -> Recording {
    let buffer_sizes = &session.cpu_config.buffer_sizes;
    let wg_counts = &session.cpu_config.workgroup_counts;
    let mut recording = Recording::default();

    // The full-range config uploaded once in `begin_phased` (draw_start=0, draw_end=n_drawobj), so
    // binning includes every draw and coarse writes a PTCL spanning all segments. seg_target is unread
    // by the front-end (only `fine` gates on it).
    let config_buf = session.config_full;

    recording.clear_all(*session.bump_buf.as_buf().unwrap());
    recording.dispatch(shaders.bbox_clear, wg_counts.bbox_clear, [config_buf, session.path_bbox_buf]);
    recording.dispatch(
        shaders.flatten,
        wg_counts.flatten,
        [config_buf, session.scene_buf, session.tagmonoid_buf, session.path_bbox_buf, session.bump_buf, session.lines_buf],
    );
    // Draw front-end + clip: run once here (there is no later phase to defer to).
    recording.dispatch(shaders.draw_reduce, wg_counts.draw_reduce, [config_buf, session.scene_buf, session.draw_reduced_buf]);
    recording.dispatch(
        shaders.draw_leaf,
        wg_counts.draw_leaf,
        [config_buf, session.scene_buf, session.draw_reduced_buf, session.path_bbox_buf, session.draw_monoid_buf, session.info_bin_data_buf, session.clip_inp_buf],
    );
    recording.free_resource(session.draw_reduced_buf);
    if wg_counts.clip_reduce.0 > 0 {
        recording.dispatch(shaders.clip_reduce, wg_counts.clip_reduce, [session.clip_inp_buf, session.path_bbox_buf, session.clip_bic_buf, session.clip_el_buf]);
    }
    if wg_counts.clip_leaf.0 > 0 {
        recording.dispatch(
            shaders.clip_leaf,
            wg_counts.clip_leaf,
            [config_buf, session.clip_inp_buf, session.path_bbox_buf, session.clip_bic_buf, session.clip_el_buf, session.draw_monoid_buf, session.clip_bbox_buf],
        );
    }
    session.did_draw_frontend = true;
    recording.dispatch(
        shaders.binning,
        wg_counts.binning,
        [config_buf, session.draw_monoid_buf, session.path_bbox_buf, session.clip_bbox_buf, session.draw_bbox_buf, session.bump_buf, session.info_bin_data_buf, session.bin_header_buf],
    );
    recording.dispatch(
        shaders.tile_alloc,
        wg_counts.tile_alloc,
        [config_buf, session.scene_buf, session.draw_bbox_buf, session.bump_buf, session.path_buf, session.tile_buf],
    );
    let indirect_count_buf =
        BufferProxy::new(buffer_sizes.indirect_count.size_in_bytes().into(), "vello.indirect_count");
    recording.dispatch(shaders.path_count_setup, wg_counts.path_count_setup, [session.bump_buf, indirect_count_buf.into()]);
    recording.dispatch_indirect(
        shaders.path_count,
        indirect_count_buf,
        0,
        [config_buf, session.bump_buf, session.lines_buf, session.path_buf, session.tile_buf, session.seg_counts_buf],
    );
    recording.dispatch(shaders.backdrop, wg_counts.backdrop, [config_buf, session.bump_buf, session.path_buf, session.tile_buf]);
    recording.dispatch(
        shaders.coarse,
        wg_counts.coarse,
        [config_buf, session.scene_buf, session.draw_monoid_buf, session.bin_header_buf, session.info_bin_data_buf, session.path_buf, session.tile_buf, session.bump_buf, session.ptcl_buf],
    );
    recording.dispatch(shaders.path_tiling_setup, wg_counts.path_tiling_setup, [session.bump_buf, indirect_count_buf.into(), session.ptcl_buf]);
    recording.dispatch_indirect(
        shaders.path_tiling,
        indirect_count_buf,
        0,
        [session.bump_buf, session.seg_counts_buf, session.lines_buf, session.path_buf, session.tile_buf, session.segments_buf],
    );
    recording.free_buffer(indirect_count_buf);
    recording
}

/// The one effects `fine` dispatch: window `[seg_lo, seg_target)` of the shared PTCL, run by the
/// `fine_packed` permutation over the packed store `out` (r32uint, read-write). `mode` is the
/// dispatch's [`FINE_MODE`] word: how the tile's register is seeded, which of `base`/`input` are
/// bound, whether value and tap reads ride the store itself, and how marks key their window.
/// `base` and `input` are r32uint sampled images — a 1×1 dummy when the mode says a slot is
/// unbound. The per-dispatch scratch origins and sparse list are consumed here.
#[cfg(feature = "wgpu")]
pub(crate) fn record_fine_packed(
    session: &mut PhasedSession,
    shaders: &FullShaders,
    seg_lo: u32,
    seg_target: u32,
    mode: u32,
    base: ImageProxy,
    input: ImageProxy,
    out_image: ImageProxy,
) -> Recording {
    let shader = shaders.fine_packed.expect("the effects path needs the fine_packed shader");
    let mut recording = Recording::default();
    let mut seg_cfg = session.cpu_config.gpu;
    seg_cfg.seg_lo = seg_lo;
    seg_cfg.seg_target = seg_target;
    seg_cfg.fine_mode = mode;
    session.cpu_config.gpu.scratch_out = [0; 2];
    session.cpu_config.gpu.scratch_in = [0; 2];
    session.cpu_config.gpu.sparse_base = 0;
    session.cpu_config.gpu.sparse_n = 0;
    session.cpu_config.gpu.base_org = [0; 2];
    session.cpu_config.gpu.base_ext = [0; 2];
    session.cpu_config.gpu.base_at = [0; 2];
    let fine_wg = if seg_cfg.sparse_n != 0 {
        (seg_cfg.sparse_n, 1, 1)
    } else {
        session.cpu_config.workgroup_counts.fine
    };
    let config_buf =
        ResourceProxy::Buffer(recording.upload_uniform("vello.config.seg", bytemuck::bytes_of(&seg_cfg)));
    recording.dispatch(
        shader,
        fine_wg,
        [
            config_buf,
            session.segments_buf,
            session.ptcl_buf,
            session.info_bin_data_buf,
            session.blend_spill_buf,
            ResourceProxy::Image(out_image),
            session.gradient_image,
            session.image_atlas,
            session.effect_params_buf,
            ResourceProxy::Image(base),
            ResourceProxy::Image(input),
        ],
    );
    recording.free_resource(config_buf);
    recording
}

/// Rect copies accumulator -> snapshot as ONE compute dispatch (the batched form of the encoder
/// blits): the rects are decomposed into 16x16 tiles, uploaded as a `(tile_x, tile_y, clip_x1,
/// clip_y1)` table, and workgroup `i` copies tile `i` — a whole refresh costs one dispatch, not
/// one per rect.
#[cfg(feature = "wgpu")]
pub(crate) fn record_snap_copy(
    shaders: &FullShaders,
    rects: &[[u32; 4]],
    src: ImageProxy,
    dst: ImageProxy,
) -> Recording {
    let mut recording = Recording::default();
    let shader = shaders.snap_copy.expect("packed accumulator needs snap_copy");
    let mut tiles: Vec<[u32; 4]> = Vec::new();
    for r in rects {
        for ty in 0..(r[3] - r[1]).div_ceil(16) {
            for tx in 0..(r[2] - r[0]).div_ceil(16) {
                tiles.push([r[0] + tx * 16, r[1] + ty * 16, r[2], r[3]]);
            }
        }
    }
    if !tiles.is_empty() {
        let buf = ResourceProxy::Buffer(
            recording.upload("vello.snapcopy.tiles", bytemuck::cast_slice::<_, u8>(&tiles).to_vec()),
        );
        let wg = (u32::try_from(tiles.len()).expect("tile table fits u32"), 1, 1);
        recording.dispatch(shader, wg, [buf, ResourceProxy::Image(src), ResourceProxy::Image(dst)]);
        recording.free_resource(buf);
    }
    recording
}

/// Free every shared buffer once all phases are done. Returns a free-only recording; the caller runs
/// it into the encoder and the frees are deferred until after submit (as with every recording).
/// Mirrors the tail frees of [`render_encoding_phased`] (`draw_reduced_buf` is freed in the first
/// phase, not here).
#[cfg(feature = "wgpu")]
pub(crate) fn record_phased_frees(session: &PhasedSession) -> Recording {
    let mut recording = Recording::default();
    recording.free_resource(session.scene_buf);
    recording.free_resource(session.config_full);
    recording.free_resource(session.info_bin_data_buf);
    recording.free_resource(session.tile_buf);
    recording.free_resource(session.segments_buf);
    recording.free_resource(session.ptcl_buf);
    recording.free_resource(session.tagmonoid_buf);
    recording.free_resource(session.path_bbox_buf);
    recording.free_resource(session.bump_buf);
    recording.free_resource(session.lines_buf);
    recording.free_resource(session.draw_monoid_buf);
    recording.free_resource(session.clip_inp_buf);
    recording.free_resource(session.clip_el_buf);
    recording.free_resource(session.clip_bic_buf);
    recording.free_resource(session.clip_bbox_buf);
    recording.free_resource(session.draw_bbox_buf);
    recording.free_resource(session.bin_header_buf);
    recording.free_resource(session.path_buf);
    recording.free_resource(session.seg_counts_buf);
    recording.free_resource(session.blend_spill_buf);
    recording.free_resource(session.effect_params_buf);
    recording.free_resource(session.gradient_image);
    recording
}

#[cfg(feature = "wgpu")]
/// Create a single recording with both coarse and fine render stages.
///
/// This function is not recommended when the scene can be complex, as it does not
/// implement robust dynamic memory.
pub(crate) fn render_encoding_full(
    encoding: &Encoding,
    resolver: &mut Resolver,
    shaders: &FullShaders,
    image_atlas: &mut Option<ImageProxy>,
    params: &RenderParams,
) -> (Recording, ResourceProxy) {
    let mut render = Render::new();
    let mut recording =
        render.render_encoding_coarse(encoding, resolver, shaders, image_atlas, params, false);
    let out_image = render.out_image();
    render.record_fine(shaders, &mut recording);
    (recording, out_image.into())
}

impl Default for Render {
    fn default() -> Self {
        Self::new()
    }
}

impl Render {
    pub fn new() -> Self {
        Self {
            fine_wg_count: None,
            fine_resources: None,
            mask_buf: None,
            #[cfg(feature = "debug_layers")]
            captured_buffers: None,
        }
    }

    /// Prepare a recording for the coarse rasterization phase.
    ///
    /// The `robust` parameter controls whether we're preparing for readback
    /// of the atomic bump buffer, for robust dynamic memory.
    pub fn render_encoding_coarse(
        &mut self,
        encoding: &Encoding,
        resolver: &mut Resolver,
        shaders: &FullShaders,
        persistent_image_atlas: &mut Option<ImageProxy>,
        params: &RenderParams,
        robust: bool,
        ) -> Recording {
        use vello_encoding::RenderConfig;
        let mut recording = Recording::default();
        let mut packed = vec![];

        let (layout, ramps, images) = resolver.resolve(encoding, &mut packed);
        let gradient_image = if ramps.height == 0 {
            ResourceProxy::new_image(1, 1, ImageFormat::Rgba8)
        } else {
            let data: &[u8] = bytemuck::cast_slice(ramps.data);
            ResourceProxy::Image(recording.upload_image(
                ramps.width,
                ramps.height,
                ImageFormat::Rgba8,
                data,
            ))
        };
        let atlas_width = images.width.max(1);
        let atlas_height = images.height.max(1);
        let (image_atlas, atlas_proxy_action) = match persistent_image_atlas {
            Some(proxy) if proxy.width == atlas_width && proxy.height == atlas_height => {
                (*proxy, AtlasProxyAction::Reused)
            }
            Some(proxy) => {
                recording.free_image(*proxy);
                let new_proxy = ImageProxy::new(atlas_width, atlas_height, ImageFormat::Rgba8);
                *persistent_image_atlas = Some(new_proxy);
                (new_proxy, AtlasProxyAction::Resized)
            }
            None => {
                let proxy = ImageProxy::new(atlas_width, atlas_height, ImageFormat::Rgba8);
                *persistent_image_atlas = Some(proxy);
                (proxy, AtlasProxyAction::Created)
            }
        };
        if !matches!(atlas_proxy_action, AtlasProxyAction::Reused)
            || !images.images.is_empty()
            || images.evicted != 0
        {
            if images.evicted == 0 {
                log::debug!(
                    "image atlas {}x{}: {:?}, {} upload(s)",
                    atlas_width,
                    atlas_height,
                    atlas_proxy_action,
                    images.images.len()
                );
            } else {
                log::debug!(
                    "image atlas {}x{}: {:?}, {} upload(s), {} eviction(s)",
                    atlas_width,
                    atlas_height,
                    atlas_proxy_action,
                    images.images.len(),
                    images.evicted
                );
            }
        }
        for image in images.images {
            recording.write_image(image_atlas, image.1, image.2, image.0.clone());
        }
        let mut cpu_config =
            RenderConfig::new(&layout, params.width, params.height, &params.base_color);
        crate::apply_frame_extent(&mut cpu_config.gpu);
        // HACK: The coarse workgroup counts is the number of active bins.
        if (cpu_config.workgroup_counts.coarse.0
            * cpu_config.workgroup_counts.coarse.1
            * cpu_config.workgroup_counts.coarse.2)
            > 256
        {
            log::warn!(
                "Trying to paint too large image. {}x{}.\n\
                See https://github.com/linebender/vello/issues/680 for details",
                params.width,
                params.height
            );
        }
        let buffer_sizes = &cpu_config.buffer_sizes;
        let wg_counts = &cpu_config.workgroup_counts;

        if packed.is_empty() {
            // HACK: wgpu doesn't allow empty buffers, so we make sure that the scene buffer we upload
            // can contain at least one array item.
            // The values passed here should never be read, because the scene size in config
            // is zero.
            packed.resize(size_of::<u32>(), u8::MAX);
        }
        let scene_buf = ResourceProxy::Buffer(recording.upload("vello.scene", packed));
            let config_buf = ResourceProxy::Buffer(
            recording.upload_uniform("vello.config", bytemuck::bytes_of(&cpu_config.gpu)),
        );
        let info_bin_data_buf = ResourceProxy::new_buf(
            buffer_sizes.bin_data.size_in_bytes() as u64,
            "vello.info_bin_data_buf",
        );
        let tile_buf =
            ResourceProxy::new_buf(buffer_sizes.tiles.size_in_bytes().into(), "vello.tile_buf");
        let segments_buf = ResourceProxy::new_buf(
            buffer_sizes.segments.size_in_bytes().into(),
            "vello.segments_buf",
        );
        let ptcl_buf =
            ResourceProxy::new_buf(buffer_sizes.ptcl.size_in_bytes().into(), "vello.ptcl_buf");
        let reduced_buf = ResourceProxy::new_buf(
            buffer_sizes.path_reduced.size_in_bytes().into(),
            "vello.reduced_buf",
        );
        // TODO: really only need pathtag_wgs - 1
        recording.dispatch(
            shaders.pathtag_reduce,
            wg_counts.path_reduce,
            [config_buf, scene_buf, reduced_buf],
        );
        let mut pathtag_parent = reduced_buf;
        let mut large_pathtag_bufs = None;
        let use_large_path_scan = wg_counts.use_large_path_scan && !shaders.pathtag_is_cpu;
        if use_large_path_scan {
            let reduced2_buf = ResourceProxy::new_buf(
                buffer_sizes.path_reduced2.size_in_bytes().into(),
                "vello.reduced2_buf",
            );
            recording.dispatch(
                shaders.pathtag_reduce2,
                wg_counts.path_reduce2,
                [reduced_buf, reduced2_buf],
            );
            let reduced_scan_buf = ResourceProxy::new_buf(
                buffer_sizes.path_reduced_scan.size_in_bytes().into(),
                "reduced_scan_buf",
            );
            recording.dispatch(
                shaders.pathtag_scan1,
                wg_counts.path_scan1,
                [reduced_buf, reduced2_buf, reduced_scan_buf],
            );
            pathtag_parent = reduced_scan_buf;
            large_pathtag_bufs = Some((reduced2_buf, reduced_scan_buf));
        }

        let tagmonoid_buf = ResourceProxy::new_buf(
            buffer_sizes.path_monoids.size_in_bytes().into(),
            "vello.tagmonoid_buf",
        );
        let pathtag_scan = if use_large_path_scan {
            shaders.pathtag_scan_large
        } else {
            shaders.pathtag_scan
        };
        recording.dispatch(
            pathtag_scan,
            wg_counts.path_scan,
            [config_buf, scene_buf, pathtag_parent, tagmonoid_buf],
        );
        recording.free_resource(reduced_buf);
        if let Some((reduced2, reduced_scan)) = large_pathtag_bufs {
            recording.free_resource(reduced2);
            recording.free_resource(reduced_scan);
        }
        let path_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.path_bboxes.size_in_bytes().into(),
            "vello.path_bbox_buf",
        );
        recording.dispatch(
            shaders.bbox_clear,
            wg_counts.bbox_clear,
            [config_buf, path_bbox_buf],
        );
        let bump_buf = BufferProxy::new(
            buffer_sizes.bump_alloc.size_in_bytes().into(),
            "vello.bump_buf",
        );
        recording.clear_all(bump_buf);
        let bump_buf = ResourceProxy::Buffer(bump_buf);
        let lines_buf =
            ResourceProxy::new_buf(buffer_sizes.lines.size_in_bytes().into(), "vello.lines_buf");
        recording.dispatch(
            shaders.flatten,
            wg_counts.flatten,
            [
                config_buf,
                scene_buf,
                tagmonoid_buf,
                path_bbox_buf,
                bump_buf,
                lines_buf,
            ],
        );
        let draw_reduced_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_reduced.size_in_bytes().into(),
            "vello.draw_reduced_buf",
        );
        recording.dispatch(
            shaders.draw_reduce,
            wg_counts.draw_reduce,
            [config_buf, scene_buf, draw_reduced_buf],
        );
        let draw_monoid_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_monoids.size_in_bytes().into(),
            "vello.draw_monoid_buf",
        );
        let clip_inp_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_inps.size_in_bytes().into(),
            "vello.clip_inp_buf",
        );
        recording.dispatch(
            shaders.draw_leaf,
            wg_counts.draw_leaf,
            [
                config_buf,
                scene_buf,
                draw_reduced_buf,
                path_bbox_buf,
                draw_monoid_buf,
                info_bin_data_buf,
                clip_inp_buf,
            ],
        );
        recording.free_resource(draw_reduced_buf);
        let clip_el_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_els.size_in_bytes().into(),
            "vello.clip_el_buf",
        );
        let clip_bic_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_bics.size_in_bytes().into(),
            "vello.clip_bic_buf",
        );
        if wg_counts.clip_reduce.0 > 0 {
            recording.dispatch(
                shaders.clip_reduce,
                wg_counts.clip_reduce,
                [clip_inp_buf, path_bbox_buf, clip_bic_buf, clip_el_buf],
            );
        }
        let clip_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.clip_bboxes.size_in_bytes().into(),
            "vello.clip_bbox_buf",
        );
        if wg_counts.clip_leaf.0 > 0 {
            recording.dispatch(
                shaders.clip_leaf,
                wg_counts.clip_leaf,
                [
                    config_buf,
                    clip_inp_buf,
                    path_bbox_buf,
                    clip_bic_buf,
                    clip_el_buf,
                    draw_monoid_buf,
                    clip_bbox_buf,
                ],
            );
        }
        recording.free_resource(clip_inp_buf);
        recording.free_resource(clip_bic_buf);
        recording.free_resource(clip_el_buf);
        let draw_bbox_buf = ResourceProxy::new_buf(
            buffer_sizes.draw_bboxes.size_in_bytes().into(),
            "vello.draw_bbox_buf",
        );
        let bin_header_buf = ResourceProxy::new_buf(
            buffer_sizes.bin_headers.size_in_bytes().into(),
            "vello.bin_header_buf",
        );
        recording.dispatch(
            shaders.binning,
            wg_counts.binning,
            [
                config_buf,
                draw_monoid_buf,
                path_bbox_buf,
                clip_bbox_buf,
                draw_bbox_buf,
                bump_buf,
                info_bin_data_buf,
                bin_header_buf,
            ],
        );
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(clip_bbox_buf);
        // Note: this only needs to be rounded up because of the workaround to store the tile_offset
        // in storage rather than workgroup memory.
        let path_buf =
            ResourceProxy::new_buf(buffer_sizes.paths.size_in_bytes().into(), "vello.path_buf");
        recording.dispatch(
            shaders.tile_alloc,
            wg_counts.tile_alloc,
            [
                config_buf,
                scene_buf,
                draw_bbox_buf,
                bump_buf,
                path_buf,
                tile_buf,
            ],
        );
        recording.free_resource(draw_bbox_buf);
        recording.free_resource(tagmonoid_buf);
        let indirect_count_buf = BufferProxy::new(
            buffer_sizes.indirect_count.size_in_bytes().into(),
            "vello.indirect_count",
        );
        recording.dispatch(
            shaders.path_count_setup,
            wg_counts.path_count_setup,
            [bump_buf, indirect_count_buf.into()],
        );
        let seg_counts_buf = ResourceProxy::new_buf(
            buffer_sizes.seg_counts.size_in_bytes().into(),
            "vello.seg_counts_buf",
        );
        recording.dispatch_indirect(
            shaders.path_count,
            indirect_count_buf,
            0,
            [
                config_buf,
                bump_buf,
                lines_buf,
                path_buf,
                tile_buf,
                seg_counts_buf,
            ],
        );
        recording.dispatch(
            shaders.backdrop,
            wg_counts.backdrop,
            [config_buf, bump_buf, path_buf, tile_buf],
        );
        recording.dispatch(
            shaders.coarse,
            wg_counts.coarse,
            [
                config_buf,
                scene_buf,
                draw_monoid_buf,
                bin_header_buf,
                info_bin_data_buf,
                path_buf,
                tile_buf,
                bump_buf,
                ptcl_buf,
            ],
        );
        recording.dispatch(
            shaders.path_tiling_setup,
            wg_counts.path_tiling_setup,
            [bump_buf, indirect_count_buf.into(), ptcl_buf],
        );
        recording.dispatch_indirect(
            shaders.path_tiling,
            indirect_count_buf,
            0,
            [
                bump_buf,
                seg_counts_buf,
                lines_buf,
                path_buf,
                tile_buf,
                segments_buf,
            ],
        );
        recording.free_buffer(indirect_count_buf);
        recording.free_resource(seg_counts_buf);
        recording.free_resource(scene_buf);
        recording.free_resource(draw_monoid_buf);
        recording.free_resource(bin_header_buf);
        recording.free_resource(path_buf);
        let out_image = ImageProxy::new(params.width, params.height, ImageFormat::Rgba8);
        let blend_spill_buf = BufferProxy::new(
            buffer_sizes.blend_spill.size_in_bytes().into(),
            "vello.blend_spill",
        );
        self.fine_wg_count = Some(wg_counts.fine);
        self.fine_resources = Some(FineResources {
            aa_config: params.antialiasing_method,
            config_buf,
            bump_buf,
            tile_buf,
            segments_buf,
            ptcl_buf,
            gradient_image,
            info_bin_data_buf,
            blend_spill_buf: ResourceProxy::Buffer(blend_spill_buf),
            effect_params_buf: ResourceProxy::Buffer(BufferProxy::new(256, "vello.effect_params")),
            image_atlas: ResourceProxy::Image(image_atlas),
            out_image,
        });
        if robust {
            recording.download(*bump_buf.as_buf().unwrap());
        }
        recording.free_resource(bump_buf);

        #[cfg(feature = "debug_layers")]
        {
            if robust {
                let path_bboxes = *path_bbox_buf.as_buf().unwrap();
                let lines = *lines_buf.as_buf().unwrap();
                recording.download(lines);

                self.captured_buffers = Some(CapturedBuffers {
                    sizes: cpu_config.buffer_sizes,
                    path_bboxes,
                    lines,
                });
            } else {
                recording.free_resource(path_bbox_buf);
                recording.free_resource(lines_buf);
            }
        }
        #[cfg(not(feature = "debug_layers"))]
        {
            recording.free_resource(path_bbox_buf);
            recording.free_resource(lines_buf);
        }

        recording
    }

    /// Run fine rasterization assuming the coarse phase succeeded.
    pub fn record_fine(&mut self, shaders: &FullShaders, recording: &mut Recording) {
        let fine_wg_count = self.fine_wg_count.take().unwrap();
        let fine = self.fine_resources.take().unwrap();
        match fine.aa_config {
            AaConfig::Area => {
                recording.dispatch(
                    shaders
                        .fine_area
                        .expect("shaders not configured to support AA mode: area"),
                    fine_wg_count,
                    [
                        fine.config_buf,
                        fine.segments_buf,
                        fine.ptcl_buf,
                        fine.info_bin_data_buf,
                        fine.blend_spill_buf,
                        ResourceProxy::Image(fine.out_image),
                        fine.gradient_image,
                        fine.image_atlas,
                        fine.effect_params_buf,
                    ],
                );
            }
            _ => {
                if self.mask_buf.is_none() {
                    let mask_lut = match fine.aa_config {
                        AaConfig::Msaa16 => make_mask_lut_16(),
                        AaConfig::Msaa8 => make_mask_lut(),
                        _ => unreachable!(),
                    };
                    let buf = recording.upload("vello.mask_lut", mask_lut);
                    self.mask_buf = Some(buf.into());
                }
                let fine_shader = match fine.aa_config {
                    AaConfig::Msaa16 => shaders
                        .fine_msaa16
                        .expect("shaders not configured to support AA mode: msaa16"),
                    AaConfig::Msaa8 => shaders
                        .fine_msaa8
                        .expect("shaders not configured to support AA mode: msaa8"),
                    _ => unreachable!(),
                };
                recording.dispatch(
                    fine_shader,
                    fine_wg_count,
                    [
                        fine.config_buf,
                        fine.segments_buf,
                        fine.ptcl_buf,
                        fine.info_bin_data_buf,
                        fine.blend_spill_buf,
                        ResourceProxy::Image(fine.out_image),
                        fine.gradient_image,
                        fine.image_atlas,
                        fine.effect_params_buf,
                        self.mask_buf.unwrap(),
                    ],
                );
            }
        }
        recording.free_resource(fine.config_buf);
        recording.free_resource(fine.tile_buf);
        recording.free_resource(fine.segments_buf);
        recording.free_resource(fine.ptcl_buf);
        recording.free_resource(fine.gradient_image);
        recording.free_resource(fine.info_bin_data_buf);
        recording.free_resource(fine.blend_spill_buf);
        recording.free_resource(fine.effect_params_buf);
        // TODO: make mask buf persistent
        if let Some(mask_buf) = self.mask_buf.take() {
            recording.free_resource(mask_buf);
        }
    }

    /// Get the output image.
    ///
    /// This is going away, as the caller will add the output image to the bind
    /// map.
    pub fn out_image(&self) -> ImageProxy {
        self.fine_resources.as_ref().unwrap().out_image
    }

    pub fn bump_buf(&self) -> BufferProxy {
        *self
            .fine_resources
            .as_ref()
            .unwrap()
            .bump_buf
            .as_buf()
            .unwrap()
    }

    #[cfg(feature = "debug_layers")]
    pub fn take_captured_buffers(&mut self) -> Option<CapturedBuffers> {
        self.captured_buffers.take()
    }
}
