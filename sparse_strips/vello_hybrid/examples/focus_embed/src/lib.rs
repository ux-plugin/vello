// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Phase 0 of the "Vello as a second render-wasm backend" plan: an **embeddable**
//! wasm-bindgen + wgpu Vello renderer.
//!
//! Unlike the `wgpu_webgl` demo — which creates its own canvas and owns the
//! animation-frame loop — [`FocusRenderer`] is a self-contained component that:
//!   * mounts onto a **host-provided** `<canvas>`,
//!   * exposes discrete `render()` / `resize()` / `key()` calls,
//!   * owns **no** event loop or `requestAnimationFrame` — the host drives it.
//!
//! That is exactly the shape Penpot's focus mode would consume: the CLJS host
//! creates the canvas, constructs the renderer, and calls `render()` when it wants
//! a frame. `main.rs` is a *mock host* that plays that role for local testing.

#![cfg(target_arch = "wasm32")]
#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this Phase-0 proof"
)]

mod model_scene;

use vello_common::kurbo::Affine;
use vello_example_scenes::{AnyScene, custom_filter, nested_ui, stacked_effects};
use vello_hybrid::{RenderSettings, RenderTargetConfig, Renderer, Scene};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;
use wgpu::{
    CurrentSurfaceTexture,
    rwh::{DisplayHandle, HandleError, HasDisplayHandle},
};

#[derive(Debug)]
struct OurDisplayHandle;
impl HasDisplayHandle for OurDisplayHandle {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(DisplayHandle::web())
    }
}

/// The wgpu device/surface/renderer bundle bound to the host canvas.
struct RendererWrapper {
    renderer: Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
}

impl RendererWrapper {
    async fn new(canvas: HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();

        // Prefer real WebGPU, fall back to wgpu-over-WebGL2. Each backend needs its own
        // instance+surface, so build them together per attempt and keep whichever yields
        // an adapter.
        async fn try_backend(
            backends: wgpu::Backends,
            canvas: &HtmlCanvasElement,
        ) -> Option<(wgpu::Instance, wgpu::Surface<'static>, wgpu::Adapter)> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends,
                ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(OurDisplayHandle))
            });
            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
                .ok()?;
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .ok()?;
            Some((instance, surface, adapter))
        }

        let (_instance, surface, adapter) =
            match try_backend(wgpu::Backends::BROWSER_WEBGPU, &canvas).await {
                Some(triple) => triple,
                None => {
                    log::warn!("WebGPU unavailable, falling back to WebGL2");
                    try_backend(wgpu::Backends::GL, &canvas)
                        .await
                        .expect("Neither WebGPU nor WebGL2 adapter available")
                }
            };

        let info = adapter.get_info();
        log::info!(
            "focus_embed backend = {:?} | adapter = {} ({:?})",
            info.backend,
            info.name,
            info.device_type
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("focus_embed device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("Device to be valid");

        // Use the canvas's preferred format (wgpu reports it first) to avoid an extra
        // per-frame copy on the WebGPU path.
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .first()
            .copied()
            .unwrap_or(wgpu::TextureFormat::Rgba8Unorm);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let renderer = Renderer::new_with(
            &device,
            &RenderTargetConfig {
                format: surface_format,
                width,
                height,
            },
            RenderSettings {
                level: vello_common::fearless_simd::Level::try_detect()
                    .unwrap_or(vello_common::fearless_simd::Level::baseline()),
                // Give the filter atlas headroom so the effect-heavy focus scenes can push far
                // before hitting Vello's hard cap (auto-clamped to the backend's real limit).
                filter_atlas_config: vello_common::multi_atlas::AtlasConfig {
                    initial_atlas_count: 0,
                    max_atlases: 32,
                    ..Default::default()
                },
                ..Default::default()
            },
        );

        Self {
            renderer,
            device,
            queue,
            surface,
            surface_format,
        }
    }

    fn reconfigure(&self, width: u32, height: u32) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };
        self.surface.configure(&self.device, &surface_config);
    }
}

/// An embeddable Vello renderer bound to a host `<canvas>`.
///
/// The host owns the lifecycle: construct via [`create_focus_renderer`], then call
/// [`FocusRenderer::render`] to produce a frame, [`FocusRenderer::resize`] on canvas
/// resize, and [`FocusRenderer::key`] to forward input. There is no internal loop.
#[wasm_bindgen]
pub struct FocusRenderer {
    canvas: HtmlCanvasElement,
    wrapper: RendererWrapper,
    scene: Scene,
    scenes: Vec<AnyScene<Scene>>,
    current: usize,
    transform: Affine,
    width: u32,
    height: u32,
}

/// Construct a [`FocusRenderer`] on a host-provided canvas. Async because adapter/device
/// acquisition is async; the JS side receives a `Promise<FocusRenderer>`.
#[wasm_bindgen]
pub async fn create_focus_renderer(canvas: HtmlCanvasElement) -> FocusRenderer {
    let width = canvas.width();
    let height = canvas.height();
    let wrapper = RendererWrapper::new(canvas.clone()).await;

    // The "focus scenes": stand-ins for what focus mode would hand off. These are
    // self-contained (no image resources) so the module needs only a minimal handoff.
    let scenes: Vec<AnyScene<Scene>> = vec![
        // The end-to-end proof: a neutral render_core::model scene, drawn by Vello.
        AnyScene::new(model_scene::NeutralModelScene::new()),
        AnyScene::new(nested_ui::NestedUiScene::new()),
        AnyScene::new(custom_filter::CustomFilterScene::new()),
        AnyScene::new(stacked_effects::StackedEffectsScene::new()),
    ];

    FocusRenderer {
        canvas,
        wrapper,
        scene: Scene::new(width as u16, height as u16),
        scenes,
        current: 0,
        transform: Affine::IDENTITY,
        width,
        height,
    }
}

#[wasm_bindgen]
impl FocusRenderer {
    /// Render one frame into the host canvas. The host decides when to call this.
    pub fn render(&mut self) {
        self.scene.reset();
        self.scenes[self.current].render(&mut self.scene, self.transform);

        let render_size = vello_hybrid::RenderSize {
            width: self.width,
            height: self.height,
        };

        let surface_texture = match self.wrapper.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) => t,
            CurrentSurfaceTexture::Occluded
            | CurrentSurfaceTexture::Timeout
            | CurrentSurfaceTexture::Outdated
            | CurrentSurfaceTexture::Suboptimal(_) => return,
            CurrentSurfaceTexture::Lost => {
                log::warn!("surface lost");
                return;
            }
            CurrentSurfaceTexture::Validation => {
                log::warn!("surface validation error");
                return;
            }
        };
        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .wrapper
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if let Err(e) = self.wrapper.renderer.render(
            &self.scene,
            self.scenes[self.current].resources_mut(),
            &self.wrapper.device,
            &self.wrapper.queue,
            &mut encoder,
            &render_size,
            &view,
            &vello_hybrid::TextureBindings::new(),
        ) {
            // Recoverable resource-limit (e.g. filter atlas exhausted): skip the frame
            // rather than panic, so the host stays alive.
            log::warn!("focus_embed frame skipped: {e:?}");
            return;
        }

        self.wrapper.queue.submit([encoder.finish()]);
        surface_texture.present();
    }

    /// Resize the render surface when the host canvas changes size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;
        self.wrapper.reconfigure(width, height);
        self.scene = Scene::new(width as u16, height as u16);
    }

    /// Forward a key press to the active scene (e.g. ArrowUp grows the nested-UI scene).
    /// Returns true if the scene consumed it (host should re-render).
    pub fn key(&mut self, key: &str) -> bool {
        self.scenes[self.current].handle_key(key)
    }

    /// Switch which focus scene is shown.
    pub fn set_scene(&mut self, index: usize) {
        if !self.scenes.is_empty() {
            self.current = index % self.scenes.len();
            self.transform = Affine::IDENTITY;
        }
    }

    /// Number of available focus scenes.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Set the view transform (a,b,c,d,e,f) — column-major affine, for pan/zoom/rotate.
    pub fn set_transform(&mut self, a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) {
        self.transform = Affine::new([a, b, c, d, e, f]);
    }

    /// Reset the view transform to identity.
    pub fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
    }

    /// Status string from the active scene (element counts, scale, etc.).
    pub fn status(&self) -> Option<String> {
        self.scenes[self.current].status()
    }
}
