// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Demonstrates using Vello Hybrid using a WebGL2 backend in the browser.

#![allow(
    clippy::cast_possible_truncation,
    reason = "truncation has no appreciable impact in this demo"
)]
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;
use vello_common::{
    fearless_simd::Level,
    kurbo::{Affine, Point},
    paint::{ImageId, ImageSource},
};
use vello_example_scenes::{AnyScene, image::ImageScene};
use vello_hybrid::{Pixmap, RenderSettings, RenderTargetConfig, Renderer, Scene};
use wasm_bindgen::prelude::*;
use web_sys::{Event, HtmlCanvasElement, KeyboardEvent, MouseEvent, WheelEvent};
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

        // Try real WebGPU first, then fall back to wgpu-over-WebGL2. Each backend needs its
        // own instance+surface (the surface is bound to the instance), so build them together
        // per attempt and keep whichever yields an adapter.
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
            "vello_hybrid backend = {:?} | adapter = {} ({:?})",
            info.backend,
            info.name,
            info.device_type
        );

        // Use the adapter's full limits. On WebGPU this unlocks far higher texture/buffer
        // limits than `downlevel_webgl2_defaults`; on the GL fallback it already reflects the
        // WebGL2 caps, so requesting exactly what the adapter offers always succeeds.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("Device to be valid");

        // Configure the surface using the canvas's *preferred* format (wgpu reports it first).
        // On a WebGPU canvas that's `bgra8unorm`; forcing a non-preferred format makes the
        // browser insert an extra per-frame copy/swizzle before compositing. The renderer builds
        // its final-target pipeline against whatever format we pass, so targeting the preferred
        // one directly is both correct and avoids that copy.
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
                level: Level::try_detect().unwrap_or(Level::baseline()),
                // The stacked-effects stress scene renders many multi-pass blur layers at once,
                // each needing initial + ping-pong scratch textures in the filter atlas. The
                // default cap (8 × 4096²) is exhausted when zoomed in; raise it so the demo can
                // push much further before it hits Vello's hard limit. Clamped down automatically
                // by `normalize_atlas_config` to the WebGL2 backend's real texture-layer limit.
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

    fn resize(&mut self, width: u32, height: u32) {
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

/// State that handles scene rendering and interactions
struct AppState {
    scenes: Box<[AnyScene<Scene>]>,
    uploaded_scene_images: Box<[bool]>,
    current_scene: usize,
    scene: Scene,
    transform: Affine,
    mouse_down: bool,
    last_cursor_position: Option<Point>,
    width: u32,
    height: u32,
    renderer_wrapper: RendererWrapper,
    need_render: bool,
    canvas: HtmlCanvasElement,
    /// Timestamp (ms) of the last successfully presented frame, for cadence measurement.
    last_present_ms: Option<f64>,
    /// Exponential moving average of the present-to-present interval, in ms.
    frame_interval_ema_ms: f64,
}

impl AppState {
    async fn new(canvas: HtmlCanvasElement, scenes: Box<[AnyScene<Scene>]>) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        let uploaded_scene_images = vec![false; scenes.len()].into_boxed_slice();

        let renderer_wrapper = RendererWrapper::new(canvas.clone()).await;

        let mut app_state = Self {
            scenes,
            uploaded_scene_images,
            current_scene: 0,
            scene: Scene::new(width as u16, height as u16),
            transform: Affine::IDENTITY,
            mouse_down: false,
            last_cursor_position: None,
            width,
            height,
            renderer_wrapper,
            need_render: true,
            canvas,
            last_present_ms: None,
            frame_interval_ema_ms: 0.0,
        };

        // Upload images to the WebGL atlas
        app_state.upload_images_to_atlas();

        app_state
    }

    fn render(&mut self) {
        if !self.need_render {
            return;
        }

        let frame_start = now_ms();

        self.scene.reset();

        // Render the current scene with transform
        self.scenes[self.current_scene].render(&mut self.scene, self.transform);

        let render_size = vello_hybrid::RenderSize {
            width: self.width,
            height: self.height,
        };

        let surface_texture = match self.renderer_wrapper.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            CurrentSurfaceTexture::Occluded
            | CurrentSurfaceTexture::Timeout
            | CurrentSurfaceTexture::Outdated
            | CurrentSurfaceTexture::Suboptimal(_) => {
                return;
            }
            CurrentSurfaceTexture::Lost => panic!("Surface was lost"),
            CurrentSurfaceTexture::Validation => {
                panic!("Validation error getting surface")
            }
        };
        let surface_texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .renderer_wrapper
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if let Err(e) = self.renderer_wrapper.renderer.render(
            &self.scene,
            self.scenes[self.current_scene].resources_mut(),
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &mut encoder,
            &render_size,
            &surface_texture_view,
            &vello_hybrid::TextureBindings::new(),
        ) {
            // A filter-heavy scene zoomed in far enough can exhaust Vello's fixed
            // filter scratch-atlas capacity (`AtlasError::AtlasLimitReached`). That's a
            // recoverable resource limit, not a fatal error — skip this frame, keep the
            // last good frame on screen, and surface the reason instead of panicking.
            log::warn!("frame skipped: {e:?}");
            update_frame_stats(
                self.frame_interval_ema_ms,
                now_ms() - frame_start,
                Some(format!("⚠ {e:?} — zoom out or use fewer stacked effects")),
            );
            self.need_render = false;
            return;
        }

        self.renderer_wrapper.queue.submit([encoder.finish()]);
        surface_texture.present();

        // CPU-side portion: time spent building the scene and encoding/submitting commands.
        let encode_ms = now_ms() - frame_start;

        // Perceived frame rate = wall-clock interval between frames that actually reach the
        // screen. With continuous rendering the browser paces `requestAnimationFrame` to real
        // display cadence and applies GPU backpressure (a saturated swapchain returns early
        // above without updating `last_present_ms`), so this present-to-present delta reflects
        // what the user sees — vsync-capped when the GPU has headroom, GPU-limited when it does
        // not. Unlike a GPU-completion fence, it is immune to event-loop scheduling lag.
        let now = now_ms();
        if let Some(prev) = self.last_present_ms {
            let interval = now - prev;
            self.frame_interval_ema_ms = if self.frame_interval_ema_ms <= 0.0 {
                interval
            } else {
                self.frame_interval_ema_ms * 0.9 + interval * 0.1
            };
        }
        self.last_present_ms = Some(now);

        update_frame_stats(
            self.frame_interval_ema_ms,
            encode_ms,
            self.scenes[self.current_scene].status(),
        );

        self.need_render = false;
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.width = width;
        self.height = height;

        self.scene = Scene::new(width as u16, height as u16);
        self.renderer_wrapper.resize(width, height);

        self.need_render = true;
    }

    fn next_scene(&mut self) {
        self.current_scene = (self.current_scene + 1) % self.scenes.len();
        self.upload_images_to_atlas();
        self.transform = Affine::IDENTITY;
        self.need_render = true;
    }

    fn prev_scene(&mut self) {
        self.current_scene = if self.current_scene == 0 {
            self.scenes.len() - 1
        } else {
            self.current_scene - 1
        };
        self.upload_images_to_atlas();
        self.transform = Affine::IDENTITY;
        self.need_render = true;
    }

    fn reset_transform(&mut self) {
        self.transform = Affine::IDENTITY;
        self.need_render = true;
    }

    /// Rotate the view about the cursor (or the canvas center if the cursor has left),
    /// matching the upstream `with_winit` demo's Q/E controls.
    fn rotate(&mut self, clockwise: bool) {
        let pivot = self.last_cursor_position.unwrap_or(Point {
            x: 0.5 * self.width as f64,
            y: 0.5 * self.height as f64,
        });
        let angle = if clockwise { -0.05 } else { 0.05 };
        self.transform = self.transform.then_rotate_about(angle, pivot);
        self.need_render = true;
    }

    fn handle_key(&mut self, key: &str) {
        if let Some(scene) = self.scenes.get_mut(self.current_scene)
            && scene.handle_key(key)
        {
            self.need_render = true;
        }
    }

    fn handle_mouse_down(&mut self, x: f64, y: f64) {
        self.mouse_down = true;
        self.last_cursor_position = Some(Point { x, y });
    }

    fn handle_mouse_up(&mut self) {
        self.mouse_down = false;
        self.last_cursor_position = None;
    }

    fn handle_mouse_move(&mut self, x: f64, y: f64) {
        let current_pos = Point { x, y };

        if self.mouse_down
            && let Some(last_pos) = self.last_cursor_position
        {
            self.transform = self.transform.then_translate(current_pos - last_pos);
            self.need_render = true;
        }

        self.last_cursor_position = Some(current_pos);
    }

    fn handle_wheel(&mut self, delta_y: f64) {
        const ZOOM_STEP: f64 = 0.1;
        let zoom_factor = (1.0 + delta_y * ZOOM_STEP).max(0.1);

        // Zoom centered at cursor position, or the center if no position is set.
        self.transform = self.transform.then_scale_about(
            zoom_factor,
            self.last_cursor_position.unwrap_or(Point {
                x: 0.5 * self.width as f64,
                y: 0.5 * self.height as f64,
            }),
        );

        self.need_render = true;
    }

    fn upload_images_to_atlas(&mut self) {
        if self.uploaded_scene_images[self.current_scene] {
            return;
        }

        let mut encoder =
            self.renderer_wrapper
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Upload Image pass"),
                });

        // 1st example — uploading pixmap directly to WebGL atlas
        let pixmap1 = ImageScene::read_flower_image();
        self.renderer_wrapper.renderer.upload_image(
            self.scenes[self.current_scene].resources_mut(),
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &mut encoder,
            &pixmap1,
        );

        // 2nd example — uploading from a WebGL texture
        let pixmap2 = ImageScene::read_cowboy_image();
        let texture2 = self.upload_image_to_texture(
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &pixmap2,
        );
        self.renderer_wrapper.renderer.upload_image(
            self.scenes[self.current_scene].resources_mut(),
            &self.renderer_wrapper.device,
            &self.renderer_wrapper.queue,
            &mut encoder,
            &texture2,
        );

        self.renderer_wrapper.queue.submit([encoder.finish()]);
        self.uploaded_scene_images[self.current_scene] = true;
    }

    fn upload_image_to_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &Pixmap,
    ) -> wgpu::Texture {
        let image_width = image.width() as u32;
        let image_height = image.height() as u32;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Uploaded Image Texture"),
            size: wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // 4 bytes per RGBA pixel
                bytes_per_row: Some(4 * image_width),
                rows_per_image: Some(image_height),
            },
            wgpu::Extent3d {
                width: image_width,
                height: image_height,
                depth_or_array_layers: 1,
            },
        );

        texture
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = requestAnimationFrame)]
    fn request_animation_frame(f: &Closure<dyn FnMut()>);
}

/// Creates a `HTMLCanvasElement` of the given dimensions and renders the given scenes into it,
/// with interactive controls for panning, zooming, and switching between scenes.
/// Current high-resolution timestamp in milliseconds.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

/// Update the on-screen overlay. `interval_avg_ms` is the smoothed present-to-present frame
/// interval (perceived cadence), from which fps is derived; `encode_ms` is the CPU-side scene
/// build + command encoding time for this frame.
fn update_frame_stats(interval_avg_ms: f64, encode_ms: f64, status: Option<String>) {
    let Some(doc) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    if let Some(el) = doc.get_element_by_id("frame-stats") {
        let fps = if interval_avg_ms > 0.0 {
            1000.0 / interval_avg_ms
        } else {
            0.0
        };
        let mut text =
            format!("{fps:.0} fps  ·  frame {interval_avg_ms:.2} ms  ·  encode {encode_ms:.2} ms");
        if let Some(status) = status {
            text.push_str("  ·  ");
            text.push_str(&status);
        }
        el.set_text_content(Some(&text));
    }
}

pub async fn run_interactive(canvas_width: u16, canvas_height: u16) {
    let canvas = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(canvas_width as u32);
    canvas.set_height(canvas_height as u32);
    canvas.style().set_property("width", "100%").unwrap();
    canvas.style().set_property("height", "100%").unwrap();

    let body = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap();
    // Apply background color so white text can be seen.
    body.style()
        .set_property("background-color", "#111")
        .unwrap();

    // Add canvas to body
    web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap()
        .append_child(&canvas)
        .unwrap();

    let scenes = vello_example_scenes::get_example_scenes(
        vello_example_scenes::Capabilities::default(),
        vec![
            ImageSource::opaque_id(ImageId::new(0)),
            ImageSource::opaque_id(ImageId::new(1)),
        ],
    );

    let app_state = Rc::new(RefCell::new(AppState::new(canvas.clone(), scenes).await));

    // Set up animation frame loop
    {
        let f = Rc::new(RefCell::new(None));
        let g = f.clone();
        let app_state = app_state.clone();

        *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            // Use `try_borrow_mut` so a slow frame (heavy stacked effects) that lets the
            // browser dispatch an input event mid-render can't re-enter and panic.
            if let Ok(mut state) = app_state.try_borrow_mut() {
                // Render every animation frame so the fps readout reflects steady-state
                // present cadence, not just frames triggered by input events.
                state.need_render = true;
                state.render();
            }
            request_animation_frame(f.borrow().as_ref().unwrap());
        }) as Box<dyn FnMut()>));

        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    // Set up window resize event handler
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |_: Event| {
            let window = web_sys::window().unwrap();
            let dpr = window.device_pixel_ratio();

            let width = window.inner_width().unwrap().as_f64().unwrap() as u32 * dpr as u32;
            let height = window.inner_height().unwrap().as_f64().unwrap() as u32 * dpr as u32;

            if let Ok(mut state) = app_state.try_borrow_mut() {
                state.resize(width, height);
            }
        }) as Box<dyn FnMut(_)>);

        let window = web_sys::window().unwrap();
        window
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Set up event handlers

    // Mouse down
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            if let Ok(mut state) = app_state.try_borrow_mut() {
                state.handle_mouse_down(event.client_x() as f64, event.client_y() as f64);
            }
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mousedown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mouse up
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |_event: MouseEvent| {
            if let Ok(mut state) = app_state.try_borrow_mut() {
                state.handle_mouse_up();
            }
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mouseup", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mouse move
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |event: MouseEvent| {
            if let Ok(mut state) = app_state.try_borrow_mut() {
                state.handle_mouse_move(event.client_x() as f64, event.client_y() as f64);
            }
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("mousemove", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Mouse wheel
    {
        let app_state = app_state.clone();
        let closure = Closure::wrap(Box::new(move |event: WheelEvent| {
            event.prevent_default();
            let delta = -event.delta_y() / 100.0; // Normalize and invert
            if let Ok(mut state) = app_state.try_borrow_mut() {
                state.handle_wheel(delta);
            }
        }) as Box<dyn FnMut(_)>);
        canvas
            .add_event_listener_with_callback("wheel", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Keyboard events (document level)
    {
        let app_state = app_state.clone();
        let document = web_sys::window().unwrap().document().unwrap();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let key = event.key();
            if let Ok(mut state) = app_state.try_borrow_mut() {
                match key.as_str() {
                    "ArrowRight" => state.next_scene(),
                    "ArrowLeft" => state.prev_scene(),
                    " " => state.reset_transform(),
                    "q" | "Q" => state.rotate(false),
                    "e" | "E" => state.rotate(true),
                    _ => {
                        state.handle_key(key.as_str());
                    }
                }
            }
        }) as Box<dyn FnMut(_)>);
        document
            .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Create instructions element
    let document = web_sys::window().unwrap().document().unwrap();
    let instructions = document.create_element("div").unwrap();
    instructions.set_inner_html(
        "Left/Right Arrow: Change scene | Q/E: Rotate | Space: Reset view | Mouse Drag: Pan | Mouse Wheel: Zoom",
    );
    let style = instructions
        .dyn_ref::<web_sys::HtmlElement>()
        .unwrap()
        .style();
    style.set_property("position", "fixed").unwrap();
    style.set_property("bottom", "10px").unwrap();
    style.set_property("left", "10px").unwrap();
    style
        .set_property("background", "rgba(0, 0, 0, 0.5)")
        .unwrap();
    style.set_property("color", "white").unwrap();
    style.set_property("padding", "5px 10px").unwrap();
    style.set_property("border-radius", "5px").unwrap();
    style.set_property("font-family", "sans-serif").unwrap();
    style.set_property("pointer-events", "none").unwrap();

    document
        .body()
        .unwrap()
        .append_child(&instructions)
        .unwrap();

    // Create the frame-time stats overlay (lower-right, updated on GPU completion each frame).
    let stats = document.create_element("div").unwrap();
    stats.set_id("frame-stats");
    stats.set_text_content(Some("— fps"));
    let stats_style = stats.dyn_ref::<web_sys::HtmlElement>().unwrap().style();
    stats_style.set_property("position", "fixed").unwrap();
    stats_style.set_property("bottom", "10px").unwrap();
    stats_style.set_property("right", "10px").unwrap();
    stats_style
        .set_property("background", "rgba(0, 0, 0, 0.6)")
        .unwrap();
    stats_style.set_property("color", "#4ade80").unwrap();
    stats_style.set_property("padding", "5px 10px").unwrap();
    stats_style.set_property("border-radius", "5px").unwrap();
    stats_style.set_property("font-family", "monospace").unwrap();
    stats_style.set_property("pointer-events", "none").unwrap();
    document.body().unwrap().append_child(&stats).unwrap();
}

/// Creates a `HTMLCanvasElement` and renders a single scene into it
pub async fn render_scene(scene: Scene, width: u16, height: u16) {
    let canvas = web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);
    canvas.style().set_property("width", "100%").unwrap();
    canvas.style().set_property("height", "100%").unwrap();

    // Add canvas to body
    web_sys::Window::document(&web_sys::window().unwrap())
        .unwrap()
        .body()
        .unwrap()
        .append_child(&canvas)
        .unwrap();

    let RendererWrapper {
        mut renderer,
        device,
        queue,
        surface,
        surface_format: _,
    } = RendererWrapper::new(canvas).await;

    let render_size = vello_hybrid::RenderSize {
        width: width as u32,
        height: height as u32,
    };
    let surface_texture = match surface.get_current_texture() {
        CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
        e => panic!("Error getting initial surface: {e:?}"),
    };
    let surface_texture_view = surface_texture
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let mut resources = vello_hybrid::Resources::new();

    renderer
        .render(
            &scene,
            &mut resources,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &surface_texture_view,
            &vello_hybrid::TextureBindings::new(),
        )
        .unwrap();

    queue.submit([encoder.finish()]);
    surface_texture.present();
}
