// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Headless stress demo: the "bad in Skia" case — many effects stacked.
//!
//! Each cell draws a filled rounded body plus a wavy stroke, then wraps that
//! content in a DEEP NEST of filter layers:
//!
//!   GaussianBlur  ->  DropShadow  ->  Custom(tint)  ->  GaussianBlur  ->  content
//!
//! In a tiled immediate-mode renderer (Skia), each nested effect is its own
//! offscreen `save_layer`, recomputed per tile per frame — the combinatorial
//! cost we discussed. Here every cell is a chain of render-graph nodes.
//!
//! Run with:
//!   cargo run -p vello_hybrid --example stacked_effects -- out.png

use std::io::BufWriter;
use vello_common::color::palette::css;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};
use vello_common::pixmap::Pixmap;
use vello_hybrid::{Resources, Scene};

const COLS: u16 = 4;
const ROWS: u16 = 3;
const CELL: u16 = 260;
const WIDTH: u16 = COLS * CELL;
const HEIGHT: u16 = ROWS * CELL;

/// A wavy (sine) open path — a cheap stand-in for a "wavy stroke".
fn wavy_path(x0: f64, y0: f64, len: f64, amp: f64, waves: f64) -> BezPath {
    let mut p = BezPath::new();
    let steps = 72;
    p.move_to((x0, y0));
    for i in 1..=steps {
        let t = f64::from(i) / f64::from(steps);
        let x = x0 + t * len;
        let y = y0 + amp * (t * waves * std::f64::consts::TAU).sin();
        p.line_to((x, y));
    }
    p
}

fn draw_content(ctx: &mut Scene) {
    // Filled body.
    ctx.set_paint(css::DEEP_SKY_BLUE);
    ctx.fill_rect(&Rect::new(40.0, 70.0, 220.0, 200.0));
    // Wavy stroke across the body.
    ctx.set_paint(css::ORANGE_RED);
    ctx.set_stroke(Stroke::new(7.0));
    ctx.stroke_path(&wavy_path(30.0, 135.0, 200.0, 34.0, 2.5));
}

/// Push the deep effect stack (outer -> inner), draw the content, then unwind.
fn draw_stacked_cell(ctx: &mut Scene, base: Affine) {
    // Outer -> inner. Each is a separate filtered layer (render-graph node).
    let stack = [
        Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 4.0,
            edge_mode: EdgeMode::None,
        }),
        Filter::from_primitive(FilterPrimitive::DropShadow {
            dx: 10.0,
            dy: 10.0,
            std_deviation: 8.0,
            color: css::BLACK,
            edge_mode: EdgeMode::None,
        }),
        Filter::from_primitive(FilterPrimitive::Custom {
            effect: 0,
            params: [0.15_f32, 0.9, 0.35, 0.45].into_iter().collect(),
            expansion: [0.0, 0.0, 0.0, 0.0],
        }),
        Filter::from_primitive(FilterPrimitive::GaussianBlur {
            std_deviation: 3.0,
            edge_mode: EdgeMode::None,
        }),
    ];

    for filter in &stack {
        ctx.set_transform(base);
        ctx.push_layer(None, None, None, None, Some(filter.clone()));
    }

    ctx.set_transform(base);
    draw_content(ctx);

    for _ in &stack {
        ctx.pop_layer();
    }
}

fn build_scene() -> Scene {
    let mut scene = Scene::new(WIDTH, HEIGHT);

    scene.set_transform(Affine::IDENTITY);
    scene.set_paint(css::WHITE);
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT)));

    for row in 0..ROWS {
        for col in 0..COLS {
            let base = Affine::translate((
                f64::from(col) * f64::from(CELL),
                f64::from(row) * f64::from(CELL),
            ));
            draw_stacked_cell(&mut scene, base);
        }
    }

    scene
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let output_filename = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "stacked_effects.png".to_string());

    let scene = build_scene();

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("Failed to find an appropriate adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_features: wgpu::Features::empty(),
            ..Default::default()
        })
        .await
        .expect("Failed to create device");

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Render Target"),
        size: wgpu::Extent3d {
            width: u32::from(WIDTH),
            height: u32::from(HEIGHT),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut renderer = vello_hybrid::Renderer::new(
        &device,
        &vello_hybrid::RenderTargetConfig {
            format: texture.format(),
            width: u32::from(WIDTH),
            height: u32::from(HEIGHT),
        },
    );
    let mut resources = Resources::new();
    let render_size = vello_hybrid::RenderSize {
        width: u32::from(WIDTH),
        height: u32::from(HEIGHT),
    };

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Vello Render"),
    });
    renderer
        .render(
            &scene,
            &mut resources,
            &device,
            &queue,
            &mut encoder,
            &render_size,
            &texture_view,
            &vello_hybrid::TextureBindings::new(),
        )
        .unwrap();

    let bytes_per_row = (u32::from(WIDTH) * 4).next_multiple_of(256);
    let texture_copy_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: u64::from(bytes_per_row) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &texture_copy_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: u32::from(WIDTH),
            height: u32::from(HEIGHT),
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    texture_copy_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            result.expect("Failed to map texture for reading");
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let mut img_data = Vec::with_capacity(usize::from(WIDTH) * usize::from(HEIGHT) * 4);
    for row in texture_copy_buffer
        .slice(..)
        .get_mapped_range()
        .chunks_exact(bytes_per_row as usize)
    {
        img_data.extend_from_slice(&row[0..usize::from(WIDTH) * 4]);
    }
    texture_copy_buffer.unmap();

    let pixmap = Pixmap::from_parts(bytemuck::cast_slice(&img_data).to_vec(), WIDTH, HEIGHT);

    let file = std::fs::File::create(&output_filename).unwrap();
    let w = BufWriter::new(file);
    let mut png_encoder = png::Encoder::new(w, u32::from(WIDTH), u32::from(HEIGHT));
    png_encoder.set_color(png::ColorType::Rgba);
    let mut writer = png_encoder.write_header().unwrap();
    writer
        .write_image_data(bytemuck::cast_slice(&pixmap.take_unpremultiplied()))
        .unwrap();

    println!(
        "wrote {output_filename} — {}x{} grid, {} nested effects per cell",
        COLS, ROWS, 4
    );
}
