// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Headless demo of the `FilterPrimitive::Custom` WGSL filter.
//!
//! Renders the same shapes twice: left is raw, right is wrapped in a `Custom`
//! filter layer (effect 0 = tint, driven by uniform params). Saves a PNG so the
//! effect is visible without a browser.
//!
//! Run with:
//!   cargo run -p vello_hybrid --example custom_filter -- out.png

use std::io::BufWriter;
use vello_common::color::palette::css;
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Circle, Rect, Shape, Stroke};
use vello_common::pixmap::Pixmap;
use vello_hybrid::{Resources, Scene};

const WIDTH: u16 = 640;
const HEIGHT: u16 = 320;

fn draw_shapes(ctx: &mut Scene) {
    ctx.set_paint(css::DEEP_SKY_BLUE);
    ctx.fill_rect(&Rect::new(60.0, 90.0, 240.0, 230.0));

    ctx.set_paint(css::ORANGE_RED);
    ctx.set_stroke(Stroke::new(8.0));
    ctx.stroke_path(&Circle::new((150.0, 160.0), 72.0).to_path(0.25));
}

fn build_scene() -> Scene {
    let mut scene = Scene::new(WIDTH, HEIGHT);

    // White background.
    scene.set_transform(Affine::IDENTITY);
    scene.set_paint(css::WHITE);
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT)));

    // Left: raw shapes, no filter.
    scene.set_transform(Affine::IDENTITY);
    draw_shapes(&mut scene);

    // Right: the same shapes wrapped in our custom WGSL tint filter.
    // effect 0 = tint; params = [r, g, b, amount].
    let filter = Filter::from_primitive(FilterPrimitive::Custom {
        effect: 0,
        params: [1.0_f32, 0.45, 0.0, 0.7].into_iter().collect(),
        expansion: [0.0, 0.0, 0.0, 0.0],
    });
    let shift = Affine::translate((320.0, 0.0));
    scene.set_transform(shift);
    scene.push_layer(None, None, None, None, Some(filter));
    scene.set_transform(shift);
    draw_shapes(&mut scene);
    scene.pop_layer();

    scene
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let output_filename = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "custom_filter.png".to_string());

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

    println!("wrote {output_filename}");
}
