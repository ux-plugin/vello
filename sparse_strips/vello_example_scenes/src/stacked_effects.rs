// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Stress scene: many effects stacked — the "bad in Skia" case.
//!
//! Each cell draws a filled body plus a wavy stroke, then wraps that content in
//! a deep nest of filter layers:
//!
//!   GaussianBlur -> DropShadow -> Custom(tint) -> GaussianBlur -> content
//!
//! Every nested effect is a separate render-graph node. Zoom in (mouse wheel)
//! and pan (drag) to force a full re-render of the whole stack each frame — the
//! dynamic recompute that a tiled immediate-mode renderer pays per tile.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Rect, Stroke};

const COLS: usize = 5;
const ROWS: usize = 4;
const CELL: f64 = 260.0;

/// Stress scene of a grid of deeply-stacked-effect shapes.
#[derive(Debug, Default)]
pub struct StackedEffectsScene {
    /// Uniform scale of the most recent root transform, shown in the title.
    current_scale: f64,
}

impl StackedEffectsScene {
    /// Create a new `StackedEffectsScene`.
    pub fn new() -> Self {
        Self::default()
    }
}

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

fn draw_content<T: RenderingContext>(ctx: &mut T) {
    ctx.set_paint(css::DEEP_SKY_BLUE);
    ctx.fill_rect(&Rect::new(40.0, 70.0, 220.0, 200.0));

    ctx.set_paint(css::ORANGE_RED);
    ctx.set_stroke(Stroke::new(7.0));
    ctx.stroke_path(&wavy_path(30.0, 135.0, 200.0, 34.0, 2.5));
}

fn draw_stacked_cell<T: RenderingContext>(ctx: &mut T, base: Affine) {
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
        ctx.push_filter_layer(filter.clone());
    }

    ctx.set_transform(base);
    draw_content(ctx);

    for _ in &stack {
        ctx.pop_layer();
    }
}

impl ExampleScene for StackedEffectsScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        self.current_scale = root_transform.determinant().abs().sqrt();

        for row in 0..ROWS {
            for col in 0..COLS {
                let base = root_transform
                    * Affine::translate((col as f64 * CELL, row as f64 * CELL));
                draw_stacked_cell(ctx, base);
            }
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "{} cells x 4 nested effects | scale {:.2}x",
            COLS * ROWS,
            self.current_scale
        ))
    }
}
