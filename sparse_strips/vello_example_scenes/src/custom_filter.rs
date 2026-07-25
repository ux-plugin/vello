// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Scene demonstrating the `FilterPrimitive::Custom` WGSL filter.
//!
//! The same shapes are drawn twice: on the left with no filter, on the right
//! wrapped in a `Custom` filter layer (effect `0` = tint, driven by uniform
//! params). The effect body lives in the `custom_effect` hook in `filters.wgsl`.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css;
use vello_common::filter_effects::{Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, Circle, Rect, Shape, Stroke};

/// Scene showing a raw vs. custom-filtered pair of shapes.
#[derive(Debug, Default)]
pub struct CustomFilterScene {}

impl CustomFilterScene {
    /// Create a new `CustomFilterScene`.
    pub fn new() -> Self {
        Self::default()
    }
}

fn draw_shapes<T: RenderingContext>(ctx: &mut T) {
    ctx.set_paint(css::DEEP_SKY_BLUE);
    ctx.fill_rect(&Rect::new(60.0, 90.0, 240.0, 230.0));

    ctx.set_paint(css::ORANGE_RED);
    ctx.set_stroke(Stroke::new(8.0));
    ctx.stroke_path(&Circle::new((150.0, 160.0), 72.0).to_path(0.25));
}

impl ExampleScene for CustomFilterScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        // Left: raw shapes, no filter.
        ctx.set_transform(root_transform);
        draw_shapes(ctx);

        // Right: same shapes wrapped in the custom WGSL tint filter.
        // effect 0 = tint; params = [r, g, b, amount].
        let filter = Filter::from_primitive(FilterPrimitive::Custom {
            effect: 0,
            params: [1.0_f32, 0.45, 0.0, 0.7].into_iter().collect(),
            expansion: [0.0, 0.0, 0.0, 0.0],
        });
        let shift = root_transform * Affine::translate((320.0, 0.0));
        ctx.set_transform(shift);
        ctx.push_filter_layer(filter);
        ctx.set_transform(shift);
        draw_shapes(ctx);
        ctx.pop_layer();
    }

    fn status(&self) -> Option<String> {
        Some("custom wgsl tint (effect 0) — left raw, right filtered".to_string())
    }
}
