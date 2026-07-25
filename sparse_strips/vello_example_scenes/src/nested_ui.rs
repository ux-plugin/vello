// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Growable nested-UI stress scene.
//!
//! A grid of "UI cards" that grows by one card per Up-arrow press (Down to shrink).
//! Each card is a genuinely nested stack of effect layers, mimicking a real design
//! surface:
//!
//!   DropShadow (card elevation)
//!     ├─ body + header + content lines
//!     ├─ GaussianBlur  (inner "frosted" avatar — nested multi-pass effect)
//!     └─ Custom(tint)  (badge — nested single-pass effect)
//!
//! So every card contributes three nested filter-layer nodes (two multi-pass, one
//! single-pass). Press Up to add cards and watch the fps counter: this is the test
//! for "many nested nodes with effects" scaling.

use crate::{ExampleScene, RenderingContext};
use vello_common::color::palette::css;
use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
use vello_common::kurbo::{Affine, BezPath, Rect, Shape};

const CARD_W: f64 = 210.0;
const CARD_H: f64 = 120.0;
const GAP: f64 = 20.0;

/// Nested filter-layer nodes emitted per card (shadow + blur + tint).
const LAYERS_PER_CARD: usize = 3;

/// A growable grid of nested-effect UI cards.
#[derive(Debug)]
pub struct NestedUiScene {
    /// Number of cards currently drawn.
    count: usize,
    /// Uniform scale of the most recent root transform, shown in the title.
    current_scale: f64,
}

impl Default for NestedUiScene {
    fn default() -> Self {
        Self {
            count: 6,
            current_scale: 1.0,
        }
    }
}

impl NestedUiScene {
    /// Create a new `NestedUiScene`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// A rounded-rectangle path in local card space.
fn rounded(x0: f64, y0: f64, x1: f64, y1: f64, r: f64) -> BezPath {
    Rect::new(x0, y0, x1, y1).to_rounded_rect(r).to_path(0.1)
}

fn drop_shadow() -> Filter {
    Filter::from_primitive(FilterPrimitive::DropShadow {
        dx: 6.0,
        dy: 8.0,
        std_deviation: 6.0,
        color: css::BLACK,
        edge_mode: EdgeMode::None,
    })
}

fn frosted_blur() -> Filter {
    Filter::from_primitive(FilterPrimitive::GaussianBlur {
        std_deviation: 3.0,
        edge_mode: EdgeMode::None,
    })
}

fn tint() -> Filter {
    Filter::from_primitive(FilterPrimitive::Custom {
        effect: 0,
        params: [0.95_f32, 0.35, 0.1, 0.5].into_iter().collect(),
        expansion: [0.0, 0.0, 0.0, 0.0],
    })
}

/// Draw one card and its nested effect layers. `base` places the card's local
/// (0,0)-(CARD_W,CARD_H) space on the canvas.
fn draw_card<T: RenderingContext>(ctx: &mut T, base: Affine) {
    // Card elevation: a drop-shadow layer wrapping the whole card.
    ctx.set_transform(base);
    ctx.push_filter_layer(drop_shadow());

    // Body.
    ctx.set_transform(base);
    ctx.set_paint(css::WHITE_SMOKE);
    ctx.fill_path(&rounded(2.0, 2.0, CARD_W - 2.0, CARD_H - 2.0, 12.0));

    // Header bar.
    ctx.set_paint(css::STEEL_BLUE);
    ctx.fill_path(&rounded(2.0, 2.0, CARD_W - 2.0, 36.0, 12.0));

    // Nested frosted avatar — an inner Gaussian-blur layer (multi-pass).
    ctx.set_transform(base * Affine::translate((14.0, 48.0)));
    ctx.push_filter_layer(frosted_blur());
    ctx.set_paint(css::DEEP_SKY_BLUE);
    ctx.fill_path(&rounded(0.0, 0.0, 44.0, 44.0, 22.0));
    ctx.pop_layer();

    // Nested tinted badge — an inner Custom (single-pass) layer.
    ctx.set_transform(base * Affine::translate((CARD_W - 50.0, 9.0)));
    ctx.push_filter_layer(tint());
    ctx.set_paint(css::ORANGE_RED);
    ctx.fill_path(&rounded(0.0, 0.0, 38.0, 18.0, 9.0));
    ctx.pop_layer();

    // Content lines.
    ctx.set_transform(base);
    ctx.set_paint(css::LIGHT_STEEL_BLUE);
    ctx.fill_rect(&Rect::new(70.0, 50.0, CARD_W - 16.0, 60.0));
    ctx.set_paint(css::GAINSBORO);
    ctx.fill_rect(&Rect::new(70.0, 66.0, CARD_W - 36.0, 74.0));
    ctx.fill_rect(&Rect::new(16.0, 100.0, CARD_W - 16.0, 108.0));

    ctx.pop_layer(); // drop-shadow
}

impl ExampleScene for NestedUiScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root_transform: Affine,
    ) {
        self.current_scale = root_transform.determinant().abs().sqrt();

        // Keep the grid roughly square as it grows.
        let cols = (self.count as f64).sqrt().ceil().max(1.0) as usize;

        for i in 0..self.count {
            let col = i % cols;
            let row = i / cols;
            let base = root_transform
                * Affine::translate((
                    col as f64 * (CARD_W + GAP),
                    row as f64 * (CARD_H + GAP),
                ));
            draw_card(ctx, base);
        }
    }

    fn handle_key(&mut self, key: &str) -> bool {
        match key {
            "ArrowUp" => {
                // Cap to keep an accidental key-repeat from locking up the tab.
                self.count = (self.count + 1).min(256);
                true
            }
            "ArrowDown" => {
                self.count = self.count.saturating_sub(1).max(1);
                true
            }
            _ => false,
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "{} cards · {} nested effect layers · scale {:.2}x · ↑/↓ to add/remove",
            self.count,
            self.count * LAYERS_PER_CARD,
            self.current_scale
        ))
    }
}
