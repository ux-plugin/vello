//! Renders a backend-neutral [`render_core::model::Scene`] with Vello.
//!
//! This is the second half of the Phase-1 / approach-B pipeline. render-wasm's converter
//! (`model_export::node_from_shape`, verified by its own tests) projects Skia `Shape`s into
//! exactly these `render_core::model` types; here we consume the same types and draw them with
//! Vello — no Skia involved. Together the two halves are the end-to-end path
//! `Penpot shape → neutral model → Vello pixels`.

use render_core::{geom as g, model as m};
use vello_common::color::AlphaColor;
use vello_common::kurbo::{Affine, BezPath, Ellipse, Point as KPoint, Rect as KRect, Shape};
use vello_example_scenes::{ExampleScene, RenderingContext};

/// A focus scene that draws a neutral model via the backend-agnostic `RenderingContext`.
#[derive(Debug)]
pub struct NeutralModelScene {
    model: m::Scene,
}

impl NeutralModelScene {
    /// Create the scene with a hand-built demo model (using the converter's output types).
    pub fn new() -> Self {
        Self {
            model: demo_model(),
        }
    }
}

impl Default for NeutralModelScene {
    fn default() -> Self {
        Self::new()
    }
}

impl ExampleScene for NeutralModelScene {
    fn render<T: RenderingContext>(
        &mut self,
        ctx: &mut T,
        _resources: &mut T::Resources,
        root: Affine,
    ) {
        for node in &self.model.nodes {
            if node.hidden {
                continue;
            }
            // First solid fill wins (gradients/strokes come in a later increment).
            let Some(fill) = node.fills.iter().find_map(|f| match f {
                m::Fill::Solid(c) => Some(*c),
                _ => None,
            }) else {
                continue;
            };

            ctx.set_transform(root * affine(&node.transform));
            ctx.set_paint(AlphaColor::from_rgba8(fill.r, fill.g, fill.b, fill.a));

            match node.kind {
                m::ShapeKind::Rect => ctx.fill_rect(&krect(node.bounds)),
                m::ShapeKind::Circle => ctx.fill_path(&ellipse_path(node.bounds)),
                m::ShapeKind::Path => {
                    if let Some(path) = &node.path {
                        ctx.fill_path(&bez(path));
                    }
                }
                _ => {}
            }
        }
    }

    fn status(&self) -> Option<String> {
        Some(format!(
            "neutral model → vello · {} nodes",
            self.model.nodes.len()
        ))
    }
}

fn affine(mtx: &g::Matrix) -> Affine {
    // kurbo Affine [a,b,c,d,e,f]: x' = a*x + c*y + e ; y' = b*x + d*y + f
    Affine::new([
        mtx.scale_x() as f64,
        mtx.skew_y() as f64,
        mtx.skew_x() as f64,
        mtx.scale_y() as f64,
        mtx.translate_x() as f64,
        mtx.translate_y() as f64,
    ])
}

fn krect(r: g::Rect) -> KRect {
    KRect::new(r.left as f64, r.top as f64, r.right as f64, r.bottom as f64)
}

fn ellipse_path(r: g::Rect) -> BezPath {
    let c = r.center();
    Ellipse::new(
        KPoint::new(c.x as f64, c.y as f64),
        ((r.width() * 0.5) as f64, (r.height() * 0.5) as f64),
        0.0,
    )
    .to_path(0.1)
}

fn bez(path: &m::Path) -> BezPath {
    let mut bp = BezPath::new();
    for seg in &path.segments {
        match *seg {
            m::PathSeg::MoveTo(p) => bp.move_to((p.x as f64, p.y as f64)),
            m::PathSeg::LineTo(p) => bp.line_to((p.x as f64, p.y as f64)),
            m::PathSeg::CubicTo { c1, c2, end } => bp.curve_to(
                (c1.x as f64, c1.y as f64),
                (c2.x as f64, c2.y as f64),
                (end.x as f64, end.y as f64),
            ),
            m::PathSeg::Close => bp.close_path(),
        }
    }
    bp
}

/// A hand-built neutral scene using the SAME types render-wasm's converter emits: a rect,
/// a circle, and a vector path, each with a solid fill and its own transform.
fn demo_model() -> m::Scene {
    let mut s = m::Scene::new();

    s.push(m::Node {
        id: 1,
        kind: m::ShapeKind::Rect,
        bounds: g::Rect::from_ltrb(0.0, 0.0, 160.0, 100.0),
        path: None,
        transform: g::Matrix::translate(40.0, 60.0),
        fills: vec![m::Fill::Solid(m::Color::rgba(56, 152, 236, 255))],
        opacity: 1.0,
        hidden: false,
    });

    s.push(m::Node {
        id: 2,
        kind: m::ShapeKind::Circle,
        bounds: g::Rect::from_ltrb(0.0, 0.0, 110.0, 110.0),
        path: None,
        transform: g::Matrix::translate(250.0, 55.0),
        fills: vec![m::Fill::Solid(m::Color::rgba(240, 90, 40, 255))],
        opacity: 1.0,
        hidden: false,
    });

    let path = m::Path::new(vec![
        m::PathSeg::MoveTo(g::Point::new(0.0, 0.0)),
        m::PathSeg::LineTo(g::Point::new(120.0, 30.0)),
        m::PathSeg::CubicTo {
            c1: g::Point::new(90.0, 90.0),
            c2: g::Point::new(60.0, 120.0),
            end: g::Point::new(30.0, 150.0),
        },
        m::PathSeg::LineTo(g::Point::new(0.0, 60.0)),
        m::PathSeg::Close,
    ]);
    s.push(m::Node {
        id: 3,
        kind: m::ShapeKind::Path,
        bounds: g::Rect::from_ltrb(0.0, 0.0, 120.0, 150.0),
        path: Some(path),
        transform: g::Matrix::translate(430.0, 40.0),
        fills: vec![m::Fill::Solid(m::Color::rgba(70, 190, 120, 255))],
        opacity: 1.0,
        hidden: false,
    });

    s
}
