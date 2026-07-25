// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Mock host for the embeddable [`focus_embed::FocusRenderer`].
//!
//! This stands in for Penpot's focus-mode JS/CLJS host: it creates the canvas,
//! constructs the renderer, **owns the `requestAnimationFrame` loop**, and forwards
//! input — calling the module's `render()` / `resize()` / `key()` from the outside.
//! The renderer itself owns none of this; that separation is the Phase-0 point.

#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;

use focus_embed::{FocusRenderer, create_focus_renderer};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, KeyboardEvent};

#[wasm_bindgen]
extern "C" {
    fn requestAnimationFrame(closure: &Closure<dyn FnMut()>) -> u32;
}

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Debug).unwrap();
    wasm_bindgen_futures::spawn_local(async {
        host().await;
    });
}

async fn host() {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let body = document.body().unwrap();
    body.style().set_property("background-color", "#111").unwrap();
    body.style().set_property("margin", "0").unwrap();

    let dpr = window.device_pixel_ratio();
    let css_w = window.inner_width().unwrap().as_f64().unwrap();
    let css_h = window.inner_height().unwrap().as_f64().unwrap();

    // The HOST creates the canvas — the renderer only mounts onto it.
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<HtmlCanvasElement>()
        .unwrap();
    canvas.set_width((css_w * dpr) as u32);
    canvas.set_height((css_h * dpr) as u32);
    canvas.style().set_property("width", "100%").unwrap();
    canvas.style().set_property("height", "100%").unwrap();
    canvas.style().set_property("display", "block").unwrap();
    body.append_child(&canvas).unwrap();

    // Status overlay (host chrome, not part of the module).
    let status = document.create_element("div").unwrap();
    status.set_id("focus-status");
    status.set_text_content(Some("Vello focus module (Phase 0) — starting…"));
    let s = status.dyn_ref::<web_sys::HtmlElement>().unwrap().style();
    s.set_property("position", "fixed").unwrap();
    s.set_property("bottom", "10px").unwrap();
    s.set_property("left", "10px").unwrap();
    s.set_property("background", "rgba(0,0,0,0.6)").unwrap();
    s.set_property("color", "#4ade80").unwrap();
    s.set_property("padding", "5px 10px").unwrap();
    s.set_property("border-radius", "5px").unwrap();
    s.set_property("font-family", "monospace").unwrap();
    s.set_property("pointer-events", "none").unwrap();
    body.append_child(&status).unwrap();

    let hint = document.create_element("div").unwrap();
    hint.set_text_content(Some(
        "Host-driven Vello module · ←/→ scene · ↑/↓ grow · Q/E rotate · Space reset",
    ));
    let hs = hint.dyn_ref::<web_sys::HtmlElement>().unwrap().style();
    hs.set_property("position", "fixed").unwrap();
    hs.set_property("top", "10px").unwrap();
    hs.set_property("left", "10px").unwrap();
    hs.set_property("background", "rgba(0,0,0,0.5)").unwrap();
    hs.set_property("color", "white").unwrap();
    hs.set_property("padding", "5px 10px").unwrap();
    hs.set_property("border-radius", "5px").unwrap();
    hs.set_property("font-family", "sans-serif").unwrap();
    hs.set_property("pointer-events", "none").unwrap();
    body.append_child(&hint).unwrap();

    // The host constructs the module and holds it.
    let renderer = Rc::new(RefCell::new(create_focus_renderer(canvas).await));

    // A tiny rotation angle so ←/→/Q/E feel alive; the host owns the transform.
    let rot: Rc<RefCell<f64>> = Rc::new(RefCell::new(0.0));

    // Keyboard: the host routes input into the module.
    {
        let renderer = renderer.clone();
        let rot = rot.clone();
        let closure = Closure::wrap(Box::new(move |event: KeyboardEvent| {
            let Ok(mut r) = renderer.try_borrow_mut() else {
                return;
            };
            match event.key().as_str() {
                "ArrowRight" => {
                    let i = next_index(&r, 1);
                    r.set_scene(i);
                }
                "ArrowLeft" => {
                    let i = next_index(&r, -1);
                    r.set_scene(i);
                }
                " " => {
                    *rot.borrow_mut() = 0.0;
                    r.reset_transform();
                }
                "q" | "Q" => {
                    *rot.borrow_mut() += 0.05;
                    apply_rot(&mut r, *rot.borrow());
                }
                "e" | "E" => {
                    *rot.borrow_mut() -= 0.05;
                    apply_rot(&mut r, *rot.borrow());
                }
                other => {
                    // ArrowUp / ArrowDown etc. go to the scene.
                    r.key(other);
                }
            }
        }) as Box<dyn FnMut(_)>);
        document
            .add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // Resize: the host tells the module its new backing size.
    {
        let renderer = renderer.clone();
        let window2 = window.clone();
        let closure = Closure::wrap(Box::new(move || {
            let dpr = window2.device_pixel_ratio();
            let w = window2.inner_width().unwrap().as_f64().unwrap() * dpr;
            let h = window2.inner_height().unwrap().as_f64().unwrap() * dpr;
            if let Ok(mut r) = renderer.try_borrow_mut() {
                r.resize(w as u32, h as u32);
            }
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
            .unwrap();
        closure.forget();
    }

    // The HOST owns the animation-frame loop and calls render() each frame.
    let f = Rc::new(RefCell::new(None));
    let g = f.clone();
    let status_el = status;
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        if let Ok(mut r) = renderer.try_borrow_mut() {
            r.render();
            let scene_status = r.status().unwrap_or_default();
            status_el.set_text_content(Some(&format!(
                "Vello focus module (Phase 0) · host-driven · {scene_status}"
            )));
        }
        requestAnimationFrame(f.borrow().as_ref().unwrap());
    }) as Box<dyn FnMut()>));
    requestAnimationFrame(g.borrow().as_ref().unwrap());
}

/// Host-side scene index stepping without duplicating the module's counter:
/// we just ask the module for its scene count and rotate through.
fn next_index(_r: &FocusRenderer, _dir: i32) -> usize {
    // The module owns `current`; for the mock host we simply advance by 1 each press
    // using a static counter kept here.
    thread_local! {
        static IDX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    IDX.with(|c| {
        let count = _r.scene_count().max(1);
        let next = if _dir >= 0 {
            (c.get() + 1) % count
        } else {
            (c.get() + count - 1) % count
        };
        c.set(next);
        next
    })
}

fn apply_rot(r: &mut FocusRenderer, angle: f64) {
    // Rotate about a rough canvas center via a column-major affine.
    let (sin, cos) = angle.sin_cos();
    // Pivot ~ (400, 300) device px; good enough for the mock host.
    let (px, py) = (400.0_f64, 300.0_f64);
    // T(p) * R * T(-p) composed into (a,b,c,d,e,f).
    let a = cos;
    let b = sin;
    let c = -sin;
    let d = cos;
    let e = px - cos * px + sin * py;
    let f = py - sin * px - cos * py;
    r.set_transform(a, b, c, d, e, f);
}
