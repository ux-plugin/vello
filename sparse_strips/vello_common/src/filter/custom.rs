// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The custom (user-authored WGSL) filter.
//!
//! This is the render-side representation of [`crate::filter_effects::FilterPrimitive::Custom`].
//! It carries the selected effect index and the flattened uniform parameters that are
//! forwarded to the `custom_effect` hook in `filters.wgsl`.

use alloc::vec::Vec;

/// A prepared custom filter.
#[derive(Debug)]
pub struct Custom {
    /// Selects which branch of the `custom_effect` hook runs in the shader.
    pub effect: u32,
    /// Uniform parameters forwarded to the shader.
    pub params: Vec<f32>,
}

impl Custom {
    /// Create a new custom filter from an effect index and its uniform parameters.
    pub fn new(effect: u32, params: Vec<f32>) -> Self {
        Self { effect, params }
    }
}
