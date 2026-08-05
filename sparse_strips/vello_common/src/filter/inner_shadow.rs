// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The inner (inset) shadow filter.
//!
//! The mirror of [`crate::filter::drop_shadow`]: the prepared representation and the offset+blur
//! plan are identical; only the final composite differs (the shadow is drawn *inside* the shape
//! rather than behind it — see `PASS_COMPOSITE_INNER_SHADOW` in `filters.wgsl`).

use crate::color::{AlphaColor, Srgb};
use crate::filter::gaussian_blur::{MAX_KERNEL_SIZE, plan_decimated_blur};
use crate::filter_effects::EdgeMode;

/// An inner shadow filter.
#[derive(Debug)]
pub struct InnerShadow {
    /// The x-offset of the shadow.
    pub dx: f32,
    /// The y-offset of the shadow.
    pub dy: f32,
    /// The color of the shadow.
    pub color: AlphaColor<Srgb>,
    /// Standard deviation for the blur (for reference/debugging).
    pub std_deviation: f32,
    /// Edge mode for blur sampling.
    pub edge_mode: EdgeMode,
    /// Number of 2x2 decimation levels to use (0 means direct convolution).
    pub n_decimations: usize,
    /// Pre-computed Gaussian kernel weights for the reduced blur.
    /// Only the first `kernel_size` elements are valid.
    pub kernel: [f32; MAX_KERNEL_SIZE],
    /// Actual length of the kernel (kernel is padded to `MAX_KERNEL_SIZE`).
    pub kernel_size: u8,
}

impl InnerShadow {
    /// Create a new inner shadow filter with the specified parameters.
    ///
    /// This precomputes the blur decimation plan and kernel, exactly as [`super::drop_shadow`] does.
    pub fn new(
        dx: f32,
        dy: f32,
        std_deviation: f32,
        edge_mode: EdgeMode,
        color: AlphaColor<Srgb>,
    ) -> Self {
        let (n_decimations, kernel, kernel_size) = plan_decimated_blur(std_deviation);

        Self {
            dx,
            dy,
            color,
            std_deviation,
            edge_mode,
            n_decimations,
            kernel,
            kernel_size,
        }
    }
}
