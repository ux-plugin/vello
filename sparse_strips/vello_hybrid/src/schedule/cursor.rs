// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Monotonic allocation cursor for scheduled rounds.

use crate::schedule::allocate::{
    AllocatedTextureRegion, Allocation, Atlases, LayerAllocationRequest,
};
use crate::target::LayerTextureId;
use alloc::vec::Vec;
use vello_common::multi_atlas::AtlasError;

#[derive(Debug)]
pub(super) struct Cursor {
    current_round: usize,
    atlases: Atlases,
    pending_releases: Vec<Vec<AllocatedTextureRegion<LayerTextureId>>>,
}

impl Cursor {
    pub(super) fn new(atlases: Atlases) -> Self {
        Self {
            current_round: 0,
            atlases,
            pending_releases: Vec::new(),
        }
    }

    pub(super) fn current_round(&self) -> usize {
        self.current_round
    }

    pub(super) fn scratch_texture(&self) -> bool {
        self.atlases.scratch_texture()
    }

    pub(super) fn require_scratch_texture(&mut self) -> Result<(), AtlasError> {
        self.atlases.require_scratch_texture()
    }

    pub(super) fn allocate_layer(
        &mut self,
        request: LayerAllocationRequest,
    ) -> Result<Allocation<AllocatedTextureRegion<LayerTextureId>>, AtlasError> {
        if let Some(allocation) =
            self.allocate_reusing(|atlases| Ok(atlases.allocate_layer(&request)))?
        {
            return Ok(allocation);
        }

        // The currently available layer textures do not have enough room to store our layer.
        // Therefore, we need to create a new one.

        self.atlases.add_layer_atlas(request.texture_parity)?;
        let allocation = self
            .atlases
            .allocate_layer(&request)
            .ok_or(AtlasError::NoSpaceAvailable)?;

        Ok(Allocation {
            allocation,
            round_idx: self.current_round,
        })
    }

    /// Advance the round counter until enough resources have been freed such that
    /// the given allocation succeeds.
    ///
    /// Return `Ok(None)` in case it's not possible to perform the allocation using the
    /// currently available resources.
    fn allocate_reusing<T: Copy>(
        &mut self,
        mut allocate: impl FnMut(&mut Atlases) -> Result<Option<T>, AtlasError>,
    ) -> Result<Option<Allocation<T>>, AtlasError> {
        loop {
            if let Some(allocation) = allocate(&mut self.atlases)? {
                return Ok(Some(Allocation {
                    allocation,
                    round_idx: self.current_round,
                }));
            }

            if self.current_round >= self.pending_releases.len() {
                return Ok(None);
            }

            self.advance();
        }
    }

    pub(super) fn release(
        &mut self,
        allocation: AllocatedTextureRegion<LayerTextureId>,
        round_idx: usize,
    ) {
        assert!(
            round_idx >= self.current_round,
            "cannot release an allocation in a round already passed by the cursor"
        );

        while self.pending_releases.len() <= round_idx {
            self.pending_releases.push(Vec::new());
        }

        self.pending_releases[round_idx].push(allocation);
    }

    /// Advance to the next round.
    fn advance(&mut self) {
        if let Some(releases) = self.pending_releases.get_mut(self.current_round) {
            for allocation in releases.drain(..) {
                self.atlases.deallocate(allocation);
            }
        }

        self.current_round += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::Cursor;
    use crate::scene::LayersConfig;
    use crate::schedule::allocate::{Atlases, LayerAllocationRequest};
    use crate::schedule::{LayerSamplePlacement, OpenLayer};
    use crate::target::{LayerTextureId, TextureParity};
    use vello_common::geometry::{RectU16, SizeU16};
    use vello_common::multi_atlas::AtlasError;
    use vello_common::record::RecordedLayerKind;

    fn cursor(max_textures: usize) -> Cursor {
        let config = LayersConfig {
            max_textures: Some(max_textures),
            ..Default::default()
        };

        Cursor::new(Atlases::new(SizeU16::new(8), config))
    }

    fn request(texture_parity: TextureParity, size: SizeU16) -> LayerAllocationRequest {
        let kind = RecordedLayerKind::Regular;
        let bbox = RectU16::new(0, 0, size.width(), size.height());
        let layer = OpenLayer {
            cmds: &[],
            kind: &kind,
            texture_parity,
            bbox,
            sample: LayerSamplePlacement::regular(bbox),
            target: None,
        };

        LayerAllocationRequest::new(&layer)
    }

    #[test]
    fn current_space() {
        let mut cursor = cursor(1);
        let request = request(TextureParity::Even, SizeU16::from_wh(4, 8));

        let first = cursor.allocate_layer(request).unwrap();
        let second = cursor.allocate_layer(request).unwrap();

        assert_eq!(first.round_idx, 0);
        assert_eq!(second.round_idx, 0);
        assert_eq!(cursor.current_round(), 0);
        assert_eq!(
            first.allocation.region.target,
            second.allocation.region.target
        );
        assert_ne!(first.allocation.region.rect, second.allocation.region.rect);
    }

    #[test]
    fn deferred_reuse() {
        let mut cursor = cursor(1);
        let request = request(TextureParity::Even, SizeU16::new(8));
        let first = cursor.allocate_layer(request).unwrap();
        cursor.release(first.allocation, 2);

        let reused = cursor.allocate_layer(request).unwrap();

        assert_eq!(reused.round_idx, 3);
        assert_eq!(cursor.current_round(), 3);
    }

    #[test]
    fn page_growth() {
        let mut cursor = cursor(3);
        let even = request(TextureParity::Even, SizeU16::new(8));
        let odd = request(TextureParity::Odd, SizeU16::new(8));
        cursor.allocate_layer(even).unwrap();
        let released = cursor.allocate_layer(odd).unwrap();
        cursor.release(released.allocation, 1);

        let grown = cursor.allocate_layer(even).unwrap();

        assert_eq!(grown.round_idx, 2);
        assert_eq!(cursor.current_round(), 2);
        assert_eq!(
            grown.allocation.region.target,
            LayerTextureId::new(TextureParity::Even, 1)
        );
    }

    #[test]
    fn texture_limit() {
        let mut cursor = cursor(2);
        let even = request(TextureParity::Even, SizeU16::new(8));
        let odd = request(TextureParity::Odd, SizeU16::new(8));
        cursor.allocate_layer(even).unwrap();
        let released = cursor.allocate_layer(odd).unwrap();
        cursor.release(released.allocation, 1);

        assert!(matches!(
            cursor.allocate_layer(even),
            Err(AtlasError::NoSpaceAvailable)
        ));
    }

    #[test]
    #[should_panic(expected = "cannot release an allocation in a round already passed")]
    fn past_release() {
        let mut cursor = cursor(1);
        let allocation = cursor
            .allocate_layer(request(TextureParity::Even, SizeU16::new(8)))
            .unwrap();
        cursor.advance();

        cursor.release(allocation.allocation, 0);
    }
}
