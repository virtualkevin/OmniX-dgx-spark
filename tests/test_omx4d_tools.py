#!/usr/bin/env python3
"""Focused contract tests for the offline OMX4D sampler and writer."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from omx4d_tools.config import IngestionLimits  # noqa: E402
from omx4d_tools.converter import (  # noqa: E402
    ConversionOptions,
    convert_pt_file,
)
from omx4d_tools.omx4d import read_manifest  # noqa: E402


class Omx4dToolsTest(unittest.TestCase):
    def make_predictions(self) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        sources, frames, height, width = 4, 2, 2, 5
        identities = torch.arange(sources * height * width, dtype=torch.float32)
        identities = identities.reshape(sources, height, width)
        trajectory = torch.empty(
            (sources, frames, height, width, 3), dtype=torch.float32
        )
        for frame in range(frames):
            trajectory[:, frame, :, :, 0] = identities
            trajectory[:, frame, :, :, 1] = float(frame)
            trajectory[:, frame, :, :, 2] = identities / 10.0

        camera_pose = torch.zeros((sources, 3, 4), dtype=torch.float32)
        camera_pose[:, :3, :3] = torch.eye(3)
        intrinsics = torch.eye(3, dtype=torch.float32).repeat(sources, 1, 1)
        dynamic_score = (
            torch.arange(sources * height * width, dtype=torch.float32)
            .reshape(sources, height, width)
            .div(sources * height * width - 1)
        )
        # The model can exceed one by a float32 ULP; serialization must clamp
        # this while preserving the raw score for ranking.
        dynamic_score[2, 1, 4] = 1.0 + torch.finfo(torch.float32).eps
        source_rgb = torch.zeros((sources, height, width, 3), dtype=torch.uint8)
        source_rgb[..., 0] = torch.arange(sources, dtype=torch.uint8)[:, None, None]
        source_rgb[..., 1] = torch.arange(width, dtype=torch.uint8)[None, None, :]
        source_rgb[..., 2] = torch.arange(height, dtype=torch.uint8)[None, :, None]
        return {
            "trajectory": trajectory,
            "camera_pose": camera_pose,
            "intrinsics": intrinsics,
            "pts3d_dynamic_score": dynamic_score,
        }, source_rgb

    def test_exact_80_20_selection_and_binary_contract(self) -> None:
        raw, source_rgb = self.make_predictions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "predictions.pt"
            output_path = root / "sample.omx4d"
            torch.save(raw, input_path)
            result = convert_pt_file(
                input_path,
                output_path,
                options=ConversionOptions(
                    point_budget=10,
                    fps=24.0,
                    candidate_source_view_count=3,
                ),
                limits=IngestionLimits(
                    max_point_budget=10,
                    max_source_pixels=1_000,
                    max_output_bytes=1_000_000,
                ),
                source_rgb=source_rgb,
            )

            manifest = result.manifest
            sampling = manifest["sampling"]
            self.assertEqual(manifest["pointCount"], 10)
            self.assertEqual(manifest["frameCount"], 2)
            self.assertEqual(sampling["dynamicThreshold"], 0.0)
            self.assertEqual(sampling["dynamicReservedFraction"], 0.8)
            self.assertEqual(sampling["dynamicSelectedPointCount"], 8)
            self.assertEqual(
                sampling["dynamicRanking"],
                "global-descending-stable-identity-tiebreak",
            )
            self.assertAlmostEqual(sampling["dynamicScoreCutoff"], 22 / 39)
            self.assertEqual(sampling["spatialSelectedPointCount"], 2)
            self.assertEqual(
                sampling["spatialDistribution"],
                "normalized-frame-zero-3d-voxel",
            )
            self.assertEqual(sampling["candidateSourceViewCount"], 3)
            self.assertEqual(sampling["excludedPaddedSourceViewCount"], 1)
            self.assertEqual(sampling["validCandidateCount"], 30)

            attributes = manifest["attributes"]
            self.assertEqual(attributes["positions"]["shape"], [2, 10, 3])
            self.assertEqual(attributes["colors"]["shape"], [10, 3])
            self.assertEqual(attributes["sourceView"]["shape"], [10])
            for descriptor in attributes.values():
                self.assertEqual(descriptor["offset"] % 8, 0)

            reread = read_manifest(output_path)
            self.assertEqual(reread, manifest)
            score_descriptor = attributes["dynamicScore"]
            scores = np.memmap(
                output_path,
                mode="r",
                dtype="<f4",
                offset=score_descriptor["offset"],
                shape=(10,),
            )
            # Only the first three source views are candidates. Their top eight
            # identities are 22..29; both spatial points come from 0..21.
            cutoff = np.float32(22 / 39)
            self.assertEqual(int(np.count_nonzero(scores >= cutoff)), 8)
            self.assertEqual(float(scores.max()), 1.0)

            source_descriptor = attributes["sourceView"]
            source_views = np.memmap(
                output_path,
                mode="r",
                dtype="<u2",
                offset=source_descriptor["offset"],
                shape=(10,),
            )
            self.assertLess(int(source_views.max()), 3)

            position_descriptor = attributes["positions"]
            positions = np.memmap(
                output_path,
                mode="r",
                dtype="<f4",
                offset=position_descriptor["offset"],
                shape=(2, 10, 3),
            )
            self.assertEqual(len(np.unique(positions[0, :, 0])), 10)
            np.testing.assert_allclose(positions[1, :, 0], positions[0, :, 0])


if __name__ == "__main__":
    unittest.main()
