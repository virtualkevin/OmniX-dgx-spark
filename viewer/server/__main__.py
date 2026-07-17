"""CLI converter for fixture generation and local diagnostics."""

from __future__ import annotations

import argparse
import json

from .config import IngestionLimits
from .converter import ConversionOptions, convert_pt_file
from .errors import ConversionError
from .images import load_source_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert OmniX predictions.pt to OMX4D")
    parser.add_argument("input", help="Path to the restricted OmniX predictions.pt")
    parser.add_argument("output", help="Destination .omx4d path")
    parser.add_argument("--point-budget", type=int, default=100_000)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--name", default="OmniX predictions")
    parser.add_argument("--image-dir", help="Optional sorted companion RGB image directory")
    args = parser.parse_args()
    try:
        source_rgb = load_source_rgb(args.image_dir) if args.image_dir else None
        result = convert_pt_file(
            args.input,
            args.output,
            options=ConversionOptions(
                point_budget=args.point_budget, fps=args.fps, name=args.name
            ),
            limits=IngestionLimits.from_env(),
            source_rgb=source_rgb,
        )
    except ConversionError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True))
        return 2
    print(json.dumps(result.manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
