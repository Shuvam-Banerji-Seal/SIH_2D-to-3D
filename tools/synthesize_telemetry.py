"""Synthesise a GPS telemetry track for a run whose video carries none.

The shipped sample drone video has no embedded GPS (verified with ffprobe:
only ``language``/``DURATION``/``ENCODER`` tags). That blocks the ``georef``
stage and with it PS criteria 2 and 6.

This script derives a *synthetic* track from the reconstructed camera centres
so the georeferencing path can be exercised and tested end to end. The result
is a legitimate test fixture -- the geometry is self-consistent and the Umeyama
fit should recover it to numerical precision -- but it is **NOT** evidence of
real-world georeferencing accuracy. Only a genuine flight log can provide that.

Output is written to ``<run_dir>/ingest/telemetry_synthetic.csv`` and merged
into ``ingest/frames.csv`` so ``Pipeline._stage_georef`` picks it up.

Usage:
    uv run python tools/synthesize_telemetry.py outputs/sample_fast \
        --lat 12.5 --lon 77.6 --alt 500
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drone3d.geo.enu import enu_to_geodetic  # noqa: E402
from drone3d.sfm.colmap_model import parse_images_text  # noqa: E402

SYNTHETIC_NOTICE = "SYNTHETIC: derived from SfM camera centres, not a real flight log"


def synthesize(
    run_dir: Path,
    *,
    origin_lat: float,
    origin_lon: float,
    origin_alt: float,
    scale: float,
) -> Path:
    """Write a synthetic GPS track derived from the SfM camera centres.

    The model is produced in an arbitrary scale, so ``scale`` maps model units
    (usually metres already) onto real metres before converting to geodetic.
    """
    text_model = run_dir / "sfm" / "sparse_txt"
    images_txt = text_model / "images.txt"
    if not images_txt.is_file():
        raise SystemExit(f"no COLMAP text model at {images_txt}; run the sfm stage first")

    poses = parse_images_text(images_txt)
    if not poses:
        raise SystemExit(f"no poses parsed from {images_txt}")

    frames_csv = run_dir / "ingest" / "frames.csv"
    if not frames_csv.is_file():
        raise SystemExit(f"no frame manifest at {frames_csv}; run the ingest stage first")

    by_name = {pose.name: pose for pose in poses}

    rows: list[dict[str, str]] = []
    with frames_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            pose = by_name.get(Path(row["path"]).name)
            if pose is None:
                rows.append(row)
                continue
            center = pose.center * scale
            lat, lon, alt = enu_to_geodetic(
                float(center[0]),
                float(center[1]),
                float(center[2]),
                lat0=origin_lat,
                lon0=origin_lon,
                alt0=origin_alt,
            )
            row["lat"] = f"{lat:.8f}"
            row["lon"] = f"{lon:.8f}"
            row["alt_m"] = f"{alt:.3f}"
            rows.append(row)

    out_csv = run_dir / "ingest" / "telemetry_synthetic.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Merge into the manifest georef actually reads.
    with frames_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    filled = sum(1 for row in rows if row.get("lat"))
    print(f"{SYNTHETIC_NOTICE}")
    print(f"origin: lat={origin_lat} lon={origin_lon} alt={origin_alt}")
    print(f"matched {filled}/{len(rows)} frames -> {out_csv}")
    print(f"merged into {frames_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="pipeline run directory")
    parser.add_argument("--lat", type=float, default=12.5, help="origin latitude (deg)")
    parser.add_argument("--lon", type=float, default=77.6, help="origin longitude (deg)")
    parser.add_argument("--alt", type=float, default=500.0, help="origin altitude (m)")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="model units -> metres (SfM is usually already metric-ish)",
    )
    args = parser.parse_args()
    synthesize(
        args.run_dir,
        origin_lat=args.lat,
        origin_lon=args.lon,
        origin_alt=args.alt,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
