"""Tests for telemetry ingestion: CSV, DJI SRT, GPX and JSON parsing + interpolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from drone3d.exceptions import IngestionError
from drone3d.io.telemetry import (
    attach_telemetry,
    interpolate_sample,
    load_telemetry,
    parse_csv_telemetry,
    parse_dji_srt,
    parse_gpx,
    parse_json_telemetry,
    telemetry_summary,
)
from drone3d.types import FrameRecord, TelemetrySample

# --- CSV -------------------------------------------------------------------


def test_parse_csv_basic(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("t,lat,lon,alt_m\n0.0,12.5,77.6,100.0\n1.0,12.6,77.7,110.0\n")

    samples = parse_csv_telemetry(path)

    assert len(samples) == 2
    assert samples[0].lat == pytest.approx(12.5)
    assert samples[1].lon == pytest.approx(77.7)
    assert samples[1].t == pytest.approx(1.0)
    assert samples[0].source.startswith("csv:")


def test_parse_csv_resolves_column_aliases(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text(
        "timestamp,latitude,longitude,absolute_altitude\n0.0,12.5,77.6,100.0\n2.0,12.6,77.7,110.0\n"
    )

    samples = parse_csv_telemetry(path)

    assert len(samples) == 2
    assert samples[0].lat == pytest.approx(12.5)
    assert samples[0].lon == pytest.approx(77.6)
    assert samples[0].alt_m == pytest.approx(100.0)


def test_parse_csv_without_timestamp_column_uses_row_index(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("lat,lon\n12.5,77.6\n12.6,77.7\n12.7,77.8\n")

    samples = parse_csv_telemetry(path)

    assert [s.t for s in samples] == [0.0, 1.0, 2.0]


def test_parse_csv_semicolon_delimiter(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("t;lat;lon\n0.0;12.5;77.6\n1.0;12.6;77.7\n")

    samples = parse_csv_telemetry(path)

    assert len(samples) == 2
    assert samples[1].lat == pytest.approx(12.6)


def test_parse_csv_treats_placeholder_values_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("t,lat,lon,alt_m\n0.0,12.5,77.6,-\n1.0,12.6,77.7,\n")

    samples = parse_csv_telemetry(path)

    assert samples[0].alt_m is None
    assert samples[1].alt_m is None


def test_parse_csv_rejects_headerless(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text("\n\n")

    with pytest.raises(IngestionError):
        parse_csv_telemetry(path)


def test_parse_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        parse_csv_telemetry(tmp_path / "nope.csv")


# --- DJI SRT ---------------------------------------------------------------


def test_parse_dji_srt(tmp_path: Path) -> None:
    path = tmp_path / "clip.srt"
    path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "latitude: 12.5000, longitude: 77.6000\n"
        "ABS_ALT: 100.0 REL_ALT: 50.0 heading: 90.0\n"
        "\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "latitude: 12.5010, longitude: 77.6010\n"
        "ABS_ALT: 101.0 REL_ALT: 51.0 heading: 92.0\n"
    )

    samples = parse_dji_srt(path)

    assert len(samples) == 2
    assert samples[0].lat == pytest.approx(12.5)
    assert samples[0].lon == pytest.approx(77.6)
    assert samples[0].alt_m == pytest.approx(100.0)
    assert samples[0].rel_alt_m == pytest.approx(50.0)
    assert samples[0].heading_deg == pytest.approx(90.0)
    assert samples[1].t == pytest.approx(1.0)


def test_parse_dji_srt_skips_blocks_without_position(tmp_path: Path) -> None:
    path = tmp_path / "clip.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nsome text only\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nlatitude: 12.5, longitude: 77.6\n"
    )

    samples = parse_dji_srt(path)

    assert len(samples) == 1
    assert samples[0].lat == pytest.approx(12.5)


def test_parse_dji_srt_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        parse_dji_srt(tmp_path / "nope.srt")


# --- GPX -------------------------------------------------------------------


def test_parse_gpx_track_points(tmp_path: Path) -> None:
    path = tmp_path / "track.gpx"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">\n'
        "  <trk><trkseg>\n"
        '    <trkpt lat="12.5" lon="77.6"><ele>100.0</ele></trkpt>\n'
        '    <trkpt lat="12.6" lon="77.7"><ele>110.0</ele></trkpt>\n'
        "  </trkseg></trk>\n"
        "</gpx>\n"
    )

    samples = parse_gpx(path)

    assert len(samples) == 2
    assert samples[0].lat == pytest.approx(12.5)
    assert samples[1].alt_m == pytest.approx(110.0)


def test_parse_gpx_skips_points_without_lat_lon(tmp_path: Path) -> None:
    path = tmp_path / "track.gpx"
    path.write_text(
        '<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        '<trkpt lat="12.5"><ele>1.0</ele></trkpt>'
        '<trkpt lat="12.6" lon="77.7"><ele>2.0</ele></trkpt>'
        "</trkseg></trk></gpx>"
    )

    samples = parse_gpx(path)

    assert len(samples) == 1
    assert samples[0].lon == pytest.approx(77.7)


def test_parse_gpx_invalid_xml_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.gpx"
    path.write_text("<gpx><trk>")

    with pytest.raises(IngestionError):
        parse_gpx(path)


# --- JSON ------------------------------------------------------------------


def test_parse_json_list(tmp_path: Path) -> None:
    path = tmp_path / "log.json"
    path.write_text('[{"t": 0.0, "lat": 12.5, "lon": 77.6}, {"t": 1.0, "lat": 12.6, "lon": 77.7}]')

    samples = parse_json_telemetry(path)

    assert len(samples) == 2
    assert samples[1].lat == pytest.approx(12.6)


@pytest.mark.parametrize("key", ["samples", "telemetry", "gps", "data", "records"])
def test_parse_json_wrapped_list(tmp_path: Path, key: str) -> None:
    path = tmp_path / "log.json"
    path.write_text(f'{{"{key}": [{{"lat": 12.5, "lon": 77.6}}]}}')

    samples = parse_json_telemetry(path)

    assert len(samples) == 1
    assert samples[0].lat == pytest.approx(12.5)


def test_parse_json_invalid_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json")

    with pytest.raises(IngestionError):
        parse_json_telemetry(path)


# --- dispatch --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("log.csv", "t,lat,lon\n0.0,12.5,77.6\n"),
        ("log.json", '[{"lat": 12.5, "lon": 77.6}]'),
        (
            "log.gpx",
            '<gpx xmlns="http://www.topografix.com/GPX/1/1">'
            '<trk><trkseg><trkpt lat="12.5" lon="77.6"/></trkseg></trk></gpx>',
        ),
    ],
)
def test_load_telemetry_dispatches_by_extension(tmp_path: Path, name: str, body: str) -> None:
    path = tmp_path / name
    path.write_text(body)

    samples = load_telemetry(path)

    assert len(samples) == 1


def test_load_telemetry_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "log.xlsx"
    path.write_text("x")

    with pytest.raises(IngestionError):
        load_telemetry(path)


# --- interpolation ---------------------------------------------------------


def _track() -> list[TelemetrySample]:
    return [
        TelemetrySample(t=0.0, lat=10.0, lon=20.0, alt_m=100.0),
        TelemetrySample(t=10.0, lat=12.0, lon=24.0, alt_m=200.0),
    ]


def test_interpolate_midpoint_blends_linearly() -> None:
    sample = interpolate_sample(_track(), 5.0)

    assert sample is not None
    assert sample.lat == pytest.approx(11.0)
    assert sample.lon == pytest.approx(22.0)
    assert sample.alt_m == pytest.approx(150.0)
    assert sample.t == pytest.approx(5.0)
    assert sample.source == "interpolated"


def test_interpolate_clamps_outside_range() -> None:
    track = _track()

    assert interpolate_sample(track, -5.0) is track[0]
    assert interpolate_sample(track, 99.0) is track[1]


def test_interpolate_empty_returns_none() -> None:
    assert interpolate_sample([], 1.0) is None


def test_interpolate_single_sample_returns_it() -> None:
    only = [TelemetrySample(t=3.0, lat=1.0, lon=2.0)]

    assert interpolate_sample(only, 99.0) is only[0]


def test_interpolate_handles_partial_fields() -> None:
    track = [
        TelemetrySample(t=0.0, lat=10.0, alt_m=100.0),
        TelemetrySample(t=10.0, lat=12.0, alt_m=200.0),
    ]

    sample = interpolate_sample(track, 5.0)

    assert sample is not None
    assert sample.lat == pytest.approx(11.0)
    assert sample.lon is None


# --- attach / summary ------------------------------------------------------


def test_attach_telemetry_fills_frame_positions() -> None:
    frames = [FrameRecord(index=0, timestamp_s=5.0, path=Path("frame_000000.jpg"))]

    filled = attach_telemetry(frames, _track())

    assert filled[0].lat == pytest.approx(11.0)
    assert filled[0].lon == pytest.approx(22.0)
    assert filled[0].alt_m == pytest.approx(150.0)


def test_attach_telemetry_with_no_samples_is_noop() -> None:
    frames = [FrameRecord(index=0, timestamp_s=5.0, path=Path("f.jpg"))]

    assert attach_telemetry(frames, []) is frames
    assert frames[0].lat is None


def test_telemetry_summary_stats() -> None:
    summary = telemetry_summary(_track())

    assert summary["n"] == 2
    assert summary["duration_s"] == pytest.approx(10.0)
    assert summary["has_position"] is True
    assert summary["bbox"]["min_lat"] == pytest.approx(10.0)
    assert summary["bbox"]["max_lon"] == pytest.approx(24.0)


def test_telemetry_summary_empty() -> None:
    assert telemetry_summary([]) == {"n": 0}
