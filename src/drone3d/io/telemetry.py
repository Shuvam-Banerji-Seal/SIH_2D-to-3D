"""Telemetry ingestion: CSV, DJI-style SRT, GPX and JSON navigation logs."""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from typing import Any

from drone3d.exceptions import IngestionError
from drone3d.geo.enu import geodetic_to_enu
from drone3d.logging_utils import get_logger
from drone3d.types import FrameRecord, TelemetrySample

__all__ = [
    "attach_telemetry",
    "interpolate_sample",
    "load_telemetry",
    "parse_csv_telemetry",
    "parse_dji_srt",
    "parse_gpx",
    "parse_json_telemetry",
    "telemetry_summary",
]

log = get_logger(__name__)

_FIELD_ALIASES: dict[str, set[str]] = {
    "t": {"t", "time", "timestamp", "time_s", "elapsed", "elapsed_s", "seconds", "offset_s"},
    "lat": {"lat", "latitude", "gps_lat", "gps_latitude", "lat_deg", "latitude_deg"},
    "lon": {
        "lon",
        "lng",
        "long",
        "longitude",
        "gps_lon",
        "gps_longitude",
        "lon_deg",
        "longitude_deg",
    },
    "alt_m": {
        "alt",
        "altitude",
        "altitude_m",
        "alt_m",
        "abs_alt",
        "absolute_altitude",
        "elevation",
        "ele",
        "gps_alt",
    },
    "rel_alt_m": {
        "rel_alt",
        "rel_alt_m",
        "relative_altitude",
        "relative_altitude_m",
        "altitude_agl",
        "agl",
    },
    "heading_deg": {"heading", "heading_deg", "compass_heading", "course"},
    "pitch_deg": {"pitch", "pitch_deg"},
    "roll_deg": {"roll", "roll_deg"},
    "yaw_deg": {"yaw", "yaw_deg"},
    "speed_ms": {"speed", "speed_ms", "ground_speed", "ground_speed_ms", "velocity"},
}

_HEADER_TO_FIELD = {alias: field for field, aliases in _FIELD_ALIASES.items() for alias in aliases}

_SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->")
_SRT_LAT = re.compile(r"latitude\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_SRT_LON = re.compile(r"longitude\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_SRT_ABS_ALT = re.compile(r"(?:abs_alt|gps_alt|altitude)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_SRT_REL_ALT = re.compile(r"rel_alt\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_SRT_ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?")
_SRT_HEADING = re.compile(r"(?:heading|yaw)\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_seconds(value: Any, base: datetime | None = None) -> float | None:
    """Convert numeric or ISO-8601 timestamps to seconds."""
    numeric = _to_float(value)
    if numeric is not None:
        return numeric
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if base is None:
        return parsed.timestamp()
    return (parsed - base).total_seconds()


def _normalize_times(samples: list[TelemetrySample]) -> list[TelemetrySample]:
    """Convert absolute timestamps to seconds relative to the first sample."""
    samples = [s for s in samples if s.lat is not None or s.lon is not None or s.alt_m is not None]
    if not samples:
        return []
    samples.sort(key=lambda s: s.t)
    if samples[0].t > 1e6:
        offset = samples[0].t
        for sample in samples:
            sample.t -= offset
    return samples


def parse_csv_telemetry(path: str | Path) -> list[TelemetrySample]:
    """Parse a telemetry CSV, auto-detecting column names and delimiter."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise IngestionError(f"telemetry CSV not found: {csv_path}")

    text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if reader.fieldnames is None:
        raise IngestionError(f"telemetry CSV has no header: {csv_path}")

    mapping = {
        header: _HEADER_TO_FIELD[_normalize_header(header)]
        for header in reader.fieldnames
        if _normalize_header(header) in _HEADER_TO_FIELD
    }
    if not {"lat", "lon"} <= set(mapping.values()):
        log.warning("CSV %s is missing latitude/longitude columns", csv_path.name)

    samples: list[TelemetrySample] = []
    for row in reader:
        sample = TelemetrySample(t=0.0, source=f"csv:{csv_path.name}")
        for header, field in mapping.items():
            value: Any = row.get(header)
            parsed = _to_seconds(value) if field == "t" else _to_float(value)
            if parsed is not None:
                setattr(sample, field, parsed)
        if sample.t or sample.lat is not None:
            samples.append(sample)

    if mapping and "t" not in mapping.values():
        for position, sample in enumerate(samples):
            sample.t = float(position)
    return _normalize_times(samples)


def parse_dji_srt(path: str | Path) -> list[TelemetrySample]:
    """Parse DJI SRT subtitles that embed GPS/altitude in telemetry brackets."""
    srt_path = Path(path)
    if not srt_path.is_file():
        raise IngestionError(f"SRT file not found: {srt_path}")

    text = srt_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\r?\n\s*\r?\n", text)
    samples: list[TelemetrySample] = []
    for block in blocks:
        lat_match = _SRT_LAT.search(block)
        lon_match = _SRT_LON.search(block)
        time_match = _SRT_TIME.search(block)
        if not (lat_match and lon_match):
            continue

        iso_match = _SRT_ISO_TIME.search(block)
        if iso_match:
            t = _to_seconds(iso_match.group(0)) or 0.0
        elif time_match:
            hours, minutes, seconds, millis = (int(g) for g in time_match.groups())
            t = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        else:
            t = 0.0

        sample = TelemetrySample(
            t=t,
            lat=float(lat_match.group(1)),
            lon=float(lon_match.group(1)),
            source=f"srt:{srt_path.name}",
        )
        alt_match = _SRT_ABS_ALT.search(block)
        if alt_match:
            sample.alt_m = float(alt_match.group(1))
        rel_match = _SRT_REL_ALT.search(block)
        if rel_match:
            sample.rel_alt_m = float(rel_match.group(1))
        heading_match = _SRT_HEADING.search(block)
        if heading_match:
            sample.heading_deg = float(heading_match.group(1))
        samples.append(sample)

    if not samples:
        log.warning("no usable telemetry found in SRT %s", srt_path.name)
    return _normalize_times(samples)


def parse_gpx(path: str | Path) -> list[TelemetrySample]:
    """Parse track points from a GPX file."""
    gpx_path = Path(path)
    if not gpx_path.is_file():
        raise IngestionError(f"GPX file not found: {gpx_path}")

    try:
        root = ET.parse(gpx_path).getroot()
    except ET.ParseError as exc:
        raise IngestionError(f"invalid GPX {gpx_path}: {exc}") from exc

    samples: list[TelemetrySample] = []
    for point in root.iter():
        if not point.tag.endswith(("trkpt", "wpt", "rtept")):
            continue
        lat = _to_float(point.attrib.get("lat"))
        lon = _to_float(point.attrib.get("lon"))
        if lat is None or lon is None:
            continue
        sample = TelemetrySample(t=0.0, lat=lat, lon=lon, source=f"gpx:{gpx_path.name}")
        for child in point:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "ele":
                sample.alt_m = _to_float(child.text)
            elif tag == "time":
                sample.t = _to_seconds(child.text) or 0.0
        samples.append(sample)
    return _normalize_times(samples)


def parse_json_telemetry(path: str | Path) -> list[TelemetrySample]:
    """Parse a JSON list (or ``{"samples"|"telemetry"|"gps": [...]}``) of samples."""
    json_path = Path(path)
    if not json_path.is_file():
        raise IngestionError(f"telemetry JSON not found: {json_path}")

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestionError(f"invalid JSON {json_path}: {exc}") from exc

    if isinstance(payload, dict):
        for key in ("samples", "telemetry", "gps", "data", "records"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise IngestionError(f"unsupported telemetry JSON structure in {json_path}")

    samples: list[TelemetrySample] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        sample = TelemetrySample(t=0.0, source=f"json:{json_path.name}")
        for key, value in entry.items():
            field = _HEADER_TO_FIELD.get(_normalize_header(str(key)))
            if field is None:
                continue
            parsed = _to_seconds(value) if field == "t" else _to_float(value)
            if parsed is not None:
                setattr(sample, field, parsed)
        if sample.lat is not None or sample.lon is not None or sample.alt_m is not None:
            samples.append(sample)
    return _normalize_times(samples)


def load_telemetry(path: str | Path) -> list[TelemetrySample]:
    """Dispatch to a parser based on file extension (CSV, SRT, GPX, JSON)."""
    telemetry_path = Path(path)
    suffix = telemetry_path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        return parse_csv_telemetry(telemetry_path)
    if suffix == ".srt":
        return parse_dji_srt(telemetry_path)
    if suffix == ".gpx":
        return parse_gpx(telemetry_path)
    if suffix in {".json", ".geojson"}:
        return parse_json_telemetry(telemetry_path)
    raise IngestionError(
        f"unsupported telemetry format '{suffix}' (expected .csv, .tsv, .txt, .srt, .gpx or .json)"
    )


def interpolate_sample(samples: list[TelemetrySample], t: float) -> TelemetrySample | None:
    """Linearly interpolate a telemetry sample at time ``t`` (seconds)."""
    if not samples:
        return None
    times = [sample.t for sample in samples]
    if len(samples) == 1 or t <= times[0]:
        return samples[0]
    if t >= times[-1]:
        return samples[-1]

    upper = bisect_left(times, t)
    lower = max(0, upper - 1)
    a, b = samples[lower], samples[upper]
    span = b.t - a.t
    if span <= 0:
        return a
    ratio = (t - a.t) / span

    def blend(name: str) -> float | None:
        first, second = getattr(a, name), getattr(b, name)
        if first is None or second is None:
            return second if first is None else first
        return first + (second - first) * ratio

    return TelemetrySample(
        t=t,
        lat=blend("lat"),
        lon=blend("lon"),
        alt_m=blend("alt_m"),
        rel_alt_m=blend("rel_alt_m"),
        heading_deg=blend("heading_deg"),
        pitch_deg=blend("pitch_deg"),
        roll_deg=blend("roll_deg"),
        yaw_deg=blend("yaw_deg"),
        speed_ms=blend("speed_ms"),
        source="interpolated",
    )


def attach_telemetry(
    records: list[FrameRecord], samples: list[TelemetrySample]
) -> list[FrameRecord]:
    """Fill ``lat``/``lon``/``alt_m`` on each frame from interpolated telemetry."""
    if not samples:
        return records
    for record in records:
        sample = interpolate_sample(samples, record.timestamp_s)
        if sample is None:
            continue
        record.lat = sample.lat
        record.lon = sample.lon
        record.alt_m = sample.alt_m if sample.alt_m is not None else sample.rel_alt_m
    return records


def telemetry_summary(samples: list[TelemetrySample]) -> dict[str, Any]:
    """Compute coverage, duration and path-length statistics for navigation data."""
    if not samples:
        return {"n": 0}

    positions = [s for s in samples if s.has_position()]
    summary: dict[str, Any] = {
        "n": len(samples),
        "duration_s": round(samples[-1].t - samples[0].t, 3),
        "has_position": bool(positions),
    }
    if positions:
        lats = [s.lat for s in positions if s.lat is not None]
        lons = [s.lon for s in positions if s.lon is not None]
        summary["bbox"] = {
            "min_lat": round(min(lats), 8),
            "max_lat": round(max(lats), 8),
            "min_lon": round(min(lons), 8),
            "max_lon": round(max(lons), 8),
        }
        origin = positions[0]
        path_length = 0.0
        previous = geodetic_to_enu(
            positions[0].lat,
            positions[0].lon,
            0.0,
            lat0=origin.lat,
            lon0=origin.lon,
            alt0=0.0,
        )
        for sample in positions[1:]:
            current = geodetic_to_enu(
                sample.lat,
                sample.lon,
                0.0,
                lat0=origin.lat,
                lon0=origin.lon,
                alt0=0.0,
            )
            path_length += float(((current - previous) ** 2).sum() ** 0.5)
            previous = current
        summary["path_length_m"] = round(path_length, 2)
        summary["gps_rate_hz"] = (
            round(len(positions) / summary["duration_s"], 3) if summary["duration_s"] else None
        )

    altitudes = [s.alt_m for s in samples if s.alt_m is not None]
    if altitudes:
        summary["altitude_m"] = {"min": round(min(altitudes), 2), "max": round(max(altitudes), 2)}
    speeds = [s.speed_ms for s in samples if s.speed_ms is not None]
    if speeds:
        summary["speed_ms"] = {
            "mean": round(sum(speeds) / len(speeds), 3),
            "max": round(max(speeds), 3),
        }
    return summary
