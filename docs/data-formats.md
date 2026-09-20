# Input and output data formats

## Video

Any container OpenCV/FFmpeg can decode: `.mp4`, `.mov`, `.mkv`, `.avi`.
1080p and 4K are expected; nothing is assumed about frame rate — telemetry is
resampled onto frame timestamps.

```yaml
ingest:
  video: data/pass_01.mp4
```

## Telemetry

`ingest.telemetry` accepts `.csv`, `.tsv`, `.txt`, `.srt` (DJI), `.gpx` and
`.json`. Column names are auto-detected (case-insensitive, units stripped).

| Canonical field | Accepted column names |
| --- | --- |
| `t` | `t`, `time`, `timestamp`, `time_s`, `elapsed_s`, `offset_s` |
| `lat` | `lat`, `latitude`, `gps_lat`, `lat_deg` |
| `lon` | `lon`, `lng`, `long`, `longitude`, `lon_deg` |
| `alt_m` | `alt`, `altitude`, `abs_alt`, `elevation`, `ele` |
| `rel_alt_m` | `rel_alt`, `relative_altitude`, `altitude_agl`, `agl` |
| `heading_deg` | `heading`, `compass_heading`, `course` |
| `pitch_deg`, `roll_deg`, `yaw_deg` | `pitch`, `roll`, `yaw` (+ `_deg`) |
| `speed_ms` | `speed`, `ground_speed`, `velocity` |

Example CSV:

```csv
timestamp,latitude,longitude,altitude,rel_alt,heading,pitch,roll,speed
0.000,12.971598,77.594562,912.4,100.0,42.1,-3.0,0.4,7.8
0.033,12.971612,77.594591,912.5,100.1,42.3,-3.1,0.3,7.9
```

DJI SRT blocks are parsed directly:

```
1
00:00:00,000 --> 00:00:00,033
2024-01-01 09:15:00.000
[latitude: 12.971598] [longitude: 77.594562]
[rel_alt: 100.0 abs_alt: 912.4] [heading: 42.1]
```

JSON accepts a list of objects or an object with a `samples` / `telemetry` /
`gps` / `data` / `records` list. GPS timestamps may be numeric seconds or
ISO-8601; absolute times are normalised to seconds relative to the first fix.

## Outputs

| Artifact | Path | Consumer |
| --- | --- | --- |
| Keyframes | `frames_selected/*.jpg` | SfM, texturing, contact sheet |
| Dynamic masks | `frames_selected/masks/*.png` | COLMAP mask input (255 = ignore) |
| Frame manifest | `ingest/frames.csv`, `preprocess/selected_frames.csv` | All stages |
| Sparse model | `sfm/sparse/` (COLMAP), `sfm/sparse.ply` | dense, georef |
| Dense cloud | `dense/fused.ply` | mesh, georef, metrics |
| Mesh | `mesh/mesh-*.ply`, `mesh/textured/mesh.obj` (+ textures) | viewer/CAD/GIS |
| Georeferenced cloud | `georef/georeferenced_*.ply` (local ENU, metres) | measurement |
| Camera track | `georef/camera_track.geojson` (WGS84) | GIS |
| Metrics | `metrics/metrics.json` | evaluation |
| Report | `report.html`, `manifest.json` | judges/operators |

Coordinate conventions:

- Local ENU metres with origin = median GPS fix (override with `geo.origin_*`).
- WGS84 / EPSG:4326 for all geographic output; other EPSG codes via `geo.epsg`
  with the optional `geo` extra.
- PLY files carry `x, y, z` (float32) and optional `red, green, blue` (uint8).
