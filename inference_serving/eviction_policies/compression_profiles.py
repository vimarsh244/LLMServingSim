import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass
class CompressionProfile:
    name: str
    points: Dict[float, float]

    def penalty(self, ratio: float) -> float:
        r = max(1e-6, min(float(ratio), 1.0))
        if not self.points:
            return max(0.0, 1.0 - r)

        xs = sorted(self.points.keys())
        if r <= xs[0]:
            return self.points[xs[0]]
        if r >= xs[-1]:
            return self.points[xs[-1]]

        for left, right in zip(xs[:-1], xs[1:]):
            if left <= r <= right:
                lp = self.points[left]
                rp = self.points[right]
                if right == left:
                    return lp
                alpha = (r - left) / (right - left)
                return lp + alpha * (rp - lp)

        return max(0.0, 1.0 - r)


_PRESET_CURVES: Dict[str, Dict[float, float]] = {
    # Baseline no-compression behavior.
    "none": {
        1.0: 0.0,
        0.75: 0.0,
        0.5: 0.0,
        0.25: 0.0,
    },
    # Paper-inspired presets for KV compression sensitivity shaping.
    "balanced": {
        1.0: 0.0,
        0.9: 0.03,
        0.75: 0.08,
        0.5: 0.18,
        0.25: 0.35,
    },
    "kivi": {
        1.0: 0.0,
        0.9: 0.02,
        0.75: 0.06,
        0.5: 0.12,
        0.25: 0.24,
    },
    "kvquant": {
        1.0: 0.0,
        0.9: 0.02,
        0.75: 0.05,
        0.5: 0.10,
        0.25: 0.22,
    },
    "gearkv": {
        1.0: 0.0,
        0.9: 0.03,
        0.75: 0.07,
        0.5: 0.15,
        0.25: 0.30,
    },
    "h2o": {
        1.0: 0.0,
        0.9: 0.04,
        0.75: 0.09,
        0.5: 0.20,
        0.25: 0.40,
    },
}


def _sanitize_points(points: Iterable[Tuple[float, float]]) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for ratio, penalty in points:
        r = max(1e-6, min(float(ratio), 1.0))
        p = max(0.0, float(penalty))
        out[r] = p
    return out


def _load_trace_json(trace_path: Path) -> Dict[float, float]:
    with open(trace_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "ratio_sensitivity" in payload and isinstance(payload["ratio_sensitivity"], dict):
            return _sanitize_points((float(k), float(v)) for k, v in payload["ratio_sensitivity"].items())
        if "points" in payload and isinstance(payload["points"], list):
            pts = []
            for item in payload["points"]:
                if not isinstance(item, dict):
                    continue
                if "ratio" not in item or "sensitivity" not in item:
                    continue
                pts.append((float(item["ratio"]), float(item["sensitivity"])))
            return _sanitize_points(pts)

    if isinstance(payload, list):
        pts = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "ratio" not in item or "sensitivity" not in item:
                continue
            pts.append((float(item["ratio"]), float(item["sensitivity"])))
        return _sanitize_points(pts)

    raise ValueError(f"Unsupported JSON compression trace format in {trace_path}")


def _load_trace_csv(trace_path: Path) -> Dict[float, float]:
    points = []
    with open(trace_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "ratio" not in row or "sensitivity" not in row:
                continue
            points.append((float(row["ratio"]), float(row["sensitivity"])))
    return _sanitize_points(points)


def load_compression_profile(profile_name: str, trace_path: str = "") -> CompressionProfile:
    name = (profile_name or "balanced").strip().lower()

    if trace_path:
        path = Path(trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Compression trace file not found: {trace_path}")
        if path.suffix.lower() == ".json":
            points = _load_trace_json(path)
        elif path.suffix.lower() == ".csv":
            points = _load_trace_csv(path)
        else:
            raise ValueError("Compression trace must be .json or .csv")

        if not points:
            raise ValueError(f"Compression trace {trace_path} did not contain valid ratio/sensitivity points")
        return CompressionProfile(name=f"trace:{path.name}", points=points)

    if name not in _PRESET_CURVES:
        known = ", ".join(sorted(_PRESET_CURVES.keys()))
        raise ValueError(f"Unknown compression profile '{profile_name}'. Choose one of: {known}")

    return CompressionProfile(name=name, points=dict(_PRESET_CURVES[name]))
