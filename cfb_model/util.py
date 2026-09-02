"""Shared helpers: nested object access, parsing, geography, shrinkage."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd

_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def camel_to_snake(name: str) -> str:
    return _CAMEL_RE.sub("_", name).replace("__", "_").lower()


def snake_to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def getv(obj: Any, key: str, default: Any = None) -> Any:
    """Read snake_case or camelCase from a dict or SDK object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        if key in obj and obj[key] is not None:
            return obj[key]
        camel = snake_to_camel(key)
        if camel in obj and obj[camel] is not None:
            return obj[camel]
        snake = camel_to_snake(key)
        if snake in obj and obj[snake] is not None:
            return obj[snake]
        return default
    if hasattr(obj, key):
        value = getattr(obj, key)
        if value is not None:
            return value
    camel = snake_to_camel(key)
    if hasattr(obj, camel):
        value = getattr(obj, camel)
        if value is not None:
            return value
    return default


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def as_int(value: Any, default: int | None = None) -> int | None:
    number = as_float(value)
    if number is None:
        return default
    return int(number)


def as_bool_int(value: Any) -> int:
    if value is True or value == 1 or value == "1":
        return 1
    if value is False or value == 0 or value == "0":
        return 0
    if isinstance(value, str) and value.lower() in {"true", "yes"}:
        return 1
    return 0


def enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def parse_pair(value: Any) -> tuple[float | None, float | None]:
    """Parse '3-7' or '18-30' into (left, right)."""
    if value is None:
        return None, None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return None, None
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[-/]\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_efficiency(value: Any) -> tuple[float | None, float | None, float | None]:
    """Return (conversions, attempts, rate) from '3-7'."""
    made, att = parse_pair(value)
    if made is None or att is None or att <= 0:
        return made, att, None
    return made, att, made / att


def parse_line_scores(value: Any) -> tuple[int | None, int | None, int | None, int | None]:
    """Parse CFBD homeLineScores / awayLineScores into (q1, q2, q3, q4)."""
    if value is None or value == "":
        return None, None, None, None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            nums = re.findall(r"-?\d+", text)
            value = [int(n) for n in nums]
    if not isinstance(value, (list, tuple)):
        return None, None, None, None
    out: list[int | None] = []
    for item in list(value)[:4]:
        out.append(as_int(item))
    while len(out) < 4:
        out.append(None)
    return out[0], out[1], out[2], out[3]


def parse_possession(value: Any) -> float | None:
    """Convert 'MM:SS' or seconds to seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if re.match(r"^\d+(\.\d+)?$", text):
        return float(text)
    match = re.match(r"^(\d+):(\d{1,2})$", text)
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def shrink(current: float | None, prior: float | None, n: float, k: float = 4.0) -> float | None:
    """James-Stein / empirical-Bayes blend toward a prior.

    weight = n / (n + k). n=0 (no games yet) is 100% prior — week 1 uses FPI,
    talent, coach scheme, and last year, not a fake in-season sample.
    """
    if isinstance(current, float) and math.isnan(current):
        current = None
    if isinstance(prior, float) and math.isnan(prior):
        prior = None
    if current is None and prior is None:
        return None
    if current is None:
        return prior
    if prior is None:
        return current
    if n <= 0:
        return prior
    weight = n / (n + k)
    return weight * current + (1.0 - weight) * prior


def zscore(s: pd.Series) -> pd.Series:
    """Season-safe z-score; constant columns become 0."""
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if std is None or pd.isna(std) or float(std) == 0.0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def shrink_series(current: pd.Series, prior: pd.Series, n: pd.Series, k: float) -> pd.Series:
    cur = pd.to_numeric(current, errors="coerce")
    base = pd.to_numeric(prior, errors="coerce")
    sample = pd.to_numeric(n, errors="coerce").fillna(0).clip(lower=0)
    weight = sample / (sample + k)
    blended = weight * cur + (1.0 - weight) * base
    blended = blended.where(cur.notna(), base)
    return blended.where(base.notna(), cur)


def haversine_miles(lat1: float | None, lon1: float | None, lat2: float | None, lon2: float | None) -> float | None:
    if None in (lat1, lon1, lat2, lon2):
        return None
    r_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r_miles * math.asin(min(1.0, math.sqrt(a)))


def iso_date(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if hasattr(value, "isoformat"):
        text = value.isoformat()
    return text


def to_plain(obj: Any) -> Any:
    """Recursively convert SDK models to snake_case dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [to_plain(item) for item in obj]
    if isinstance(obj, dict):
        return {camel_to_snake(str(k)): to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return to_plain(obj.to_dict())
    if hasattr(obj, "value"):
        return obj.value
    return obj
