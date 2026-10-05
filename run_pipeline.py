#!/usr/bin/env python3
"""Collect CFBD data and train the college football prediction model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collect import collect  # noqa: E402
from src.config import END_YEAR, START_YEAR  # noqa: E402
from src.train import train  # noqa: E402
from src.weather import fetch_weather  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if not args.skip_collect:
        collect(args.start_year, args.end_year)
    if not args.skip_weather:
        fetch_weather()
    if not args.skip_train:
        train()


if __name__ == "__main__":
    main()
