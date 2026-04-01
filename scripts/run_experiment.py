#!/usr/bin/env python3
"""Run an experiment from a YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hdc_nids.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HDC NIDS experiment")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    args = parser.parse_args()
    summary = run_experiment(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
