"""Waymo TFRecord smoke test demo.

This demo is intended to be run in an environment with:
- tensorflow
- waymo-open-dataset
- local Waymo TFRecord files

It converts TFRecords into `episode.json` shards and validates them.

Usage:
  python3 -m demos.waymo_tfrecord_smoke.run --tfrecord /path/to/file.tfrecord \
      --out-dir out/episodes/waymo_smoke --max-frames 50 --no-write-images

Notes:
- `--no-write-images` avoids writing JPEGs; it still keeps camera keys present.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys

from data.waymo.validate_episode import validate_episode_dict


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("out/episodes/waymo_smoke"))
    p.add_argument("--max-frames", type=int, default=50)
    p.add_argument("--no-write-images", action="store_true")
    args = p.parse_args()

    python = sys.executable

    cmd = [
        python,
        "-m",
        "data.waymo.convert",
        "--out-dir",
        str(args.out_dir),
        "--split",
        "smoke",
        "--max-frames",
        str(args.max_frames),
    ]
    if args.no_write_images:
        cmd.append("--no-write-images")

    for tfrec in args.tfrecord:
        cmd += ["--tfrecord", str(tfrec)]

    run(cmd)

    # Validate all written episodes.
    eps = sorted(args.out_dir.glob("*.json"))
    if not eps:
        raise SystemExit(f"No episode JSONs found in {args.out_dir}")

    for ep_path in eps:
        ep = json.loads(ep_path.read_text())
        validate_episode_dict(ep)

    print(f"\nValidated {len(eps)} episode(s) in: {args.out_dir}")


if __name__ == "__main__":
    main()
