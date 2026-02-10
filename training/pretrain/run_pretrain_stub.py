"""Pre-train stub.

This file exists to document wiring. It is not meant to run meaningful pretraining.

Expected future responsibilities:
- load a large-scale dataset (vision/video/text)
- train a backbone encoder
- export a checkpoint that downstream SFT/RL can load

See: training/pretrain/README.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PretrainConfig:
    out_dir: Path = Path("out/pretrain")
    steps: int = 1000


def main() -> None:
    cfg = PretrainConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder artifact so downstream scripts have a path to point at.
    # In real use this would be a weights file.
    (cfg.out_dir / "README.txt").write_text(
        "This is a placeholder for a pretrained checkpoint.\n"
        "Implement real pretraining when dataset + backbone are chosen.\n"
    )

    print(f"[pretrain] wrote stub artifact to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
