"""Teacher→student distillation stub.

This documents the intended interface:
- teacher_policy(obs, task) -> action sequence (a[t], kappa[t]) or trajectory
- student_policy(obs, task) -> same
- optimize student to match teacher outputs (supervised loss)

Implement once we pick:
- teacher model access method
- dataset format
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Distillation not implemented yet. See docstring for intended interface."
    )


if __name__ == "__main__":
    main()
