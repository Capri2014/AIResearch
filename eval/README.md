# Eval (skeleton)

Evaluation should be consistent across SFT and RL.

Design goal:
- Every run emits a machine-readable artifact: `metrics.json`

Near-term:
- reuse `demos/alpamayo_r1_toy/run_demo.py --metrics-out metrics.json`

Future:
- `eval/run_eval.py --policy <path> --dataset <path> --out metrics.json`
- aggregate metrics across episodes and report summary
