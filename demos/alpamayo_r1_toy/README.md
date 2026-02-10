# Alpamayo‑R1 — toy demo (unicycle integration)

This is a **CPU‑only** toy demo to understand a key modeling choice used in Alpamayo‑style driving policies:

> Predict **actions** (acceleration + curvature) → integrate a simple vehicle model → get a **physically plausible trajectory**.

It is **not** the real Alpamayo model.

## Run

```bash
cd demos/alpamayo_r1_toy
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Interactive plot
python run_demo.py

# Headless + write metrics artifact
python run_demo.py --no-plot --metrics-out metrics.json
```

## What you’ll see
- A plotted trajectory (x/y)
- Velocity over time
- Curvature + acceleration signals

## Extend it
- Replace the synthetic action generator with a learned model.
- Add simple “scenario” inputs (turn left vs right) and measure metrics.
