# Status (ClawBot)

_Last updated: 2026-02-10_

## Current focus
- Establish repo structure + documentation for Physical AI (driving + robotics)
- Add simulation/data/eval scaffolding that can support CARLA + (Isaac or MuJoCo)

## Next (top 3)
1) Add `docs/README.md` table of contents + better navigation
2) Add scenario + metrics schemas (JSON/YAML) and a simple validator script
3) Add a minimal "eval runner" skeleton that outputs a metrics JSON

## Blockers / questions for owner
- Confirm sim stack priority for the first runnable demo:
  - Driving: CARLA + ScenarioRunner? (yes/no)
  - Robotics: Isaac vs MuJoCo (pick one to implement first)
