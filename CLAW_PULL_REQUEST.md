# Pull Request: Increase toy waypoint env max_steps

## Title
Increase toy waypoint env max_steps from 100 to 500 for meaningful RL evaluation

## Body
## Summary
Increase the maximum episode length in the toy waypoint environment from 100 to 500 steps, allowing episodes to actually reach the goal and provide meaningful training signal for RL refinement.

## Changes
- **waypoint_env.py**: Add `max_steps` as configurable parameter with default 500
- **eval_toy_waypoint_rl.py**: Update hardcoded max_steps from 100 to 500

## Results
With max_steps=500:
```
SFT:  ADE=3.010m, FDE=3.838m, Success=10.0%
RL:   ADE=3.099m, FDE=3.562m, Success=10.0%
Delta: ADE -3.0%, FDE +7.2%, Success +0.0%
```

## Theme
RL environment tuning - enabling meaningful success signals

## Pipeline Context
Waymo episodes → pretrain → waypoint BC → RL refinement → ScenarioRunner eval

## Commands
```bash
gh pr create --title "Increase toy waypoint env max_steps from 100 to 500" --body-file CLAW_PULL_REQUEST.md
```
