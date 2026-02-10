# Data schema (draft)

This repo aims to support multiple training paradigms (pretrain → SFT/IL → RL). To keep things consistent, we define a minimal **episode** schema.

## Episode
An episode is a sequence of steps with optional task context.

### Required
- `steps`: list of step objects

### Optional
- `episode_id`: string
- `task`: natural language instruction (string)
- `scenario`: structured scenario tag (string, e.g. `LEFT_TURN`)
- `metadata`: free-form dict

## Step
A step is one time index `t`.

### Required
- `t`: integer
- `dt`: float (seconds)
- `obs`: observation object (see below)
- `action`: action object (see below)

### Optional (used by RL / evaluation)
- `reward`: float
- `cost`: float (safety constraint cost)
- `done`: bool
- `info`: dict

## Observation (obs)
This is intentionally flexible. Examples:
- kinematics: `{x,y,yaw,v}`
- camera: `{rgb: <path|bytes>, intrinsics: ...}`
- BEV: `{bev: <array>}`
- text: `{task: "..."}`

## Action
For the toy driving policy we currently use:
- `a`: acceleration (m/s^2)
- `kappa`: curvature (1/m)

So an action can be:
```json
{ "a": 0.3, "kappa": 0.05 }
```

## Notes
- We keep schemas descriptive rather than strict for now; later we can add JSONSchema and validation.
