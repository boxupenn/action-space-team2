# Team 2 defense policy — what to copy into your checkout

Our defense runs as a `swarm_ops` policy named `defend` under your umbrella, next to your
`formation` attackers. It reads the attackers from the telemetry bus (`formation/drone<N>`),
localises our own drones from their cameras (any mapped tag of `config/map.yaml`) and sends
`rc` commands. Nothing else of your infra changes.

## Files (paths relative to `drone_deployment_infra/`)

| Action | Path | What |
|---|---|---|
| **ADD** | `swarm_ops/policies/defend.py` | the policy (obs builder, nets, bus attacker source, safety) |
| **ADD** | `config/defend.yaml` | its settings: the base position, fence from your map, launch spots, `attackers: bus` |
| **REPLACE** | `swarm_ops/policies/__init__.py` | yours + two lines: `from .defend import DefendPolicy` and `"defend": DefendPolicy` in `POLICIES` |
| **ADD** | `../simulation_infra/policies/defend_3v2_arena_s2/` | `defender_0.pt`, `defender_1.pt`, `defender_2.pt` (TorchScript) and the same nets as `defender_<i>.npz` (numpy, used when `torch` is not installed), `README.md` with the sim numbers |
| optional | `swarm_ops/policies/base.py`, `policies/umbrella.py`, `swarm_ops/swarm_policies.py` | our copies add a `preflight(d)` hook (waits for a tag fix before FLY) and a `--defend-dir` flag. **Not required**: `defend.py` starts its video streams in `setup()` and works with your unmodified `swarm_policies.py`. |
| optional | `swarm_ops/test_defend.py`, `swarm_ops/defend_offline.py` | tests (`python -m pytest test_defend.py -q`) and a no-drone kinematic demo |

Do not replace `map.yaml`, `tags.yaml`, `dongles.json` or `camera/*.yaml`: we use yours.

## Requirements on the laptop

- Your `swarm_ops/requirements.txt` (numpy, opencv, av, pyyaml, pupil-apriltags).
- `cal_and_map` installed (`pip install -e drone_deployment_infra/cal_and_map`): `core/positional.py`
  localises our drones with it. It is the package `map_tags.py` / `05_map.py` use.
- `torch` is optional: with it the `.pt` files are used, without it the `.npz` twins.

## Run

```
cd drone_deployment_infra/swarm_ops
python connect.py 4 5 6 7 9
python swarm_policies.py --run defend:4,5,6 --run formation:7,9            # DRY-RUN first (no takeoff)
python swarm_policies.py --run defend:4,5,6 --run formation:7,9 --fly --duration 60
```

- The first drone of `--run defend:a,b,c` is `defender_0`, then `defender_1`, `defender_2`.
- Dry-run: every tick each defender prints `p=` (its own tag fix, sim frame), `att=` (the attackers
  it sees on the bus) and the `rc` it would send. **All three must show a `p=` before `--fly`**;
  a defender with no fix hovers (`rc 0 0 0 0`) and lands after 6 s.
- Place the defenders at the launch spots in `config/defend.yaml` (tag-10 frame x, z; altitude
  ~1.9-2.2 m), **facing tag 10**: that is where the goalkeeper waits. With no attacker in the air a defender
  holds position; reset both teams between rounds as usual.
- If your attackers are not `drone7`/`drone9`, set `attacker_names` in `config/defend.yaml`, or leave
  it empty: the first two `formation/` topics on the bus are taken.
- Our defenders publish `defend/drone<N>` (same fields as `formation/`) on the bus every tick.
- After your umbrella's `takeoff` (~1 m) each defender first climbs to its launch height
  (`launch_height_cm` in `defend.yaml`: 190 / 205 / 220 cm, above the attackers) with its camera on the
  tags, then the net flies it; it commands full stick (`speed_cap: 1.0`, ~2 m/s) on the rams.

## Behaviour

3 defenders vs 2 attackers. Each defender rams the attacker it is assigned to (assignment computed
on every drone from the same numbers, no communication); the third waits at its launch spot and is
dispatched when an attacker gets within 3 m of the base with nobody about to ram it. A defender
that rammed is out; attackers that stopped publishing / landed count as down. Defenders keep 0.6 m
from each other (the higher-index one holds), stay inside the fence box of `defend.yaml` and
between 50 and 225 cm height. Full description: `swarm_ops/README.md`, section 5.1.
