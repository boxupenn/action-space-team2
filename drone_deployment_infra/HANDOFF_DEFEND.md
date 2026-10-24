# Team 2 defense policy — hand-off for the organisers

Our defense is a `swarm_ops` policy named **`defend`**. It runs **in your umbrella** (`swarm_policies.py`), on
the 3 drones you give it, next to your `formation` attackers. Nothing else of your infra changes: your map, tags,
dongles, camera calibrations, `connect.py`, `swarm_formation.py` stay as they are.

What it does, in one paragraph: every tick each of our drones localises itself from its own camera (any mapped tag
of `config/map.yaml`, the same `core/positional.py` localiser your tools use), reads your attackers from the
telemetry bus (`formation/drone<N>`), builds the observation the simulation was trained on, runs its network and
sends `rc a b c d`. 3 defenders vs 2 attackers: two are assigned one attacker each (every drone computes the same
assignment from the same numbers, no communication), the third waits next to the base and is dispatched when an
attacker gets close with nobody about to ram it. A defender that rammed is out. Attackers that land, drop to the
floor or stop publishing count as down.

---

## 1. Files to copy (paths relative to `drone_deployment_infra/`)

| Action | Path | What |
|---|---|---|
| **ADD** | `swarm_ops/policies/defend.py` | the policy: observation builder, networks, bus attacker source, climb phase, safety |
| **ADD** | `config/defend.yaml` | its settings: base position, fence, launch spots/heights, attacker names, thresholds |
| **REPLACE** | `swarm_ops/policies/__init__.py` | yours + two lines: `from .defend import DefendPolicy` and `"defend": DefendPolicy` in `POLICIES` |
| **ADD** | `../simulation_infra/policies/defend_3v2_base_s2/` | `defender_0.pt`, `defender_1.pt`, `defender_2.pt` (TorchScript) and the same networks as `defender_<i>.npz` (numpy; used automatically when `torch` is not installed), `README.md` with the simulation numbers |
| optional | `swarm_ops/test_defend.py`, `swarm_ops/defend_offline.py` | 24 unit checks (no drones) and a no-drone kinematic demo |
| optional | `swarm_ops/policies/base.py`, `swarm_ops/policies/umbrella.py`, `swarm_ops/swarm_policies.py` | our copies add a `preflight(d)` hook (waits for a tag fix before FLY) and a `--defend-dir` flag. **Not required**: `defend.py` starts its video streams in `setup()` and works with your unmodified `swarm_policies.py`. |

Do **not** replace `config/map.yaml`, `config/tags.yaml`, `config/dongles.json` or `config/camera/*.yaml`: we use yours.

`policy_dir` in `config/defend.yaml` points at the network folder (relative to the yaml); if you put the folder
somewhere else, change that line.

## 2. Requirements on the laptop

- Your `swarm_ops/requirements.txt` (numpy, opencv-python, av, pyyaml, pupil-apriltags).
- **`cal_and_map` installed**: `pip install -e drone_deployment_infra/cal_and_map`. `core/positional.py` localises
  our drones with it (the package `map_tags.py` / `05_map.py` use, so it is probably there already).
  Check: `python -c "import tello_calibration"`.
- `torch` is **optional**: with it the `.pt` files are used, without it the `.npz` twins (same numbers).
- Camera calibration `config/camera/tello_<n>.yaml` for the drone numbers you give us; a missing one falls back to
  `config/camera/default.yaml`.

Quick check without any drone (from `drone_deployment_infra/swarm_ops`):
```
python -m pytest test_defend.py -q        # expect: 24 passed
python defend_offline.py --seconds 12     # kinematic demo: 3 fake defenders vs 2 scripted attackers, prints obs/rc
```

## 3. Running it

```
cd drone_deployment_infra/swarm_ops
python connect.py 4 5 6 7 9
python swarm_policies.py --run defend:4,5,6 --run formation:7,9                            # DRY-RUN (no takeoff)
python swarm_policies.py --run defend:4,5,6 --run formation:7,9 --fly --duration 60 2>&1 | tee defend_run.log
```

- The first drone of `--run defend:a,b,c` is `defender_0`, the second `defender_1`, the third `defender_2`.
- **Attacker numbers**: by default we take the first two `formation/` topics that appear on the bus. If other
  `formation` drones fly at the same time, set `attacker_names: [drone7, drone9]` in `config/defend.yaml`.
- `--duration` lands everybody when it expires (your umbrella's rule); between rounds reset both teams by hand.

### Placement before FLY (the figure)

Launch spots, in the tag-10 frame (x right along the wall seen from the room, z out of the wall), and what each
drone does from there:

| drone | tag x | z out of the wall | role when idle |
|---|---|---|---|
| `defender_0` | 1.00 m | 1.37 m | interceptor, mid-room |
| `defender_1` | 1.52 m | 0.50 m | interceptor, behind the line |
| `defender_2` | 0.33 m | 0.99 m | goalkeeper, next to the base |

All three **facing tag 10** (they must have a tag in view: no fix = they do not fly). After your umbrella's
`takeoff` (~1 m) each climbs to `launch_height_cm` (145 / 150 / 155 cm) with its camera on the tags, then the
network takes over. Base: the 13 cm box 0.86 m out of the tag-10 wall, 0.94 m from the left long wall; the line
1.30 m from the wall is the one that counts.

### Dry-run: what to look for

Without `--fly` nothing takes off; every tick each defender prints what it would do:
```
[drone4] defend0 p=[1.37, 1.0, 0.25] yaw=178 att=[[4.1, 0.8, 1.3], None]
    obs=[...34 numbers...]
    action=[0.83, -0.21, 0.12, 0.0] -> rc +41 -12 +6 +2
```
- `p=` = its own tag fix in the sim frame [out of the wall, right, up relative to the base]. **All three must show a
  `p=`** (not `NO FIX`) before you fly. `NO FIX` = the camera does not see a mapped tag: turn the drone to tag 10.
- `att=` = the attackers as we read them from the bus, `None` until a `formation/drone<N>` topic with a position
  exists. With no attacker in the air the defenders just hold position (by design).
- `rc` = what would be sent (a right, b forward, c up, d yaw; our yaw channel keeps the camera on the tags).

### Flying: the log

With `--fly` each defender prints one line per second (`tee` it to a file for us):
```
[drone4] defend0 CLIMB p=[...] att=- h=108 rc +0 +0 +30 +2                 climbing to its launch height
[drone4] launch height 145 cm: reads 147 cm after 2.3 s -> policy           handed over to the network
[drone4] defend0 policy p=[...] att=[[4.1, 0.8, 1.3], None] h=150 rc +41 -12 -5 +0   chasing
[drone4] defend0 NO ATTACKER ON THE BUS -> hold p=[...] att=[None, None]   nobody to ram: holding
[drone4] defend0 NO FIX (age 1.2s, frames 212) ... rc +0 +0 +0 +0          lost the tags: hovering
[drone4] SEP GUARD: defender1 at 0.52 m -> holding until > 0.8 m            too close to a teammate
[drone4] no tag fix for 6 s                                                  -> that drone lands itself
```

## 4. Behaviour and limits

- Targets: an attacker counts while its `formation/drone<N>` topic is younger than 1 s, has a position, reads
  above `attacker_min_height_cm` (50 cm) and its state is not LAND / DONE / EMERGENCY. A rammed or landed attacker
  keeps its last position with "alive = 0".
- Assignment: each drone rebuilds the 3 x 2 intercept-time table from the same numbers (own and teammates' tag
  fixes, attacker positions and velocities) and takes the pairing with the smallest total time; the leftover drone
  is the goalkeeper. It is dispatched when the nearest attacker is within `alert_radius` (2.2 m) of the base and no
  active teammate is within `cover_m` (1.2 m) of it, or anyway within `alert_force` (1.7 m).
- A defender that rammed is "spent": its teammates stop avoiding it; your umbrella's status/battery guards still
  apply to it.
- Speed: the networks command world-frame velocities as a fraction of full stick (`speed_cap: 1.0`, ~2 m/s). Lower
  `speed_cap` only if you must: the geometry was trained at 1.0.
- Separation guard: two defenders closer than `sep_guard_m` (0.6 m) -> the higher-index one holds horizontally
  until they are `sep_release_m` (0.8 m) apart.
- Fence (`fence:` in the yaml, tag frame) and `height_cm` (50-225 cm) zero any stick that pushes further out.
- No tag fix for 0.5 s -> hover; for 6 s -> that drone lands. No attacker in the air -> hold position.
- Our defenders publish `defend/drone<N>` on the bus every tick (same fields as `formation/drone<N>`: `p` in the
  map frame, `psi_deg`, `fix_age`, `height_cm`, `rc`, `state` DEFEND / SPENT / NOFIX), in case you want to log or
  display them.

## 5. If something is off

| Symptom | Cause | Fix |
|---|---|---|
| defenders never take off / `FAIL: policy preflight` | no tag fix at the launch spot | point the camera at tag 10; check `config/camera/tello_<n>.yaml` |
| `ImportError: tello_calibration` | `cal_and_map` not installed | `pip install -e drone_deployment_infra/cal_and_map` |
| `FileNotFoundError: ... defender_0.pt (needs torch) or defender_0.npz` | wrong `policy_dir` | fix the path in `config/defend.yaml` |
| they hover and print `NO ATTACKER ON THE BUS` | the attackers are not under the same umbrella, have no fix yet, or are below 50 cm | run `--run formation:...` in the same command; or set `attacker_names` |
| they hover and print `NO FIX` | lost the tags while turning / too far from the wall | they land after 6 s; re-place them facing tag 10 |
| `CLIMB` for 6 s then `-> policy` with a strange height | tof reading missing | harmless: the network climbs by itself |
| a defender turns but does not translate | fence or height limit active (see `fence:` / `height_cm`), or sep guard | check the log line's `rc` and the yaml |

Simulation numbers for this version (`simulation_infra/policies/defend_3v2_base_s2/README.md`): over 1000 rounds
in the figure's arena, both teams taking off from the floor to 1.5 m, 15 x 15 x 3 cm hit box: teacher rule 100 % / 0 %
(300 rounds), the cloned network 99.7 % interception / 0.1 % breach (1000 rounds).

Contact: Team 2 (Lucas / xieqiyi@seas.upenn.edu).
