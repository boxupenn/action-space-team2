# swarm_ops: Tello swarm commands

All commands run from `drone_deployment_infra/swarm_ops/`. Config lives in `../config/`
(`dongles.json`, `map.yaml`, `tags.yaml`, `camera/tello_<n>.yaml`).

```
cd ~/Action_Space/drone_deployment_infra/swarm_ops
source <your venv>/bin/activate
```

---

## 1. One-time setup (per laptop)

```
pip install -r requirements.txt
echo "net.ipv4.conf.all.rp_filter=2" | sudo tee /etc/sysctl.d/99-tello.conf
sudo sysctl --system
```
- Plug the USB Wi-Fi adapters into a **powered hub** (adapter names in `dongles.json`
  belong to those exact dongles).
- Create every drone's Wi-Fi profile from `dongles.json` (drones can be OFF):
```
python connect.py --create --all
```
  Each profile is locked to its adapter + its drone's radio (BSSID), auto-connects,
  uses static IP `192.168.10.10<n>`, 2.4 GHz only.

**New drone?** Add its entry in `../config/dongles.json` (`iface`, `drone`, `ssid`, `bssid`),
then `python connect.py --create <n>`.
Find a drone's BSSID: `nmcli -f SSID,BSSID dev wifi list | grep TELLO`.

---

## 2. Connect drones (every session)

Turn the drones on, wait ~20-30 s (they join by themselves), then check:
```
python connect.py 4 5 6 7 9          # verify; connects any that aren't up yet
python connect.py --all              # every drone in dongles.json
python connect.py 4 5 6 7 9 --wait 60   # run FIRST, then power on: each joins as it appears
python connect.py 4 5 6 7 9 --down   # disconnect
```
Every drone must say `OK`. `NOT DETECTED` = adapter unplugged / hub unpowered.

Tip: chain it so a flight only starts when all drones are ready:
```
python connect.py 4 5 6 7 9 && python swarm_formation.py --only 4,5,6,7,9 --fly --fast --headless
```

---

## 3. Swarm formation (`swarm_formation.py`)

Drones take off, find tag 10 (or any mapped tag), fly to slots on an arc in front of
tag 10, hold, land. **Without `--fly` it is a dry-run** (no takeoff, no motion).

| Mode | Command |
|---|---|
| Dry-run (props off, check tags/video) | `python swarm_formation.py --only 4,5,6` |
| Normal (slow, careful) | `python swarm_formation.py --only 4,5,6 --fly` |
| Fast | `python swarm_formation.py --only 4,5,6 --fly --fast` |
| Headless (no windows, terminal only) | `python swarm_formation.py --only 4,5,6 --fly --fast --headless` |
| Flip wave (back flip one by one) | `python swarm_formation.py --only 4,5,6 --fly --fast --flip` |
| Recommended | `python swarm_formation.py --only 4,5,6 --fly --fast --headless --rest-radius 0.15 --hold 1` |

**Options**
| Flag | Meaning |
|---|---|
| `--only 4,5,6` | which drones |
| `--fly` | actually fly (without it: dry-run) |
| `--fast` | faster search/approach/turns (same final precision) |
| `--headless` | no windows; keys in terminal: `q`+Enter land all, `e`+Enter EMERGENCY |
| `--no-video` | no camera windows, only the top-view map |
| `--display-hz 5` | refresh windows less often (saves CPU) |
| `--rest-radius 0.15` | slot "close enough" zone in m (default 0.08); inside it the drone just holds |
| `--hold 1` | seconds to hold the formation before landing (default 5) |
| `--flip` | back flip wave once in formation (needs battery >= 55 %, ~1 m free behind each slot) |
| `--radius 1.3` | slot distance from tag 10 in m (don't go below ~1.2) |
| `--slot-offset 0` | slot height vs tag 10 center, cm |
| `--no-map` | use tag 10 only (ignore map.yaml) |
| `--ignore-tags 6,7` | skip specific mapped tags |

**Keys** (window mode): `q` = land all, `SPACE` = EMERGENCY motor stop. `Ctrl+C` = land all.

**Placement:** drones 2-3 m in front of tag 10 (within +-60 deg), 1 m+ apart,
**facing tag 10** (search becomes instant). More drones -> the arc moves further out
(5 drones ~1.5 m, 6 drones ~2 m from tag 10).

**Logs:** `logs/formation_<date>_<time>.csv` (state, position, commands, battery per drone).

---

## 4. Multi hover (`multi_hover.py`, no video)

All selected drones take off together, hover, land. Good for testing many drones.
```
python multi_hover.py --config ../config/dongles.json --only 1,2,3,4,5,8,9 --hover 10
```
All-or-nothing preflight (one failure -> nobody takes off). `Ctrl+C` lands all.
Drones ~1.5 m apart.

---

## 5. Attack + defense, scripted policies (`swarm_policies.py`, umbrella)

Runs one or more policies at once, each on its own drones: `--run <policy>:<drones>`,
repeatable. **Without `--fly` it is a dry-run.** One FLY prompt for everyone,
all-or-nothing preflight (formation drones included), `Ctrl+C` lands all.
Policies live one per file in `policies/` (add one there and register it in `policies/__init__.py`).

| Policy | What it does |
|---|---|
| `spin` | defense dummy: take off, hold the takeoff height, spin in place (`--yaw`, default 40) |
| `levels` | lower-numbered half at 3 ft (91 cm), the rest at 6 ft (183 cm), hover |
| `hover` | hold the takeoff height, or `--height <cm>` (40-250) |
| `formation` | `swarm_formation.py` on those drones, as its own process |
| `defend` | the sim-trained 3-defender policy on real drones (section 5.1) |

```
python swarm_policies.py --run spin:6,7,9 --run formation:4,5           # dry-run
python swarm_policies.py --run spin:6,7,9 --run formation:4,5 --fly     # defense spins, attack forms up
python swarm_policies.py --run hover:4,5,6,7,9 --height 200 --fly --duration 60   # photo hover at 2 m
python swarm_policies.py --run levels:4,5,6,7,9 --fly
```

**Options**
| Flag | Meaning |
|---|---|
| `--run spin:6,7,9` | a policy and its drones (repeat per group; a drone in one group only) |
| `--fly` | actually fly (without it: dry-run, 15 s) |
| `--duration 60` | land all after this many s (default 20; with `formation`: until it lands) |
| `--height 200` | `hover` height above the floor, cm |
| `--yaw -60` | `spin` speed and direction (-100..100) |
| `--formation-args "--fast --hold 15"` | passed to `swarm_formation.py` |

With `formation`, its camera windows and keys (`q`, `SPACE`) still work, but its
`--headless` terminal keys don't (its stdin is the umbrella): use `Ctrl+C` in the
umbrella's terminal. Formation drones only avoid each other: keep the other groups out
of the space in front of tag 10. Safety: a drone with no status stream for 3 s or
battery < 15 % lands on its own; takeoff needs battery >= 40 %.

### 5.1 `defend`: the trained anti-drone policy on real Tellos

`policies/defend.py` flies `simulation_infra/policies/defend_3v2_ram9_s2/defender_{0,1,2}.pt`
(3 defenders vs 2 attackers, 34-dim obs; sim: 100 % interception / 0 % breach with simultaneous and
with staggered attacker launches; goalkeepers hold a 0.8 m ring around the site). Settings:
`../config/defend.yaml`; another version via `--defend-dir`.

```
python connect.py 4 5 6
python swarm_policies.py --run defend:4,5,6                 # dry-run: no takeoff, obs + rc every tick
python swarm_policies.py --run defend:4,5,6 --fly --duration 40
python swarm_policies.py --run defend:4,5,6 --defend-dir ../../simulation_infra/policies/defend_3v2_smart2_dagger4   # another version
python defend_offline.py --seconds 14                        # no drones at all: kinematic demo (a defender
                                                             # that rams is "spent" and drops out, as in the sim)
python defend_offline.py --seconds 14 --stagger 3            # arena case: the 2nd attacker launches 3 s late
python -m pytest test_defend.py -q                           # frame / rc / obs / policy checks
```
The first drone in `--run defend:a,b,c` is `defender_0`, the second `defender_1`, the third
`defender_2` (the goalkeeper on the attacker nearest the site). Needs `torch` in the venv
(the Isaac venv has it: `source ~/isaac/activate.sh`) and the localizer package:
`pip install -e ../cal_and_map` (plus `pyyaml`, `av`, `pupil-apriltags`).

**How it works, per drone and tick (13.5 Hz)**
1. **Where am I**: the drone's own video (`streamon`, UDP 11111 bound to its adapter) ->
   `core/positional.py` (`Positioner`, any mapped tag of `map.yaml`) -> position + heading in the
   tag-10 frame; velocity = finite difference of the fixes over 0.4 s.
2. **Frames**: tag frame T = x right along the wall (facing the tag), y up, z out of the wall.
   Sim world S = (T_z, T_x, T_y): sim x out of the wall, sim y right along the wall, sim z up.
   The protected site (sim: the cylinder centre) is `site_xyz` in T, default 0.3 m below tag 10
   and 0.6 m into the room.
3. **Observation** (34, as `tello_defend_env.py::_get_observations`): own position minus the
   site (3), own world velocity (3), gravity in the body frame = [0,0,-1] (3), per attacker:
   relative position (3) and relative velocity (3), both zeroed when it is down, alive flags (2),
   then per teammate (ascending defender index): its position relative to me (3, clipped +-12 m,
   from its own tag fix, last known when lost) and an `active` flag (1 flying; 0 when landed or
   landing, its status stream stopped for 3 s, its height reads below 30 cm / on the ground for 1 s
   (rammed), or no fix for 6 s: until then it is still in the air and teammates keep avoiding its
   last position); last the assignment feature: a one-hot of the attacker this
   defender is assigned to and a goalkeeper flag, computed on every drone from the same numbers
   with the sim's intercept-time rule (`chase.py::assignment_from_obs`, ported in
   `assignment_feature`), so no communication is needed. Older nets get the same data in their
   own width: 31 (`defend_3v2_ram6_s2`, no assignment), 29 (`defend_3v2_ram4_*`, no flags) or 23
   (`defend_3v2_smart2_dagger4`, no teammate block).
4. **Policy**: `defender_N.pt` (TorchScript, CPU, normalisation inside) -> `[vx, vy, vz, yaw]` in
   [-1, 1], WORLD frame, 1.0 = 1 m/s. Another version: `--defend-dir <folder>` (e.g.
   `../../simulation_infra/policies/defend_3v2_ram5_A`), or `policy_dir` in `defend.yaml`.
5. **rc**: rotate `[vx, vy]` by minus the drone's heading: `b` (forward) = vx cos + vy sin,
   `a` (right) = vx sin - vy cos, `c` = vz, each x 100 x `speed_cap` (default 0.5 -> rc +-50).
   Horizontal rc below 14 is raised to 14 (a Tello ignores smaller sticks). `d` (yaw) does NOT
   come from the policy: a P-controller keeps the camera on the site/tags so the drone keeps
   knowing where it is (the policy never observes yaw, so its world-frame velocity is unaffected).
6. **Separation guard** (`sep_guard_m` 0.6 / `sep_release_m` 0.8): two defenders closer than
   0.6 m -> the one with the HIGHER defender index holds horizontally (`a b` = 0, logged) until they
   are 0.8 m apart again. The lower index keeps flying. Inactive teammates (see the `active` flag)
   are ignored by the guard: a defender that rammed is down anyway.
7. **Safety**: fix older than 0.5 s -> `rc 0 0 0 0` (hover); no fix for 6 s -> that drone lands;
   `fence` (tag-frame box) and `height_cm` (tof) zero any rc pushing further out; the umbrella's
   battery / status-stream guards still apply. Preflight also needs video frames and, with `--fly`,
   a tag fix from every defender (place them 1-2.5 m in front of the wall, **facing tag 10**).

**Attackers** (`attackers:` in `defend.yaml`)
- `scripted` (default, for testing with no real attackers): each attacker flies straight from
  `start` at `speed` toward the site after `delay`; it is "down" within `capture_radius` of a
  defender or on arrival, and restarts after `respawn_s`.
- `telemetry` (**the arena setting**): the organisers fly the attackers with
  `drone_gui/drone_gui.py`, which publishes every drone as UDP JSON v1
  (`../drone_telemetry/README.md`, default `127.0.0.1:9100`). Ask them to add **our laptop's LAN
  address:9100** to its destinations (GUI telemetry box or `roles.yaml`; not a 192.168.10.x
  address). We read the `team: attack` entries by slot (or `attacker_names`), an attacker is alive
  while `fly` is true and its `pose` is younger than 0.8 s; a drone that landed or rammed keeps its
  last position with alive 0. Its `pose.xyz` is the camera in **its own tag's** frame (x right,
  **y down**, **z into the wall**); we flip y and z, move it to the tag-10 frame with that tag's
  `R, t` from `map.yaml` (a tag not in the map gives no position: map every tag the attackers
  can see), then into the sim frame. Velocity is differenced here (the `vel` field is body-frame).
  Check the chain without hardware: `drone_gui.py --sim 8` on the same machine, then
  `python defend_offline.py --attackers telemetry --seconds 10`: the dry-run lines show `att=`
  (what we use) next to `truth=` (the GUI's ground truth, sim source only). Checked 2026-10-24
  on the box with `attacker_names: [SIM-4, SIM-6]` (sim i looks at tag i; only tags 4, 6, 10 are
  mapped): distance from the wall and height agree within ~5 cm; the along-wall coordinate is off
  by 10-20 cm and can flip momentarily, the telemetry README's known mirror ambiguity of small
  tags, so expect the same jitter on real attackers (the policy saw 3 cm fix noise in training,
  not this; a bigger tag for the attackers' side would help).
  Both programs bind each dongle's Tello ports, so in the arena drone_gui runs on the organisers'
  laptop and swarm_policies on ours.
- `udp` (our own test feed): JSON datagrams to `udp_port` (5005): `{"attackers": [[x,y,z],
  [x,y,z]]}` in the **tag-10 frame, metres**, `null` for an attacker that is down or unseen (an
  entry not refreshed for 1 s counts as down).

**Dry-run output** (`--run defend:... ` without `--fly`, and `defend_offline.py`): every tick per
drone `p` (sim frame), `yaw`, the attackers, the 34 obs values, the action and the rc. Sanity:
a drone facing the wall (yaw ~180) with an attacker behind it gets `b` < 0; the first print
after `fix:` in preflight should put the drone at sim x ~ 1-2.5 (its distance from the wall),
sim z ~ -0.5..0 (below tag 10). A `WARNING: fix behind the wall` means the map frame is mirrored:
check `map.yaml`'s `frame` line before flying.

Not done yet / to check in the arena: `site_xyz` and `fence` for the real room; the attacker
tracker (`udp`); the Tello's real response to rc +-50 vs the sim's 0.5 m/s (`speed_cap`).
**Telemetry between policies** (`policies/telemetry.py`). Any policy can publish live data
and read what the others publish; only the latest value per topic is kept, with its age.
```python
self.bus.publish("spin/drone6", {"yaw_rc": 40})        # in step() or anywhere
hit = self.bus.latest("formation/drone4", max_age=0.5)  # (data, age_s) or None if missing/stale
self.bus.since("formation/")                            # {topic: (data, age_s)}
```
In-process policies get the umbrella's bus as `self.bus`. A policy in its own process uses
`TelemetryClient.from_env()` (same calls, over localhost UDP to the umbrella; a private bus
when run alone). Published automatically every tick:
| Topic | From | Data |
|---|---|---|
| `status/drone<N>` | umbrella, its own drones | `policy`, `height_cm`, `battery`, `rc` |
| `formation/drone<N>` | `swarm_formation.py` | `state`, `p` (m, map frame; `None` = no fix yet), `psi_deg`, `fix_age`, `height_cm`, `rc` |

Steer on telemetry only while it is fresh (`max_age`): a stalled publisher must not steer a drone.

---

## 6. Map the tags (`map_tags.py`, no flying)

Needed whenever tags are added or moved. Hold the drone (props off) so it sees
tag 10 together with each other tag until all are green, then press `q` to save.
```
python connect.py 4
python map_tags.py --drone 4 --tags-list 10,6,7
```
Every tag needs its size in `../config/tags.yaml` (`sizes_mm`, black square in mm).
Check: dry-run shows no `MAP CHECK` warnings.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT DETECTED` / `OSError: No such device` | adapter unplugged or hub unpowered: replug, `nmcli dev status \| grep wlx` |
| `drone not broadcasting` | drone off / asleep / flat battery: power it on |
| Drones switch off while connecting | idle auto-off / heat: power on just before flying, connect right away |
| Takeoff failed (height 0) | low battery, hot, tilted surface, or IMU calibration needed (Tello app) |
| Pixelated video with 5+ drones | 2.4 GHz crowding: power drones on ~10 s apart, spread adapters, EDU video lower (see `EDU_MODIF_README.md`) |
| `Map: tags [...]` doesn't match the wall | re-map with `map_tags.py` |
| Wi-Fi name `TELLO-TELLO-6` | in the Tello app type only `6` (it adds `TELLO-`) |
