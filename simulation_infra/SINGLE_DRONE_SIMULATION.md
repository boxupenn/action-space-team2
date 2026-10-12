# Single-drone simulation: train, tune, watch

The guide is written for the single-drone task (`AS-Tello-Room-Single-v0`); every command
also runs the 4-drone task with `--task AS-Tello-Room-v0`.

Two Isaac Lab tasks fly simulated DJI Tellos in the real mapped room. In both, a drone's
**only** position source is its AprilTag sightings, the same as `core/positional.py` on the laptop.

| Task | Drones | Start | Goal | Use it for |
|---|---|---|---|---|
| `AS-Tello-Room-Single-v0` | 1 | on the floor, then the SDK `takeoff` | land on one of the 4 spots (random) | **tuning rewards** (fast to learn, easy to read) |
| `AS-Tello-Room-v0` | 4, one central controller | hovering (takeoff already done) | each lands on its own spot, no contact | the full formation |

Both start inside a 2 m × 2 m box, 6–7 m from tag 10. The landing spots are on the floor
1 m in front of the tag-10 wall, 0.5 m apart. Each spot has a **waypoint 1.5 m above it**
(radius 0.3 m).

---

## 1. Shell setup (every new terminal)

Isaac Lab lives in a uv virtual environment, **not** a conda env. Just the environment:

```bash
source ~/isaac/env_isaaclab/bin/activate
```

The full setup, for every new terminal:

```bash
cd ~/Action_Space/simulation_infra && unset CONDA_PREFIX CONDA_DEFAULT_ENV && source ~/isaac/env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd && mkdir -p outputs
```

Every command below assumes this shell, in `simulation_infra/`. Always run Python with `-u`
and send output to a log under `outputs/`: Isaac Sim prints hundreds of harmless warnings,
and it can drop buffered output when it exits. Conda's `(base)` can stay active, but the
`unset` is required: otherwise Isaac Lab picks conda's python.

---

## 2. Check the environment first (~3 min)

```bash
python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Single-v0 > outputs/check_room_single.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero |PASSED|FAILED" outputs/check_room_single.log
```

Pass: 21 `[ok]` lines and `ALL CHECKS PASSED`. For the 4-drone task, drop `--task` (49 checks).
To run only the Kalman filter checks: add `--test kalman`.

---

## 3. Tune the rewards

### Where they are

The reward weights are in
`source/action_space_kit/action_space_kit/tasks/direct/tello_room/tello_room_env_cfg.py`
(the `# -- rewards` block). The single-drone task inherits them. All are per drone, per policy step (74 ms):

| Field | Default | Logged as | What it does |
|---|---|---|---|
| `rew_closer_scale` | `10.0` | `closer_reward` | + per metre gained toward the **waypoint**. Only until the waypoint is reached, and never during takeoff or land |
| `rew_away_ratio` | `0.5` | `closer_reward` | moving away costs this fraction of `rew_closer_scale` per metre (−5 per metre by default) |
| `rew_time` | `-0.5` | `landing_loss` | every step until landed on the spot; ending early (crash, landing off the spot) pays the remaining steps |
| `rew_no_fix` | `-0.5` | `no_tag_loss` | every step the drone flies with no tag fix |
| `rew_violation` | `-100.0` | `crash_loss` | crash, wall, ceiling, flipping, contact with another drone |
| `rew_early_land` | `-100.0` | `early_land_loss` | **land gate**: each `land` asked for before the waypoint is blocked (not sent) and costs this |
| `rew_land` | `100.0` | `land_reward` | at touchdown. Land is only sent after the waypoint, so this only ever pays after the bubble |
| `rew_all_landed` | `200.0` | `formation_reward` | success bonus: every drone landed within `success_radius` of its spot |
| `rew_scale_separation` | `-20.0` | `separation_loss` | per second, closer than 0.5 m to another drone (always 0 with one drone) |
| `rew_scale_action_rate` | `-0.05` | `action_rate_loss` | per second, jerky sticks |

Geometry that shapes the rewards:

| Field | Default | Meaning |
|---|---|---|
| `waypoint_height` | `1.5` m | waypoint height above the spot |
| `waypoint_radius` | `0.3` m | inside this (by the published tag fix) counts as reached: from then on, only the landing rewards and the penalties pay |
| `success_radius` | `0.3` m | touchdown this close to the spot counts as on the spot |
| `episode_length_s` | `75` s | episode cap |
| `takeoff_height` | `(0.8, 1.0)` m | where the firmware takeoff stops (single-drone task) |

The formulas themselves are in `_get_rewards()` in `tello_room_env.py`.

### Two ways to change them

**Per run, from the command line (recommended for tuning):** append `env.<field>=<value>`
to the train command. The code is untouched, and the values used are saved with the run in
`logs/sb3/<task>/<run>/params/env.yaml`.

```bash
... env.rew_closer_scale=20.0 env.rew_no_fix=-0.25 env.rew_land=150.0
```

**Permanently:**
- **Single-drone task only:** add the field to the class in `tello_room_single_env_cfg.py`
  (e.g. `rew_closer_scale = 20.0`).
- **Both tasks:** edit the defaults in `tello_room_env_cfg.py`.

### Reading the result

Each term gets its own graph, `Rewards/avg_<term>_per_episode` (averaged per drone, per
episode), so you can see which term dominates. Outcomes are under `Episode/`:

- `success_rate`
- `drones_reached_waypoint`
- `drones_landed`
- `drones_landed_on_spot`
- `spot_err_m`
- `crash_rate`
- `duration_s`
- `tag_fix_frac`
- `tag_losses`

---

## 4. Train

### Headless (full speed: use this for real runs)

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  env.rew_closer_scale=10.0 env.rew_time=-0.5 env.rew_no_fix=-0.5 env.rew_land=100.0 \
  > outputs/train_room_single.log 2>&1 &
```

**2048 envs, `closer_reward` +2 per metre** (a 4x bigger batch per iteration; same PPO
settings, minibatches of 131k samples; ~20 s per iteration and ~26.6k steps/s on the RTX 5080,
so 1500 iterations is ~8.3 h):

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 2048 --max_iterations 1500 \
  env.rew_closer_scale=2.0 > outputs/train_room_single_2048.log 2>&1 &
```

- The run keeps going after you close the terminal.
- Output: `logs/sb3/AS-Tello-Room-Single-v0/<timestamp>/`. Checkpoints are saved every 50
  iterations (`model_<steps>_steps.zip` + `model_vecnormalize_<steps>_steps.pkl`), plus
  `model.zip` at the end.
- To follow it: `grep -E "success_rate|drones_landed_on_spot|iterations" outputs/train_room_single.log | tail`
- To stop it: `pkill -TERM -f "[A]S-Tello-Room-Single"`
- To resume from a checkpoint: add `--checkpoint logs/sb3/AS-Tello-Room-Single-v0/<run>/model_<steps>_steps.zip`

For the 4-drone task, use `--task AS-Tello-Room-v0`. It takes about 27 s per iteration at 512 envs, so 1500 iterations is about 11 h.

### With or without the Kalman filter

The position the policy sees is, by default, the **Kalman-filtered** estimate from
`drone_deployment_infra/core/positional_kalman_filter.py`, the same file the laptop uses.
`env.kalman.enabled` switches it; the tag noise is the same either way. The setting is saved
with each run in `logs/sb3/<task>/<run>/params/env.yaml` (`kalman: enabled: ...`).

**With the filter:**

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=True > outputs/train_room_single_kalman.log 2>&1 &
```

**Without the filter** (the raw per-frame fix, as `core/positional.py` alone gives it):

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=False > outputs/train_room_single_raw.log 2>&1 &
```

- **Running both at once** to compare them works (separate logs and run folders), at about
  half speed each. In TensorBoard, tell them apart by start time, or check each run's `params/env.yaml`.
- **Deploying:** a policy trained with the filter must fly with the filter, and one trained
  without it must fly without it. On the laptop, use `FilteredPositioner` from
  `positional_kalman_filter.py` in place of `Positioner`.
- **Filter tuning** (`env.kalman.<field>=<value>`, same values on the laptop: `KalmanCfg` in that file):

  | Field | Default | Meaning |
  |---|---|---|
  | `accel_sigma` | `1.5` m/s² | how hard the drone may accelerate between ticks (higher: follows faster, smooths less) |
  | `meas_sigma` | `0.15` m | noise assumed on one fix (higher: trusts fixes less, smooths more) |
  | `gate_chi2` | `16.27` | outlier gate: fixes further than this (Mahalanobis²) from the prediction are dropped |
  | `max_rejects` | `3` | this many dropped in a row: restart from the fix |

- **Filter self-test, no Isaac needed:** `python3 ../drone_deployment_infra/core/positional_kalman_filter.py`.
  Pass: `SELFTEST PASSED`.

Two training runs at once share the GPU, and each goes about half speed.

### With rendering (watch it learn)

Run these at the desktop: they open the Isaac Sim window. No `setsid nohup ... &`: closing
the window ends the run. The **first** GUI launch compiles shaders for a minute or more
with a blank window: wait, don't Ctrl-C. It learns slower than headless; use it to check
behaviour.

**Every env is overlapped in one room** (the default, `env.scene.env_spacing=0.0`): all the
drones of the batch fly in the same room, so you see the whole batch at once: where the policy
sends drones, how many reach the waypoint, where they land. Overlapping changes only the
picture, not the training: PhysX filters contacts between envs, and crash / collision /
rewards use each env's own positions (measured: no slower than a grid).

**All 512 envs, with the Kalman filter:**

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=True env.debug_vis=True > outputs/train_room_single_gui_kalman.log 2>&1
```

**All 512 envs, without the Kalman filter:**

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --num_envs 512 --max_iterations 1500 \
  env.kalman.enabled=False env.debug_vis=True > outputs/train_room_single_gui_raw.log 2>&1
```

**Fewer drones on screen** (16 envs, a lighter window; learns much slower):

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --num_envs 16 --max_iterations 1500 \
  env.debug_vis=True > outputs/train_room_single_gui.log 2>&1
```

To lay the envs out side by side instead (rooms 14 m apart), add `env.scene.env_spacing=14.0`.

- `env.debug_vis=True` draws the markers (see **In the frame** below). Leave it off headless.
- Camera: `viewer` in `tello_room_env_cfg.py` (env 0, the "overview" view of `record_room.py`).
  Isaac Sim's *Viewer Settings* panel (IsaacLab tab) can move it to another env.
- Any of these also works headless: add `--headless` (and `setsid nohup ... &`) for full speed.

PPO settings are in `tasks/direct/tello_room/agents/sb3_ppo_single_cfg.yaml` (or
`sb3_ppo_cfg.yaml` for 4 drones). Override them the same way: `agent.learning_rate=1e-4`.

---

## 5. Dashboard (TensorBoard from anywhere)

```bash
dashboard/serve.sh     # prints a public https://….trycloudflare.com URL (also in dashboard/URL.txt)
dashboard/stop.sh
```

It shows every run under `logs/sb3/`, and new runs appear on their own. The quick-tunnel URL
is public and changes each time you restart it. See `dashboard/README.md`.

---

## 6. Watch a trained policy

### Video (headless, mp4)

```bash
python -u scripts/record_room.py --task AS-Tello-Room-Single-v0 --episodes 3 \
  --out outputs/videos/room_single.mp4 > outputs/record_room_single.log 2>&1
grep "\[rec\]" outputs/record_room_single.log
```

- **Checkpoint:** with no `--checkpoint`, it records the newest checkpoint of the newest
  run, so it works mid-training. `--checkpoint <path.zip>` picks one.
- **Real time:** one frame per rc command.
- **Options:**
  - `--view overview|side|top`
  - `--seed N`: different starts and spot assignments
  - `--no_dr`: nominal dynamics
  - `--pilot zero`: no policy, a scene check
  - `--set field=value`: same fields as `env.`
- **In the frame** (and in the GUI with `env.debug_vis=True`), each drone has one color
  (single drone: orange), shared by:
  - small sphere: its true position
  - cube: its published tag fix: where the drone *believes* it is, i.e. the position the
    policy observes. It comes only from AprilTag sightings, 0.2 s late, with 3 cm noise and
    a 3 cm bias, so it trails and jitters around the sphere; it freezes when no tag is in view
  - translucent sphere: its waypoint
  - disc: its spot
  - bottom panel: each drone's phase (`TAKEOFF`, `to waypoint`, `at waypoint`, `LANDING`,
    `LANDED on/OFF spot`, `LOST`), the rc it sends and its fix age.

### Live, with the GUI

```bash
python -u scripts/sb3/play.py --task AS-Tello-Room-Single-v0 --num_envs 64 --real-time env.debug_vis=True
```

All 64 drones fly in the same room (overlapped). Pass the same `env.kalman.enabled=...` the
policy was trained with.

### Score a checkpoint (headless, numbers only)

```bash
python -u scripts/sb3/play.py --task AS-Tello-Room-Single-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint logs/sb3/AS-Tello-Room-Single-v0/<run>/model.zip > outputs/play_single.log 2>&1
grep "episodes\]" outputs/play_single.log | tail -1
```

Pass the same `env.*` overrides you trained with.

---

## 7. What the policy sees and sends

**Observation:** per drone, 24 numbers, plus 1 taking-off flag in the single-drone task.
PPO stacks the last 8 steps. The numbers:

- position (x, y, z): the Kalman estimate (`env.kalman.enabled=False`: the last raw tag fix), and heading
- whether a fix arrived this tick, and seconds since the last fix
- `lost`: no fix for 0.4 s. The position freezes until a tag is back in view (the filter stops
  extrapolating and restarts from the next fix); there is no odometry
- waypoint and landing spot, relative, in the drone's heading frame
- waypoint-reached flag
- bearing to tag 10
- Tello height `h`
- its previous command
- landing flag

Noise on the fixes:
- the camera model: field of view, range, view angle, pixel size, dropout, 0.2 s video delay
- 3 cm noise per fix
- 3 cm bias per drone
- 0–15% link loss per drone

**Kalman filter (on by default):** constant velocity per axis. It predicts every tick, fuses
each fix, and drops outlier fixes (e.g. pose flips). Same file and settings as on the laptop. In
the checks, flying at rc 35, it puts the position ~10 cm from the truth vs ~13 cm for the raw fix.

**Action:** the real Tello commands.
- `rc right forward up yaw`: integers −100…100, sent once per 74 ms step (13.5 Hz, the
  measured flight rate), with 0–16 ms jitter.
- A 5th output above 0.5 for 3 steps in a row asks for `land`. **Land gate:** it is sent only
  once the drone's published fix has been inside its waypoint (the observation's
  waypoint-reached flag). Before that, it is blocked and costs `rew_early_land`. The laptop
  must apply the same gate at deploy.

At deploy, send `env.last_rc` as `rc a b c d`.

**Dynamics:** randomized per drone, per episode. No wind.
- thrust-to-weight ×0.95–1.05
- rotor drag ×0.5–2.0
- body-rate PID: kp/ki ×0.85–1.15, kd ×0.7–1.3
- rc → speed gains from `control/tello_motion_fit.yaml`, fitted from flight 027

**Takeoff (single-drone task):**
- The drone starts on the floor with motors off.
- The firmware climbs at 0.5 m/s to 0.8–1.0 m, ignoring sticks and `land` (about 2.2 s),
  then the policy flies.
- Takeoff steps pay no closer reward and no no-tag loss, but the landing loss runs.

To start the 4-drone task from the floor too, set `start_on_floor = True` in its config.
That changes its observation size, so it needs a fresh training run.

---

## 8. Files

| Path (under `source/action_space_kit/action_space_kit/`) | What |
|---|---|
| `tasks/direct/tello_room/tello_room_env_cfg.py` | **all knobs**: rewards, geometry, noise, dynamics ranges |
| `tasks/direct/tello_room/tello_room_single_env_cfg.py` | single-drone overrides |
| `tasks/direct/tello_room/tello_room_env.py` | the env: rewards (`_get_rewards`), dones, observation, reset |
| `tasks/direct/tello_room/agents/sb3_ppo*_cfg.yaml` | PPO settings |
| `control/rc_action.py`, `control/tello_motion_fit.yaml` | rc → speed (fitted from real flights) |
| `tags/` | room map, tag sensing model, tag textures |
| `tags/kalman.py` | runs the laptop's filter (`drone_deployment_infra/core/positional_kalman_filter.py`) in the sim, on torch |
| `rl/sb3_shared.py`, `rl/policy_export.py` | SB3 wrappers; checkpoint → TorchScript |

| Script (`scripts/`) | What |
|---|---|
| `check_room_env.py` | environment checks |
| `sb3/train.py`, `sb3/play.py` | train / score / watch (GUI unless `--headless`) |
| `record_room.py` | video |
| `render_room_overview.py` | one annotated still of the room |
| `fit_tello_motion.py` | re-fit the motion from new `recordings/*.train.jsonl` |

The room itself is read live from `drone_deployment_infra/config/map.yaml` + `tags.yaml`.
The floor height under tag 10 (`tag10_height`, 1.2 m) is **assumed**: measure it.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named isaaclab` / wrong python | redo the shell setup (step 1); `CONDA_PREFIX` must be unset |
| log stops with no error | Isaac Sim drops buffered output; you forgot `-u` |
| first launch hangs for minutes / blank GUI window | shader compilation, first run only: wait, don't Ctrl-C |
| GUI opens but no drones visible | add `env.debug_vis=True`; the camera looks at the start box (all envs overlap there) |
| `TELLO_USD_PATH` / asset not found | `export TELLO_USD_PATH=$HOME/isaac/assets/tello.usd` |
| obs shape mismatch loading a checkpoint | the checkpoint is from the other task (25 vs 96 inputs) or an older observation layout |
| training slow | another run or recorder is using the GPU: `nvidia-smi` |
