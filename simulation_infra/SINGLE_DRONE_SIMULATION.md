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
1 m in front of the tag-10 wall, 0.5 m apart. Each spot has a **waypoint 1.22 m above it**
(radius 0.3 m). Entering it counts only at or below 0.3 m/s (`waypoint_max_speed`, the Kalman
filter's speed estimate, judged once the filter's velocity is settled: `speed_trust_std`).
New runs use reward version 2: fast entries can recover, progress uses discounted potential
shaping, and accurate touchdown is the main reward. Reaching the waypoint does not by itself
authorize landing: the current estimated pose, speed, uncertainty and freshness must pass
the shared landing gate continuously for 0.5 seconds.

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

When using an isolated checkout, ensure its source wins over any editable installation:

```bash
export PYTHONPATH="$PWD/source/action_space_kit${PYTHONPATH:+:$PYTHONPATH}"
```

---

## 2. Check the environment first (~3 min)

```bash
python -u scripts/check_room_env.py --headless --task AS-Tello-Room-Single-v0 > outputs/check_room_single.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero |PASSED|FAILED" outputs/check_room_single.log
```

Pass: all `[ok]` lines and `ALL CHECKS PASSED`.
To run only the Kalman filter checks: add `--test kalman`.

The policy-specific checks accept any saved single-drone checkpoint; the integration check
temporarily uses two-second episodes to exercise timeout/reset behavior:

```bash
python -u tests/test_policy_contract.py > outputs/policy_contract_tests.log 2>&1
python -u scripts/check_policy_pipeline.py --headless --checkpoint <run>/model.zip > outputs/policy_pipeline.log 2>&1
python -u tests/check_export_replay.py --checkpoint <run>/model.zip > outputs/export_replay.log 2>&1
```

These check terminal observations through normalization/frame stacking, exact episode counts,
landing-gate failures, four controlled touchdowns, and simulator/adapter/export inference parity.

---

## 3. Tune the rewards

### Where they are

The reward weights are in
`source/action_space_kit/action_space_kit/tasks/direct/tello_room/tello_room_env_cfg.py`
(the `# -- rewards` block). The single-drone task inherits them. All are per drone, per policy step (74 ms):

| Field | Default | Logged as | What it does |
|---|---|---|---|
| `rew_potential_scale` | `40.0` | `closer_reward` | bounded potential toward waypoint, then descent; `gamma * Phi(next) - Phi(previous)`, zero next potential on true termination |
| `reward_gamma` | `0.995` | — | must equal PPO gamma; timeouts retain their final-state potential and bootstrap |
| `rew_waypoint` | `25.0` | `waypoint_reward` | once on slow waypoint entry; does not open the landing gate by itself |
| `rew_settle_position` / `rew_settle_speed` | `-2.0` / `-1.0` | `settling_loss` | per-second position and speed costs after waypoint entry, before landing |
| `rew_slowdown` | `0.0` | `slowdown_reward` | legacy new-low-speed term, disabled by default |
| `rew_time` | `-0.5` | `landing_loss` | every step until landed on the spot; ending early (crash, landing off the spot) pays the remaining steps |
| `rew_no_fix` | `-0.5` | `no_tag_loss` | every step the drone flies with no tag fix |
| `rew_violation` | `-100.0` | `crash_loss` | crash, wall, ceiling, flipping |
| `rew_collision` | `-1000.0` | `collision_loss` | each drone within 0.25 m of another (4-drone task); ends the episode |
| `rew_early_land` | `-5.0` | `early_land_loss` | confirmed requests before the waypoint; afterward blocked requests wait for the gate without a repeated penalty |
| `rew_land` | `1000.0` | `land_reward` | only touchdown within the assigned spot's success radius |
| `rew_all_landed` | `200.0` | `formation_reward` | success bonus: every drone landed within `success_radius` of its spot |
| `rew_scale_separation` | `-20.0` | `separation_loss` | per second, closer than 0.5 m to another drone (always 0 with one drone) |
| `rew_scale_action_rate` | `-0.05` | `action_rate_loss` | per second, jerky sticks |

Geometry that shapes the rewards:

| Field | Default | Meaning |
|---|---|---|
| `waypoint_height` | `1.22` m | aligned with the measured tag center; the old 1.5 m target clips the large tag in the camera view |
| `waypoint_radius` | `0.3` m | slow entry counts as reached; settling guidance continues afterward |
| `success_radius` | `0.3` m | touchdown this close to the spot counts as on the spot |
| `episode_length_s` | `75` s | episode cap |
| `takeoff_height` | `(0.8, 1.0)` m | where the firmware takeoff stops (single-drone task) |

The formulas themselves are in `_get_rewards()` in `tello_room_env.py`.

### Two ways to change them

**Per run, from the command line (recommended for tuning):** append `env.<field>=<value>`
to the train command. The code is untouched, and the values used are saved with the run in
`logs/sb3/<task>/<run>/params/env.yaml`.

```bash
... env.rew_potential_scale=30.0 env.rew_no_fix=-0.25 env.rew_land=1000.0
```

**Permanently:**
- **Single-drone task only:** add the field to the class in `tello_room_single_env_cfg.py`
  (e.g. `rew_potential_scale = 30.0`).
- **Both tasks:** edit the defaults in `tello_room_env_cfg.py`.

### Reading the result

Each term gets its own graph, `Rewards/avg_<term>_per_episode` (averaged per drone, per
episode), so you can see which term dominates. Outcomes are under `Episode/`:

- `success_rate`
- `drones_reached_waypoint`
- `waypoint_reached_pct`: % of drones that got inside their waypoint, per episode
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
  --curriculum landing env.auto_land=True env.episode_length_s=15.0 \
  > outputs/train_room_single.log 2>&1 &
```

Advance through `--curriculum landing`, `approach`, then `full`, evaluating each stage before
advancing. All three keep the same 25 observations and five actions. Landing starts hovering
within 10 cm per horizontal axis of its assigned pad (`spawn_near_target_radius`); approach
restores takeoff nearer the target; full restores the 6–7 m start. Monitor
`Episode/land_gate_ready_frac` to distinguish failure to settle from failure to request land.
`--checkpoint` resumes both PPO and its matching normalization statistics and transition count.
`--max_iterations` specifies additional iterations on resume.

The command above uses gated auto-land: PPO flies the RC channels, and the controller sends
SDK land only when the current-state gate passes. This avoids depending on an undertrained
fifth action. Use `env.auto_land=False` to research policy-requested landing, and qualify that
mode independently. The saved checkpoint config records the chosen mode.

The objective is the complete mission from the 6–7 m start, including learned approach,
braking, tag retention/reacquisition, and arrival at the assigned pad. A landing-stage score
does not qualify that mission. When a checkpoint already navigates the room, retain that
behavior by fine-tuning on full starts with shorter episodes mixed in:

```bash
python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 \
  --checkpoint <full-room-checkpoint>/model.zip --config-source current --curriculum full \
  --max_iterations 40 --save_interval 10 env.auto_land=True \
  env.spawn_short_fraction=0.30 env.spawn_hover_fraction=0.15 agent.learning_rate=0.0001 \
  > outputs/train_full_mission.log 2>&1
```

Fractions apply to episode resets: 55% keep the full start, 30% take off from X=1.5–5.5 m,
and 15% start hovering near their assigned pad. This changes only initial conditions;
navigation and recovery remain learned. Per-episode records include start mode and position.
Resume with saved config after the initial transition to the corrected task. Full/dropout
evaluation suites explicitly disable both replay fractions, so easy starts cannot inflate
the full-mission score. For a direct `play.py` evaluation of a mixed-start checkpoint, add
`env.spawn_short_fraction=0.0 env.spawn_hover_fraction=0.0`.

For a controlled fine-tuning experiment, `--action-std 0.2` reduces Gaussian exploration
on resume without changing the deterministic actor or value predictions. It also clears
the optimizer moments for that exploration parameter. `agent.freeze_normalization=True`
keeps the matching checkpoint's input and reward statistics fixed; this requires a
checkpoint. Both are optional, recorded in the run, and need full-mission evaluation
against the parent checkpoint before promotion. They do not add a navigation controller.

For example, after the landing stage passes its evaluations:

```bash
setsid nohup python -u scripts/sb3/train.py --task AS-Tello-Room-Single-v0 --headless --num_envs 512 --max_iterations 1500 \
  --checkpoint <landing-run>/model.zip --curriculum approach env.episode_length_s=75.0 \
  > outputs/train_room_single_approach.log 2>&1 &
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


### Watch the simulation live (Isaac Sim WebRTC stream, over Tailscale)

This is the real Isaac Sim viewport, streamed live: the newest checkpoint flying in real time,
16 envs overlapped in one room, with the markers on.

**On the desktop:**

```bash
setsid nohup python -u scripts/stream_room.py > outputs/stream_room.log 2>&1 &
grep "\[stream\]" outputs/stream_room.log      # "streaming ...", then one line per finished episode
```

- **Keeps up with training:** every 5 minutes it swaps in the newest checkpoint (`--reload_min`).
- **Options:**
  - `--checkpoint <zip>`: fly one fixed checkpoint
  - `--num_envs 8`: fewer drones on screen
  - `--task AS-Tello-Room-v0`: the 4-drone task
- **To stop it:** `pkill -TERM -f "[s]tream_room.py"`
- **Not `sb3/play.py --livestream`:** the stream loads an older numpy that can't read the
  checkpoint files, so `play.py` crashes. `stream_room.py` converts the checkpoint first,
  like the recorder.

**On the computer you watch from:**

1. **Install the client:** NVIDIA's *Isaac Sim WebRTC Streaming Client* (Isaac Sim docs, "Livestream
   Clients" page; Linux AppImage, Windows, macOS). Use the version made for Isaac Sim 5.x.
2. **Join the tailnet:** be on the same Tailscale network as the desktop (`tailscale status` lists it).
3. **Connect:** in the client, enter the desktop's Tailscale address, **`100.122.106.39`**, and
   connect. On the home network, `192.168.68.58` works too.

**Limits:**
- **Tailscale only:** the stream doesn't go through the Cloudflare links (they only carry web
  pages), so you need Tailscale or the same local network.
- **Cost:** it uses about 5 GB of GPU memory and some GPU time, so training is a little slower
  while it runs. With the video page also recording, the 16 GB card can run out of memory: run
  one or the other.
- **One viewer:** one client at a time.

### Watch the simulation render from anywhere (video page)

```bash
dashboard/render_serve.sh      # prints a public https://….trycloudflare.com URL (also in dashboard/RENDER_URL.txt)
dashboard/render_stop.sh
```

- **What it shows:** while training runs, `dashboard/render_loop.py` records one full episode
  of the newest checkpoint with `record_room.py`, at most once every 30 minutes. It uses the
  same task and Kalman setting the run trains with.
- **The page:** it plays the newest video and lists the earlier ones (the last 30). When a new
  one is ready it shows a "new video" button, so playback is never cut off.
- **Cost:** each recording takes about 10 min of GPU time while training runs (about 3 min on an
  idle GPU), so training slows a little while it records.
- **Options:** add them after `render_serve.sh`:
  - `--min_gap 900`: record every 15 min
  - `--max_seconds 30`: shorter clips, less GPU time
  - `--view top`: a top-down camera
- **Link:** it has its own URL, separate from TensorBoard. The URL is public and changes each
  time you start the page.
- **Live GUI:** a live stream of the Isaac Sim window (Isaac's WebRTC livestream) doesn't work
  through this kind of tunnel, so the page shows recorded episodes instead.
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
  - `--view overview|side|top|landing`: `landing` gives a close-up of the target area
  - `--pads 0 1 2 3`: single-drone attempts on each pad in order; overrides `--episodes`
  - `--hide_waypoints`: hide the waypoint spheres when they obscure the drone
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

The evaluator restores saved config automatically; explicit `env.*` overrides win. It seeds
after loading PPO and collects a fixed quota per simulator slot. `--report outputs/eval.json`
saves individual episode IDs, target assignments, results, and per-pad Wilson intervals.
Use `--config-source current` only to deliberately test a checkpoint under changed conditions.
Legacy runs have no external-asset snapshot, so their historical replay is approximate.

New runs save `params/run.json`, `params/source.tar.gz`, `params/working.diff`, dependency
versions, and copies/hashes of the map, tag sizes and motion fit. Missing normalization is an
error. The config snapshot is taken after motion-fit application.

```bash
python -u scripts/sb3/evaluate_suite.py --checkpoint <run>/model.zip --scenario full --out outputs/full_suite
python -u scripts/sb3/evaluate_suite.py --checkpoint <run>/model.zip --scenario dropout --out outputs/dropout_suite
```

Default suites cover three seeds and all four pads. Promotion requires at least 95% accurate
touchdowns and zero crashes in each seed/pad cohort; inspect confidence intervals too.
The one-drone checkpoint cannot be loaded as a four-drone central controller: their input and
output dimensions differ. Team training remains a separate step after single-drone validation.

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
- motion blur: a whole frame is lost with probability 0.3 per m/s of (speed + angular rate)
  above 0.2, capped at 0.6 (`tag_sensor.blur_*`), fitted to real flights 032-034
- 3 cm noise per fix
- 3 cm bias per drone
- 0–15% link loss per drone

**Kalman filter (on by default):** constant velocity per axis. It predicts every tick, fuses
each fix, and drops outlier fixes (e.g. pose flips). Same file and settings as on the laptop. In
the checks, flying at rc 35, it puts the position ~10 cm from the truth vs ~13 cm for the raw fix.

**Action:** the real Tello commands.
- `rc right forward up yaw`: integers −100…100, sent once per 74 ms step (13.5 Hz, the
  measured flight rate), with 0–16 ms jitter.
- By default, the fifth output above 0.5 for three steps requests `land`.
- Both requested landing and optional `env.auto_land=True` require the same gate: waypoint
  reached, projected XY error <=15 cm, horizontal speed <=0.1 m/s, height within 20 cm of the
  waypoint, position std <=12 cm, velocity std <=0.35 m/s, and accepted-fix age including video
  delay <=0.35 s, continuously for 0.5 s. Lost, uninitialized, or non-finite estimates block it.
  These are initial simulation tuning values, not hardware-validated limits.

`drone_deployment_infra/core/room_policy.py` implements inference, map/camera/body transforms,
normalization-before-frame-stacking, RC rounding, and the shared gate. Its CLI replays recorded
feeds offline and sends no commands. The flight engine owns takeoff, command acknowledgement,
and touchdown detection. Record full filter covariance and accepted-fix timestamps for replay.

**Dynamics:** randomized per drone, per episode. No wind.
- thrust-to-weight ×0.95–1.05
- rotor drag ×0.5–2.0
- body-rate PID: kp/ki ×0.85–1.15, kd ×0.7–1.3
- rc → speed gains from `control/tello_motion_fit.yaml`, fitted from flight 027

**Takeoff (single-drone task):**
- The drone starts on the floor with motors off.
- The firmware climbs at 0.5 m/s to 0.8–1.0 m, ignoring sticks and `land` (about 2.2 s),
  then the policy flies.
- Takeoff steps pay no no-tag loss; time cost and version-2 potential shaping continue.

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
Tag 10's centre is 1.22 m above the floor (`tag10_height`, measured ~48 in). The left wall,
facing tag 10, is 2 m left of tag 10's centre (`room_y_min` = -2.0).

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

## Research and calibration

The changes retain PPO and the existing Tello RC interface. The [Swift drone-racing
paper](https://www.nature.com/articles/s41586-023-06419-4) supports measuring dynamics and
perception errors on the real platform and incorporating them in simulation. Its thrust/body-rate
actions are not directly available through Tello RC, so its control interface is not copied here.
The [inclined-landing PPO paper](https://arxiv.org/abs/2103.09043) supports starting near a goal
and increasing difficulty gradually. These motivate our experiments; they do not validate our
particular thresholds or establish that this policy transfers to real Tellos.

```bash
python -u scripts/audit_flight_coverage.py '../drone_deployment_infra/recordings/*.train.jsonl' \
  --report outputs/flight_coverage.json > outputs/flight_coverage.log 2>&1
```

The coverage report identifies missing command directions, independent flights, pose rotations
and filter covariance. Fit and hold out entire flights, including different drones. Verify
telemetry velocity units and camera/map/body axes before fitting. The historical fit's one-sided
command envelope (no positive up) is insufficient for this task; do not blindly enable it and
make climbing impossible. No updated dynamics fit should replace the current one until those
data gaps and held-out residuals have been checked.
