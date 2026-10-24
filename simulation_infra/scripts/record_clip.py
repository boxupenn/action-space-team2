"""Record a clip of the defend scenario by capturing the Kit viewport.

Replicator's render-variable path is broken on this machine, which takes out camera
sensors and Isaac Lab's built-in video recording. Kit viewport rendering works -- it is
what feeds the WebRTC livestream -- so this captures viewport frames and stitches them
into an mp4 with ffmpeg.

Captures are asynchronous: each one needs the app to tick a few times before the file
lands, which is why the tick counts below are generous.

    python scripts/record_clip.py --frames 150 --out /mnt/data/isaac/videos/clip.mp4
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record a defend-scenario clip.")
parser.add_argument("--frames", type=int, default=150, help="Frames to capture.")
parser.add_argument("--capture_every", type=int, default=2, help="Sim steps between captures.")
parser.add_argument("--ticks_per_capture", type=int, default=3, help="App ticks after each capture.")
parser.add_argument("--out", type=str, default="outputs/videos/defend_chase.mp4")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--frame_dir", type=str, default="outputs/frames_clip")
parser.add_argument("--pilot", type=str, default="chase", choices=["chase", "smart", "smart2", "zero", "policy"],
                    help="Who flies the defenders: a scripted baseline, or 'policy' with --checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="skrl checkpoint for --pilot policy (mean actions).")
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="env config overrides, e.g. respawn_waves=False capture_radius=1.0")
parser.add_argument("--eye", type=float, nargs=3, default=(13.0, 13.0, 8.0), help="camera position")
parser.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="camera target")
args, _ = parser.parse_known_args()

if sys.platform == "win32":
    import h5py  # noqa: E402, F401  Windows: load h5py's hdf5.dll before Kit's isaacsim.sensors.rtx loads an incompatible one
    # (Linux: importing h5py here drags in the venv numpy before Kit's prebundled one -> 'numpy.dtype size changed')
from isaaclab.app import AppLauncher  # noqa: E402

# livestream mode is not optional here. In plain headless mode Isaac Lab sets
# /isaaclab/render/active_viewport to false and leaves the viewport context unset, so
# there is nothing to capture -- captures silently produce no files. Turning livestream
# on keeps the viewport pipeline alive, which is what makes these captures work.
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from action_space_kit.tasks.direct.tello_defend.chase import TEACHERS  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

os.makedirs(args.frame_dir, exist_ok=True)
for stale in os.listdir(args.frame_dir):
    os.remove(os.path.join(args.frame_dir, stale))

env_cfg = parse_env_cfg("AS-Defend-v0", device="cuda:0", num_envs=1)
env_cfg.seed = 42
import ast  # noqa: E402

for item in args.set:
    key, value = item.split("=", 1)
    try:
        value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    if not hasattr(env_cfg, key):
        raise SystemExit(f"[clip] env config has no field {key!r}")
    setattr(env_cfg, key, value)
    print(f"[clip] override {key} = {value!r}", flush=True)
# frame the protected site so interceptions are visible
env_cfg.viewer.eye = tuple(args.eye)
env_cfg.viewer.lookat = tuple(args.lookat)

gym_env = gym.make("AS-Defend-v0", cfg=env_cfg)
env = gym_env.unwrapped

policy = None
if args.pilot == "policy":
    if not args.checkpoint:
        raise SystemExit("[clip] --pilot policy needs --checkpoint")
    from skrl.utils.runner.torch import Runner  # noqa: E402
    from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
    from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

    agent_cfg = load_cfg_from_registry("AS-Defend-v0", f"skrl_{args.algorithm.lower()}_cfg_entry_point")
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(SkrlVecEnvWrapper(gym_env, ml_framework="torch"), agent_cfg)
    runner.agent.load(args.checkpoint)
    if hasattr(runner.agent, "set_running_mode"):
        runner.agent.set_running_mode("eval")
    else:
        runner.agent.enable_training_mode(False)
    policy = runner.agent
    print(f"[clip] policy loaded from {args.checkpoint}", flush=True)

obs, _ = env.reset()

viewport = get_active_viewport()
print(f"[clip] viewport: {viewport}", flush=True)


def tick(n: int) -> None:
    """Tick the app n times for rendering/capture flushing WITHOUT advancing physics (see sim.render())."""
    env.sim.set_setting("/app/player/playSimulations", False)
    for _ in range(n):
        simulation_app.update()
    env.sim.set_setting("/app/player/playSimulations", True)

# let the renderer settle before the first capture
tick(30)


def capture(index: int) -> str:
    path = os.path.join(args.frame_dir, f"frame_{index:05d}.png")
    # wait_for_result is a coroutine, so there is nothing useful to call synchronously;
    # ticking the app is what actually flushes the capture to disk
    draw = getattr(env, "draw_markers", None)
    if draw is not None:
        draw()
    capture_viewport_to_file(viewport, path)
    tick(args.ticks_per_capture)
    return path


num_attackers = env.cfg.num_attackers
captured = 0
step = 0

print(f"[clip] capturing {args.frames} frames", flush=True)
while captured < args.frames:
    if policy is not None:
        with torch.inference_mode():
            states = {agent: None for agent in env.cfg.possible_agents}
            outputs = policy.act(obs, states, timestep=0, timesteps=0)
            actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.cfg.possible_agents}
    elif args.pilot == "zero":
        actions = {a: torch.zeros(env.num_envs, 4, device=env.device) for a in env.cfg.possible_agents}
    else:
        actions = TEACHERS[args.pilot](env, obs)

    prev_credit = env._episode_credit[0].clone()
    obs, _, _, _, _ = env.step(actions)
    env.sim.render()
    step += 1
    # ponytail: event log so the clip can be narrated -- sim time t, video time v (frames / fps).
    # credit (captures per defender) survives wave respawns and is latched into outcome_credit at episode end.
    t, v = step * env.step_dt, captured / args.fps
    ended = bool(env.outcome_valid[0])
    credit = env.outcome_credit[0] if ended else env._episode_credit[0]
    who = (credit - prev_credit > 0).nonzero().flatten().tolist()
    if who:
        print(f"[clip] t={t:6.2f}s v={v:5.1f}s CAPTURE by defender {who}", flush=True)
    if ended:
        kind = "BREACH" if env.outcome_breached[0] else ("CLEARED" if env.outcome_captures[0] >= num_attackers else "TIMEOUT")
        print(f"[clip] t={t:6.2f}s v={v:5.1f}s EPISODE END {kind} captures={int(env.outcome_captures[0])} "
              f"time={env.outcome_time[0]:.1f}s min_sep={env.outcome_min_sep[0]:.2f}m", flush=True)

    if step % args.capture_every == 0:
        path = capture(captured)
        captured += 1
        if captured == 1:
            # fail fast rather than simulating for minutes and writing nothing
            tick(60)
            if not os.path.exists(path):
                print("[clip] first capture produced no file -- aborting", flush=True)
                env.close()
                simulation_app.close()
                sys.exit(1)
            print(f"[clip] first frame OK ({os.path.getsize(path)} bytes)", flush=True)

# let the trailing captures flush
tick(120)

frames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".png"))
print(f"[clip] wrote {len(frames)} frames", flush=True)

if not frames:
    print("[clip] no frames captured", flush=True)
    sys.exit(1)

import imageio_ffmpeg  # noqa: E402

os.makedirs(os.path.dirname(args.out), exist_ok=True)
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(),
    "-y",
    "-framerate",
    str(args.fps),
    "-i",
    os.path.join(args.frame_dir, "frame_%05d.png"),
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-vf",
    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    args.out,
]
proc = subprocess.run(cmd, capture_output=True, text=True)
ok = proc.returncode == 0 and os.path.exists(args.out)
if ok:
    print(f"[clip] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
else:
    print(f"[clip] ffmpeg failed: {proc.stderr[-600:]}", flush=True)

# close last: simulation_app.close() exits the process, so nothing after it would run
env.close()
simulation_app.close()
sys.exit(0 if ok else 1)
