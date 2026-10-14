"""Record a video of AS-Tello-Room-v0, flown by a trained (central) SB3 policy or by zero sticks.

The core recorder for the room task: point it at any checkpoint (default: the newest one of
the newest run, so it works mid-training) and it writes an mp4 at real time, one frame per
rc command (74 ms). Like ``record_approach.py`` it captures Kit viewport frames (Replicator
camera sensors are broken here) through a dedicated wide camera, then stitches them with ffmpeg.

In frame, per drone one color shared by: its beacon (small sphere = true position), its
published fix (small cube = what core/positional.py reports; it freezes when the drone is
lost), its waypoint (big sphere, 1.5 m up, radius = waypoint_radius) and its landing spot
(disc on the floor). The panel at the bottom shows, per drone, its phase (to waypoint / at
waypoint / LANDING / landed on or off its spot / LOST), the rc it is sending and its tag-fix age;
the top line shows time, attempt and the team return so far. Each attempt ends on a 2 s hold
with its result.

    # the single-drone task (starts on the floor)
    python -u scripts/record_room.py --task AS-Tello-Room-Single-v0 --out outputs/videos/room_single.mp4
    # newest checkpoint of the newest run, 2 attempts
    python -u scripts/record_room.py --out outputs/videos/room_policy.mp4 > outputs/record_room.log 2>&1
    # a given checkpoint, nominal dynamics, top view
    python -u scripts/record_room.py --checkpoint logs/sb3/AS-Tello-Room-v0/<run>/model.zip --no_dr --view top
    # scene check without a policy: zero sticks (hover)
    python -u scripts/record_room.py --pilot zero --episodes 1 --max_seconds 10
    grep "\\[rec\\]" outputs/record_room.log
"""

import argparse
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Record an AS-Tello-Room-v0 video.")
parser.add_argument("--task", choices=["AS-Tello-Room-v0", "AS-Tello-Room-Single-v0"], default="AS-Tello-Room-v0")
parser.add_argument("--pilot", choices=["policy", "zero"], default="policy")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="SB3 model .zip (default: newest checkpoint of the newest run under logs/sb3/<task>).")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--max_seconds", type=float, default=None, help="Cap on each attempt (default: the episode length).")
parser.add_argument("--view", choices=["overview", "side", "top"], default="overview")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--no_dr", action="store_true", help="Nominal dynamics instead of randomized.")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--set", nargs="*", default=[], help="Extra env cfg overrides, key=python_literal.")
parser.add_argument("--label", type=str, default="", help="Text shown in the top line.")
parser.add_argument("--ticks_per_capture", type=int, default=3, help="App ticks after each capture.")
parser.add_argument("--out", type=str, default="outputs/videos/room.mp4")
parser.add_argument("--frame_dir", type=str, default="outputs/frames_room")
parser.add_argument("--keep_frames", action="store_true")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

# Livestream mode keeps the viewport pipeline alive in headless mode; without it captures
# silently produce no files (see record_clip.py).
app_launcher = AppLauncher({"headless": True, "enable_cameras": True, "livestream": 1})
simulation_app = app_launcher.app

import ast  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.vec_env import VecFrameStack  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from action_space_kit.rl.policy_export import export_policy, latest_checkpoint  # noqa: E402
from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

TASK = args.task
H_AP = 20.955  # camera horizontal aperture [mm]
# (eye, target, focal length [mm]) in the room frame: tag-10 wall at X = 0, start box at X ~6.5
VIEWS = {
    "overview": ((9.6, -3.4, 3.0), (3.2, 0.0, 0.6), 15.0),  # behind and above the start box
    "side": ((3.8, -7.5, 2.4), (3.6, 0.0, 0.8), 11.0),  # from the open side: the whole flight left to right
    "top": ((3.8, -0.4, 10.0), (3.8, 0.0, 0.0), 11.0),  # straight down: paths, crossings and spot accuracy
}
HOLD_S = 2.0


def setup_camera(u, stage):
    """A dedicated camera prim (the session layer resets the default viewport camera's focal length)."""
    from pxr import Gf, UsdGeom

    viewport = get_active_viewport()
    viewport.resolution = (args.width, args.height)
    eye, target, focal = VIEWS[args.view]
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    cam = UsdGeom.Camera.Define(stage, "/World/RecordCam")
    cam.GetHorizontalApertureAttr().Set(H_AP)
    cam.GetVerticalApertureAttr().Set(H_AP * args.height / args.width)
    cam.GetFocalLengthAttr().Set(float(focal))
    cam.GetClippingRangeAttr().Set((0.01, 100.0))
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    up = np.cross(r, f)
    eye_w = eye + u.scene.env_origins[0].cpu().numpy()
    rows = [list(r) + [0.0], list(up) + [0.0], list(-f) + [0.0], list(eye_w) + [1.0]]
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*[float(v) for row in rows for v in row]))
    viewport.camera_path = "/World/RecordCam"
    return viewport


def drone_state(u) -> list[dict]:
    """Env 0, per drone: what the panel shows."""
    pos = u._positions()[0]
    wp = (u._spots + u._wp_offset)[0]
    rows = []
    for i in range(u.cfg.num_drones):
        if bool(u.landed[0, i]):
            err = float(u.touchdown_err[0, i]) * 100
            phase = f"LANDED on spot ({err:.0f} cm)" if bool(u.land_ok[0, i]) else f"LANDED OFF spot ({err:.0f} cm)"
        elif bool(u.landing[0, i]):
            phase = "LANDING"
        elif bool(u.taking_off[0, i]):
            phase = "TAKEOFF (firmware)"
        elif bool(u.at_waypoint[0, i]):
            phase = "at waypoint"
        else:
            phase = f"to waypoint {float(torch.linalg.norm(wp[i] - pos[i])):.1f} m"
        rows.append({
            "phase": phase,
            "lost": bool(u.lost[0, i]),
            "rc": [int(v) for v in u.last_rc[0, i].tolist()],
            "fix_age": float(u.fix_age[0, i]),
        })
    return rows


def overlay(path: str, rec: dict, colors: list[tuple[int, int, int]], result: str | None) -> None:
    img = cv2.imread(path)
    if img is None:
        return
    font, W = cv2.FONT_HERSHEY_SIMPLEX, img.shape[1]

    def text(s, x, y, color=(255, 255, 255), scale=0.6, thick=1):
        cv2.putText(img, s, (x, y), font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, s, (x, y), font, scale, color, thick, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
    text(f"{rec['head']}   t = {rec['t']:5.1f} s   team return {rec['ret']:+8.0f}", 10, 23)
    n = len(rec["drones"])
    y0 = img.shape[0] - 12 - 26 * n
    cv2.rectangle(img, (0, y0 - 24), (760, img.shape[0]), (0, 0, 0), -1)
    for label, x in (("drone", 10), ("phase", 60), ("rc: right  fwd   up  yaw", 400), ("fix age", 640)):
        text(label, x, y0 - 6, (200, 200, 200), 0.5)
    for i, d in enumerate(rec["drones"]):
        y = y0 + 22 + 26 * i
        cv2.rectangle(img, (12, y - 14), (28, y + 2), colors[i], -1)
        phase = d["phase"] + ("   LOST: no tag" if d["lost"] else "")
        color = (80, 80, 255) if d["lost"] or "OFF" in d["phase"] else (120, 255, 120) if "on spot" in d["phase"] else (255, 255, 255)
        text(f"{i}", 36, y, colors[i], 0.55)
        text(phase, 60, y, color, 0.55)
        for k, v in enumerate(d["rc"]):
            text(f"{v:4d}", 430 + 50 * k, y, (255, 255, 255), 0.55)
        text(f"{d['fix_age']:4.1f} s", 640, y, (80, 80, 255) if d["lost"] else (255, 255, 255), 0.55)
    if result:
        (tw, th), _ = cv2.getTextSize(result, font, 1.0, 2)
        x, y = (W - tw) // 2, img.shape[0] // 2
        cv2.rectangle(img, (x - 16, y - th - 16), (x + tw + 16, y + 16), (0, 0, 0), -1)
        cv2.putText(img, result, (x, y), font, 1.0, (120, 255, 120) if result.startswith("SUCCESS") else (80, 160, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main() -> int:
    os.makedirs(args.frame_dir, exist_ok=True)
    for stale in os.listdir(args.frame_dir):
        os.remove(os.path.join(args.frame_dir, stale))

    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.sim.device = "cuda:0"
    cfg.debug_vis = True
    for kv in args.set:
        k, v = kv.split("=", 1)
        obj = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, ast.literal_eval(v))
    if args.no_dr:
        cfg.dynamics.domain_randomization = False
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    venv = Sb3CentralVecEnv(env)

    head = args.label
    if args.pilot == "policy":
        ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path("logs", "sb3", TASK))
        ckpt = ckpt.resolve()
        agent_cfg = load_cfg_from_registry(TASK, "sb3_cfg_entry_point")
        n_stack = agent_cfg.get("frame_stack", 1)
        policy = torch.jit.load(str(export_policy(ckpt, n_stack, os.path.join(args.frame_dir, "policy.pt"))), map_location="cuda:0")
        os.remove(os.path.join(args.frame_dir, "policy.pt"))
        if n_stack > 1:
            venv = VecFrameStack(venv, n_stack=n_stack)
        print(f"[rec] policy: {ckpt}", flush=True)
        head = head or f"{ckpt.parent.name}/{ckpt.stem}"

        def act(obs):
            with torch.inference_mode():
                return policy(torch.as_tensor(obs, dtype=torch.float32, device="cuda:0")).cpu().numpy()
    else:
        head = head or "zero sticks"

        def act(obs):
            return np.zeros((1, venv.action_space.shape[0]), dtype=np.float32)

    import omni.usd

    viewport = setup_camera(u, omni.usd.get_context().get_stage())
    obs = venv.reset()
    for _ in range(30):
        u.sim.render()  # renders without stepping physics

    colors = [tuple(int(255 * c) for c in reversed(rgb)) for rgb in u.DRONE_COLORS]  # BGR
    frame = 0
    records: dict[int, dict] = {}
    results: dict[int, str] = {}

    # Ticks go through sim.render(), which pauses physics around the app update (a bare
    # simulation_app.update() would step physics without the controller forces).
    def capture(rec):
        nonlocal frame
        capture_viewport_to_file(viewport, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
        records[frame] = rec
        for _ in range(args.ticks_per_capture):
            u.sim.render()
        frame += 1

    max_s = args.max_seconds or cfg.episode_length_s
    max_steps = int(max_s / u.step_dt)
    hold = int(HOLD_S / u.step_dt)
    summary = []
    for ep in range(args.episodes):
        ep_head = f"{head}   attempt {ep + 1}/{args.episodes}"
        steps, ret, done, rec = 0, 0.0, False, None
        while steps < max_steps:
            obs, rew, dones, _ = venv.step(act(obs))
            steps += 1
            ret += float(rew[0])
            if bool(dones[0]):
                done = True
                break  # the env has already auto-reset: this frame would show the next episode's start
            u.sim.render()
            rec = {"head": ep_head, "t": steps * u.step_dt, "ret": ret, "drones": drone_state(u)}
            capture(rec)
        o = {k: float(getattr(u, f"outcome_{k}")[0]) for k in ("success", "waypoints", "landed", "land_ok", "collision", "crash", "spot_err")}
        if done:
            verdict = "SUCCESS" if o["success"] else "FAIL"
            why = "  collision" if o["collision"] else "  crash" if o["crash"] else ""
            short = (f"{verdict}  waypoints {o['waypoints']:.0f}/{u.cfg.num_drones}  landed {o['landed']:.0f}/{u.cfg.num_drones}  "
                     f"on spot {o['land_ok']:.0f}/{u.cfg.num_drones}{why}")
        else:
            short = f"cut at {max_s:.0f} s"
            obs = venv.reset()
        line = f"episode {ep + 1}: {short}, {steps * u.step_dt:.1f} s, team return {ret:+.0f}"
        summary.append(line)
        print(f"[rec] {line}", flush=True)
        # hold the last frame with the result, so attempts are clearly separated
        if rec is not None:
            last = os.path.join(args.frame_dir, f"frame_{frame - 1:05d}.png")
            for _ in range(300):
                if os.path.exists(last):
                    break
                u.sim.render()  # the capture is async; let it land before copying
            for _ in range(hold):
                shutil.copyfile(last, os.path.join(args.frame_dir, f"frame_{frame:05d}.png"))
                records[frame], results[frame] = rec, short
                frame += 1

    for _ in range(120):
        u.sim.render()  # let the last async captures land
    frames = sorted(f for f in os.listdir(args.frame_dir) if f.endswith(".png"))
    print(f"[rec] wrote {len(frames)} frames", flush=True)
    if not frames:
        return 1
    for i, rec in records.items():
        overlay(os.path.join(args.frame_dir, f"frame_{i:05d}.png"), rec, colors, results.get(i))

    import imageio_ffmpeg

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fps = f"{1.0 / u.step_dt:.4f}"  # one frame per rc command: real time
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-framerate", fps,
        "-i", os.path.join(args.frame_dir, "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", args.out,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(args.out):
        print(f"[rec] ffmpeg failed: {proc.stderr[-600:]}", flush=True)
        return 1
    if not args.keep_frames:
        shutil.rmtree(args.frame_dir, ignore_errors=True)
    print(f"[rec] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        # Exit directly: with livestream on, simulation_app.close() can hang in shutdown
        # after the video is already written.
        sys.stdout.flush()
        os._exit(code)
