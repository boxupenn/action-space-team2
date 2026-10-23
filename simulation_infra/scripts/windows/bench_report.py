"""Summarise outputs/bench (written by scripts/windows/bench.ps1) as a Markdown table.

Reads timeline.tsv plus every step's .log/.err and extracts, per step: wall time, exit code,
Isaac Sim startup time, training time, seconds per PPO iteration, and the step's key result
line (play score, evaluate summary, recorder/export output). Standard library only.

    python scripts/windows/bench_report.py                 # print the table
    python scripts/windows/bench_report.py --out outputs/bench/report.md
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BENCH = ROOT / "outputs" / "bench"

# skrl PPO rollouts per iteration (agent.rollouts in each skrl_*_cfg.yaml)
SKRL_ROLLOUTS = {"AS-Tello-Hover-v0": 24, "AS-Tello-Waypoint-v0": 24, "AS-Defend-v0": 16}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def last(pattern: str, text: str, flags: int = re.M) -> str | None:
    hits = re.findall(pattern, text, flags)
    return hits[-1] if hits else None


def arg_of(cmd: str, flag: str) -> str | None:
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmd)
    return m.group(1) if m else None


def tb_scalar_first_last(run_dir: Path, tag: str) -> tuple[float, float] | None:
    """First and last value of a TensorBoard scalar in run_dir (needs the tensorboard package: run with the Isaac venv)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    if not run_dir.exists():
        return None
    try:
        acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            return None
        vals = [e.value for e in acc.Scalars(tag)]
        return (vals[0], vals[-1]) if vals else None
    except Exception:
        return None


def summarise(step: str, cmd: str, wall: float, log: str, err: str) -> dict:
    out: dict = {"startup_s": None, "train_s": None, "iter_s": None, "iters": None, "result": ""}
    task = arg_of(cmd, "--task") or ""
    num_envs = arg_of(cmd, "--num_envs")
    iters = arg_of(cmd, "--max_iterations")

    train_s = last(r"Training time:\s*([\d.]+)\s*seconds", log)
    if train_s:
        out["train_s"] = float(train_s)
        out["startup_s"] = round(wall - float(train_s), 1)
    if iters:
        out["iters"] = int(iters)

    if "skrl\\train.py" in cmd or "skrl/train.py" in cmd:
        # tqdm's final rate is timesteps/s; one iteration = `rollouts` timesteps + the update
        # tqdm's trailing "N/M [..., R it/s]" samples; the very last one is the whole-run average (includes
        # warm-up), so take the median of the preceding samples as the steady-state rate
        rates = [float(r) for r in re.findall(r"([\d.]+)it/s\]", log + "\n" + err)]  # the kit's train.py bars go to stdout
        window = rates[-21:-1] if len(rates) > 5 else rates
        steady = sorted(window)[len(window) // 2] if window else 0.0
        rollouts = SKRL_ROLLOUTS.get(task, 24)
        if steady > 0:
            out["iter_s"] = round(rollouts / steady, 2)
        elif out["train_s"] and iters:
            out["iter_s"] = round(out["train_s"] / int(iters), 2)
        avg = f", whole-run avg {rates[-1]:.1f} timesteps/s" if rates else ""
        out["result"] = f"{num_envs} envs, {iters} iters, steady {steady:.1f} timesteps/s{avg}"
        # run dir = <log root>/<run name prefix>_<experiment_name>; the log prints the two halves separately
        log_root = last(r"Logging experiment in directory:\s*(.+)", log)
        run_name = last(r"Exact experiment name requested from command line:\s*(.+)", log)
        if log_root and run_name:
            matches = sorted(Path(log_root.strip()).glob(run_name.strip() + "*"))
            rew = tb_scalar_first_last(matches[-1], "Reward / Total reward (mean)") if matches else None
            if rew:
                out["result"] += f", total reward (mean) {rew[0]:.1f} -> {rew[1]:.1f}"
    elif "sb3\\train.py" in cmd or "sb3/train.py" in cmd:
        # SB3's last stats table: time_elapsed / iterations (training only, no startup)
        it = last(r"\|\s*iterations\s*\|\s*(\d+)", log)
        el = last(r"\|\s*time_elapsed\s*\|\s*(\d+)", log)
        fps = last(r"\|\s*fps\s*\|\s*(\d+)", log)
        rews = re.findall(r"\|\s*ep_rew_mean\s*\|\s*(-?[\d.e+-]+)", log)
        if it and el and int(it) > 0:
            out["iter_s"] = round(int(el) / int(it), 2)
        elif out["train_s"] and iters:
            out["iter_s"] = round(out["train_s"] / int(iters), 2)
        out["result"] = f"{num_envs} envs, {iters} iters" + (f", {fps} transitions/s" if fps else "")
        if rews:
            out["result"] += f", ep_rew_mean {rews[0]} -> {rews[-1]}"
    elif "evaluate.py" in cmd:
        inter = last(r"interception rate:\s*(.+)", log)
        breach = last(r"breach rate:\s*(.+)", log)
        tti = last(r"time to intercept:\s*(.+)", log)
        out["result"] = "; ".join(x.strip() for x in (f"intercept {inter}" if inter else "", f"breach {breach}" if breach else "", f"tti {tti}" if tti else "") if x)
        vid = last(r"(?:Saving video to|Moviepy - Done|wrote)\s*(.+\.mp4)", log)
        if vid:
            out["result"] += f"; video {vid.strip()}"
    elif "sb3\\play.py" in cmd or "sb3/play.py" in cmd:
        out["result"] = (last(r"(\[\d+ episodes\].*)", log) or "").strip()
    elif "skrl\\play.py" in cmd or "skrl/play.py" in cmd:
        stop = last(r"(\[play\] stopping after \d+ steps)", log)
        ret = last(r"(\[play\] \d+ episodes finished: .*)", log)
        vid = ""
        folder = last(r"video_folder:\s*(.+)", log)  # gym's RecordVideo wrapper prints nothing when it writes
        if folder and Path(folder.strip()).exists():
            mp4s = sorted(Path(folder.strip()).glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            if mp4s:
                vid = f"video {mp4s[-1]} ({mp4s[-1].stat().st_size} bytes)"
        out["result"] = "; ".join(x for x in (stop or "", ret or "", vid) if x).strip()
    elif "record_" in cmd:
        ep = last(r"(\[rec\] episode \d+: .*)", log)
        wrote = last(r"(\[(?:rec|clip)\] wrote .*)", log)
        out["result"] = " | ".join(x.strip() for x in (ep or "", wrote or "") if x)
    elif "export_sb3_policy.py" in cmd:
        out["result"] = (last(r"(\[export\] wrote .*)", log) or "").strip()
    elif "check_room_env.py" in cmd:
        ok = len(re.findall(r"^\s*\[ok\]", log, re.M))
        fail = len(re.findall(r"^\s*\[FAIL\]", log, re.M))
        verdict = last(r"(ALL CHECKS PASSED|CHECKS FAILED.*)", log) or ""
        out["result"] = f"{ok} ok, {fail} FAIL {verdict}".strip()

    tb = last(r"(Traceback \(most recent call last\):[\s\S]{0,600})", log + "\n" + err)
    if tb and not out["result"]:
        tail = [ln for ln in tb.strip().splitlines() if ln.strip()][-1]
        out["result"] = f"ERROR: {tail[:160]}"
    elif tb:
        tail = [ln for ln in tb.strip().splitlines() if ln.strip()][-1]
        out["result"] += f"  [traceback: {tail[:120]}]"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default=None, help="Also write the Markdown table here.")
    args = parser.parse_args()

    timeline = BENCH / "timeline.tsv"
    rows = list(csv.DictReader(read_text(timeline).lstrip("﻿").splitlines(), delimiter="\t"))
    lines = ["| step | start | wall s | exit | startup s | train s | s / iter | result |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        step = r["step"]
        wall = float(r["wall_s"] or 0)
        s = summarise(step, r["command"], wall, read_text(BENCH / f"{step}.log"), read_text(BENCH / f"{step}.err"))
        fmt = lambda v: "" if v is None else str(v)
        lines.append(f"| {step} | {r['start'][11:]} | {int(wall)} | {r['exit']} | {fmt(s['startup_s'])} | {fmt(s['train_s'])} | {fmt(s['iter_s'])} | {s['result'].replace('|', '/')} |")
    table = "\n".join(lines)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
