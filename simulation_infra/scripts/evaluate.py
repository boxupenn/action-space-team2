# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Measure how well a defender policy actually defends.

Training curves say little about the thing that matters on day two: how often attackers
are stopped before they reach the site. This script runs whole episodes and reports
interception rate, breach rate and time-to-intercept, for a trained checkpoint or for
either built-in baseline.

    python scripts/evaluate.py --task AS-Defend-v0 --baseline chase --episodes 20
    python scripts/evaluate.py --task AS-Defend-v0 --checkpoint logs/skrl/.../best_agent.pt

Always evaluate a policy against ``--baseline chase`` before trusting it: a policy that
cannot beat greedy pursuit is not yet worth flying.
"""

import argparse
import ast
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a defender policy.")
parser.add_argument("--task", type=str, default="AS-Defend-v0", help="Task to evaluate.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--episodes", type=int, default=20, help="Episodes to finish before reporting.")
parser.add_argument("--checkpoint", type=str, default=None, help="Trained skrl checkpoint to load.")
parser.add_argument(
    "--baseline",
    type=str,
    default=None,
    choices=["zero", "chase", "smart", "smart2"],
    help="Evaluate a scripted baseline instead of a checkpoint: zero does nothing, chase pursues the nearest attacker, smart assigns targets one-to-one and leads them.",
)
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"], help="Algorithm of the checkpoint.")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions like training does instead of using the mean.")
parser.add_argument("--diag", type=int, default=0, help="Print up to N breached episodes: attacker azimuth gap, speeds, captures, time.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the run.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
parser.add_argument("--video_dir", type=str, default="videos/evaluate", help="Where to write the video.")
AppLauncher.add_app_launcher_args(parser)
# extra `env.<field>=<python literal>` arguments override the env config (e.g. env.capture_radius=1.0);
# `agent.<key>=<value>` ones override the skrl agent yaml the checkpoint was trained with (same paths as
# train.py's Hydra overrides, e.g. agent.models.policy.output='tanh(ACTIONS)'), so the model is rebuilt to match
args_cli, _overrides = parser.parse_known_args()
env_overrides, agent_overrides = [], []
for _arg in _overrides:
    if re.fullmatch(r"env\.[A-Za-z_][A-Za-z0-9_.]*=.+", _arg):
        env_overrides.append(_arg)
    elif re.fullmatch(r"agent\.[A-Za-z_][A-Za-z0-9_.]*=.+", _arg):
        agent_overrides.append(_arg)
    else:
        parser.error(f"unrecognized argument: {_arg}")

args_cli.headless = True
if args_cli.video:
    args_cli.enable_cameras = True  # rendering is needed to capture frames

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import action_space_kit.tasks  # noqa: F401
from action_space_kit.tasks.direct.tello_defend.chase import TEACHERS
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    for _arg in env_overrides:
        key, value = _arg[len("env."):].split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # keep it as a string
        target = env_cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise SystemExit(f"[evaluate] env config has no field {key!r}")
        setattr(target, parts[-1], value)
        print(f"[evaluate] override env.{key} = {value!r}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        import os

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.abspath(args_cli.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[evaluate] recording {args_cli.video_length} steps to {args_cli.video_dir}", flush=True)
    env = env.unwrapped if not args_cli.video else env

    raw = env.unwrapped  # RecordVideo wraps the env; outcome buffers live on the raw one

    policy = None
    if args_cli.checkpoint:
        from skrl.utils.runner.torch import Runner
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry

        entry = f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point"
        agent_cfg = load_cfg_from_registry(args_cli.task, entry)
        for _arg in agent_overrides:
            key, value = _arg[len("agent."):].split("=", 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # keep it as a string
            node = agent_cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[part]
            if parts[-1] not in node:
                raise SystemExit(f"[evaluate] agent config has no key {key!r}")
            node[parts[-1]] = value
            print(f"[evaluate] override agent.{key} = {value!r}", flush=True)
        # the wrapper takes no algorithm argument; multi-agent envs stay multi-agent
        wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
        runner = Runner(wrapped, agent_cfg)
        runner.agent.load(args_cli.checkpoint)
        if hasattr(runner.agent, "set_running_mode"):
            runner.agent.set_running_mode("eval")
        else:
            runner.agent.enable_training_mode(False)
        policy = runner.agent

    obs, _ = env.reset()
    episodes = breaches = 0
    intercepted = attacker_slots = 0
    intercept_times: list[float] = []
    alive_prev = raw._attacker_alive.clone()
    step_in_episode = torch.zeros(raw.num_envs, device=raw.device)
    episode_intercept_times: list[list[float]] = [[] for _ in range(raw.num_envs)]
    ep_return = {a: torch.zeros(raw.num_envs, device=raw.device) for a in raw.cfg.possible_agents}
    credit = torch.zeros(raw.cfg.num_defenders, device=raw.device)
    min_seps: list[float] = []
    diag_printed = 0
    returns: dict[str, list[float]] = {a: [] for a in raw.cfg.possible_agents}
    episode_lengths: list[float] = []
    action_norm_sum, action_steps = 0.0, 0

    while episodes < args_cli.episodes:
        if policy is not None:
            with torch.inference_mode():
                # multi-agent skrl agents take and return dicts keyed by agent, and
                # they index states[agent_id], so the states argument must be a dict too
                states = {agent: None for agent in raw.cfg.possible_agents}
                outputs = policy.act(obs, states, timestep=0, timesteps=0)
                actions = {
                    a: outputs[0][a] if args_cli.stochastic else outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in raw.cfg.possible_agents
                }
        elif args_cli.baseline in TEACHERS:
            actions = TEACHERS[args_cli.baseline](raw, obs)
        else:
            actions = {a: torch.zeros(raw.num_envs, 4, device=raw.device) for a in raw.cfg.possible_agents}

        obs, rew, terminated, truncated, _ = env.step(actions)
        step_in_episode += 1
        for a in raw.cfg.possible_agents:
            ep_return[a] += rew[a].flatten()
        action_norm_sum += sum(actions[a][:, :3].norm(dim=-1).mean().item() for a in raw.cfg.possible_agents) / len(raw.cfg.possible_agents)
        action_steps += 1

        newly_down = alive_prev & ~raw._attacker_alive
        if newly_down.any():
            for row in newly_down.any(dim=-1).nonzero(as_tuple=False).flatten().tolist():
                episode_intercept_times[row].append(step_in_episode[row].item() * raw.step_dt)
        alive_prev = raw._attacker_alive.clone()

        # outcomes are latched by the env before its auto-reset clears them
        finished = raw.outcome_valid.nonzero(as_tuple=False).flatten()
        if len(finished):
            episodes += len(finished)
            breaches += int(raw.outcome_breached[finished].sum().item())
            intercepted += int(raw.outcome_captures[finished].sum().item())
            credit += raw.outcome_credit[finished].sum(dim=0)
            min_seps += raw.outcome_min_sep[finished].tolist()
            if args_cli.diag:
                for row in finished.tolist():
                    if raw.outcome_breached[row] and diag_printed < args_cli.diag:
                        ang = raw.outcome_spawn_angle[row]
                        gap = torch.remainder(ang[0] - ang[1] + 3.14159, 6.28318) - 3.14159
                        print("[diag] breach: az gap %5.0f deg, speeds %s m/s, captures %d, t=%.1f s, min sep %.2f" % (
                            abs(gap.item()) * 57.3, [round(x, 2) for x in raw.outcome_attacker_speed[row].tolist()],
                            int(raw.outcome_captures[row].item()), raw.outcome_time[row].item(), raw.outcome_min_sep[row].item()), flush=True)
                        diag_printed += 1
            attacker_slots += len(finished) * raw.cfg.num_attackers
            for row in finished.tolist():
                intercept_times += episode_intercept_times[row]
                episode_intercept_times[row] = []
            for a in raw.cfg.possible_agents:
                returns[a] += ep_return[a][finished].tolist()
                ep_return[a][finished] = 0.0
            episode_lengths += (step_in_episode[finished] * raw.step_dt).tolist()
            step_in_episode[finished] = 0.0

    label = args_cli.checkpoint or f"baseline:{args_cli.baseline or zero}"
    print("\n=== %s on %s ===" % (label, args_cli.task), flush=True)
    print("episodes:            %d" % episodes, flush=True)
    print("interception rate:   %.1f%% of attackers" % (100.0 * intercepted / max(attacker_slots, 1)), flush=True)
    print("breach rate:         %.1f%% of episodes" % (100.0 * breaches / max(episodes, 1)), flush=True)
    for a in raw.cfg.possible_agents:
        print("return %-11s %.1f (mean of %d episodes)" % (a + ":", sum(returns[a]) / max(len(returns[a]), 1), len(returns[a])), flush=True)
    print("captures/episode:    " + "  ".join("%s %.2f" % (a, credit[i].item() / max(episodes, 1)) for i, a in enumerate(raw.cfg.possible_agents)), flush=True)
    if min_seps:
        finite = [m for m in min_seps if m < 1.0e3]
        coll = sum(m < raw.cfg.defender_collision_radius for m in finite)
        srt = sorted(finite)
        print("defender separation: min %.2f m, p5 %.2f m, median %.2f m; collisions (<%.2f m) %.1f%% of episodes" % (
            srt[0], srt[len(srt) // 20], srt[len(srt) // 2], raw.cfg.defender_collision_radius, 100.0 * coll / max(len(finite), 1)), flush=True)
    print("episode length:      %.1f s" % (sum(episode_lengths) / max(len(episode_lengths), 1)), flush=True)
    print("mean |v cmd|:        %.2f" % (action_norm_sum / max(action_steps, 1)), flush=True)
    if intercept_times:
        mean_t = sum(intercept_times) / len(intercept_times)
        print("time to intercept:   %.1f s (mean of %d)" % (mean_t, len(intercept_times)), flush=True)
    else:
        print("time to intercept:   n/a (nothing intercepted)", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
