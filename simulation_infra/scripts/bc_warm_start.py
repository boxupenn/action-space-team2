# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Behaviour-cloning warm start for the Defend task: teach the IPPO policy to chase before RL.

Plain IPPO on ``AS-Defend-v0`` learns to engage only in about half of the seeds (handbook
section 11): whether the first few iterations stumble onto "move toward the attacker" decides
the run. The greedy ``chase`` baseline intercepts 100% at the easy settings, so we regress
the policy mean onto its actions first and hand the result to ``skrl/train.py --checkpoint``.

    python -u scripts/bc_warm_start.py --headless --num_envs 512 --steps 200 \
        env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4] \
        --out outputs/exports/bc_defend_easy.pt
    python -u scripts/evaluate.py --task AS-Defend-v0 --headless --checkpoint outputs/exports/bc_defend_easy.pt ...
    python -u scripts/skrl/train.py --task AS-Defend-v0 --algorithm IPPO --headless --checkpoint outputs/exports/bc_defend_easy.pt ...

The checkpoint is a normal skrl IPPO save (policy, value, optimizer, preprocessors per agent):
the observation scaler is fitted on the collected data, the policy mean on the chase actions,
the value head is left untrained (PPO fits it in the first iterations) and the policy std is
set to ``--log_std`` so exploration noise does not drown the cloned mean.

Data is collected DART-style: the executed action is chase + Gaussian noise (``--noise``), the
label is the clean chase action, so the policy also sees states slightly off the teacher's path.
"""

import argparse
import ast
import os
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Behaviour-cloning warm start from the chase baseline.")
parser.add_argument("--task", type=str, default="AS-Defend-v0")
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=200, help="Env steps to collect (samples = steps * num_envs per agent).")
parser.add_argument("--noise", type=float, default=0.3, help="Std of the Gaussian noise added to the executed teacher action.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--log_std", type=float, default=-1.0, help="Policy log std after cloning (-1.0 -> std 0.37).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, default="outputs/exports/bc_defend.pt")
AppLauncher.add_app_launcher_args(parser)
# extra `env.<field>=<literal>` / `agent.<key>=<literal>` arguments, same as evaluate.py / train.py
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

import action_space_kit.tasks  # noqa: F401
from action_space_kit.tasks.direct.tello_defend.chase import chase_actions
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def _literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_env_overrides(env_cfg):
    for _arg in env_overrides:
        key, value = _arg[len("env."):].split("=", 1)
        target, parts = env_cfg, key.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise SystemExit(f"[bc] env config has no field {key!r}")
        setattr(target, parts[-1], _literal(value))
        print(f"[bc] override env.{key} = {_literal(value)!r}", flush=True)


def apply_agent_overrides(agent_cfg):
    for _arg in agent_overrides:
        key, value = _arg[len("agent."):].split("=", 1)
        node, parts = agent_cfg, key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise SystemExit(f"[bc] agent config has no key {key!r}")
        node[parts[-1]] = _literal(value)
        print(f"[bc] override agent.{key} = {_literal(value)!r}", flush=True)


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    apply_env_overrides(env_cfg)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped

    agent_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{args_cli.algorithm.lower()}_cfg_entry_point")
    apply_agent_overrides(agent_cfg)
    agent_cfg["seed"] = args_cli.seed
    # no TensorBoard run / periodic checkpoints for this script; the only output is --out
    agent_cfg["agent"]["experiment"]["directory"] = os.path.join("outputs", "bc_logs")
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    agent = Runner(SkrlVecEnvWrapper(env, ml_framework="torch"), agent_cfg).agent

    # ---- collect (obs, chase action) pairs -------------------------------------------------
    obs, _ = raw.reset()
    buffers = {uid: ([], []) for uid in raw.cfg.possible_agents}
    for _ in range(args_cli.steps):
        teacher = chase_actions(raw, obs)
        for uid in raw.cfg.possible_agents:
            buffers[uid][0].append(obs[uid].clone())
            buffers[uid][1].append(teacher[uid].clone())
        executed = {
            uid: (teacher[uid] + args_cli.noise * torch.randn_like(teacher[uid])).clamp(-1.0, 1.0)
            for uid in raw.cfg.possible_agents
        }
        obs, _, _, _, _ = raw.step(executed)
    n = args_cli.steps * raw.num_envs
    print(f"[bc] collected {n} samples per agent ({args_cli.steps} steps x {raw.num_envs} envs)", flush=True)

    # ---- fit scaler + policy mean per agent -------------------------------------------------
    for uid in raw.cfg.possible_agents:
        X = torch.cat(buffers[uid][0]); Y = torch.cat(buffers[uid][1])
        prep = agent._observation_preprocessor[uid]
        if hasattr(prep, "running_mean"):
            prep(X, train=True)  # one-shot running statistics over the whole dataset
        Xn = prep(X)
        policy = agent.policies[uid]
        opt = torch.optim.Adam(policy.parameters(), lr=args_cli.lr)
        holdout = max(1, n // 10)
        Xtr, Ytr, Xho, Yho = Xn[holdout:], Y[holdout:], Xn[:holdout], Y[:holdout]
        first = last = None
        for epoch in range(args_cli.epochs):
            perm = torch.randperm(len(Xtr), device=Xtr.device)
            total = 0.0
            for i in range(0, len(perm), args_cli.batch):
                idx = perm[i : i + args_cli.batch]
                _, out = policy.act({"observations": Xtr[idx]}, role="policy")
                loss = torch.nn.functional.mse_loss(out["mean_actions"], Ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            last = total / len(Xtr)
            first = last if first is None else first
        with torch.no_grad():
            _, out = policy.act({"observations": Xho}, role="policy")
            mean = out["mean_actions"]
            ho_mse = torch.nn.functional.mse_loss(mean, Yho).item()
            moving = Yho[:, :3].norm(dim=-1) > 0.5  # teacher is moving (an attacker is alive)
            cos = torch.nn.functional.cosine_similarity(mean[moving, :3], Yho[moving, :3], dim=-1).mean().item()
            policy.log_std_parameter.fill_(args_cli.log_std)
        print(
            f"[bc] {uid}: train mse {first:.4f} -> {last:.4f}, holdout mse {ho_mse:.4f}, "
            f"direction cosine {cos:.3f} (n={int(moving.sum())}), log_std set to {args_cli.log_std}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    agent.save(args_cli.out)
    print(f"[bc] wrote {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
