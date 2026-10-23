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
the value head (shared trunk, ``models.separate: False``) on Monte Carlo returns of the noisy
teacher rollouts through the value preprocessor, and the policy std is set to ``--log_std`` so
exploration noise does not drown the cloned mean. Fitting the critic matters: with an untrained
value head PPO's first updates push garbage gradients through the shared trunk and the cloned
policy is destroyed within ~100 iterations (reward +132 -> -38 in the first attempt).

Note: ``skrl/train.py --checkpoint`` runs are bit-identical across ``--seed`` values (verified
2026-10-23 for any checkpoint), so vary ``--seed`` HERE to get different starting points.

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
parser.add_argument(
    "--teacher_scale", type=float, default=1.0,
    help="Scale of the chase unit vector (label and executed action). <1 keeps the cloned mean off the +-1 clip "
    "boundary: skrl computes log_prob on the CLIPPED sample, so a mean sitting on the boundary gives PPO a biased gradient.",
)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--log_std", type=float, default=-1.0, help="Policy log std after cloning (-1.0 -> std 0.37).")
parser.add_argument("--value_coef", type=float, default=0.5, help="Weight of the value-head loss in the joint BC loss.")
parser.add_argument("--drop_tail", type=int, default=50, help="Trailing steps without a return bootstrap, dropped from the datasets.")
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

    # ---- collect (obs, chase action, reward, done) per step ----------------------------------
    obs, _ = raw.reset()
    buffers = {uid: ([], [], []) for uid in raw.cfg.possible_agents}  # obs, teacher action, reward
    dones = []
    for _ in range(args_cli.steps):
        teacher = {uid: a * args_cli.teacher_scale for uid, a in chase_actions(raw, obs).items()}
        for uid in raw.cfg.possible_agents:
            buffers[uid][0].append(obs[uid].clone())
            buffers[uid][1].append(teacher[uid].clone())
        executed = {
            uid: (teacher[uid] + args_cli.noise * torch.randn_like(teacher[uid])).clamp(-1.0, 1.0)
            for uid in raw.cfg.possible_agents
        }
        obs, rew, term, trunc, _ = raw.step(executed)
        for uid in raw.cfg.possible_agents:
            buffers[uid][2].append(rew[uid].clone().flatten())
        a0 = raw.cfg.possible_agents[0]
        dones.append((term[a0] | trunc[a0]).float().flatten())
    print(f"[bc] collected {args_cli.steps * raw.num_envs} samples per agent ({args_cli.steps} steps x {raw.num_envs} envs)", flush=True)

    # Monte Carlo returns along each env's trajectory; the last --drop_tail steps have no
    # bootstrap and are dropped from every dataset (policy, value) to keep them aligned.
    gamma = float(agent_cfg["agent"]["discount_factor"])
    keep = args_cli.steps - args_cli.drop_tail
    done_t = torch.stack(dones)  # (steps, envs)

    # ---- fit scaler + policy mean + value head per agent (shared trunk: one joint loss) ----
    for uid in raw.cfg.possible_agents:
        rew_t = torch.stack(buffers[uid][2])  # (steps, envs)
        ret_t = torch.zeros_like(rew_t)
        running = torch.zeros_like(rew_t[0])
        for t in reversed(range(args_cli.steps)):
            running = rew_t[t] + gamma * (1.0 - done_t[t]) * running
            ret_t[t] = running
        X = torch.cat(buffers[uid][0][:keep]); Y = torch.cat(buffers[uid][1][:keep])
        R = ret_t[:keep].reshape(-1, 1)
        n = len(X)
        prep = agent._observation_preprocessor[uid]
        if hasattr(prep, "running_mean"):
            prep(X, train=True)  # one-shot running statistics over the whole dataset
        Xn = prep(X)
        fit_value = args_cli.value_coef > 0  # MAPPO's critic reads the global state, not the observation: policy-only there
        vprep = agent._value_preprocessor[uid]
        if fit_value and hasattr(vprep, "running_mean"):
            vprep(R, train=True)
        Rn = vprep(R)  # PPO regresses the value head onto preprocessed returns; mirror that
        policy, value = agent.policies[uid], agent.values[uid]
        params = list(policy.parameters())
        if fit_value:
            params += [p for p in value.parameters() if all(p is not q for q in policy.parameters())]
        opt = torch.optim.Adam(params, lr=args_cli.lr)
        holdout = max(1, n // 10)
        tr = slice(holdout, n); ho = slice(0, holdout)
        first = last = None
        for epoch in range(args_cli.epochs):
            perm = torch.randperm(n - holdout, device=Xn.device) + holdout
            total = 0.0
            for i in range(0, len(perm), args_cli.batch):
                idx = perm[i : i + args_cli.batch]
                _, out = policy.act({"observations": Xn[idx]}, role="policy")
                loss_pi = torch.nn.functional.mse_loss(out["mean_actions"], Y[idx])
                loss = loss_pi
                if fit_value:
                    v, _ = value.act({"observations": Xn[idx]}, role="value")
                    loss = loss + args_cli.value_coef * torch.nn.functional.mse_loss(v, Rn[idx])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss_pi.item() * len(idx)
            last = total / (n - holdout)
            first = last if first is None else first
        with torch.no_grad():
            _, out = policy.act({"observations": Xn[ho]}, role="policy")
            mean = out["mean_actions"]
            ho_mse = torch.nn.functional.mse_loss(mean, Y[ho]).item()
            moving = Y[ho][:, :3].norm(dim=-1) > 0.5 * args_cli.teacher_scale  # teacher is moving (an attacker is alive)
            cos = torch.nn.functional.cosine_similarity(mean[moving, :3], Y[ho][moving, :3], dim=-1).mean().item()
            v_ev = float("nan")
            if fit_value:
                v, _ = value.act({"observations": Xn[ho]}, role="value")
                v_ev = 1.0 - (v - Rn[ho]).var().item() / max(Rn[ho].var().item(), 1e-8)  # explained variance
            policy.log_std_parameter.fill_(args_cli.log_std)
        print(
            f"[bc] {uid}: policy mse {first:.4f} -> {last:.4f} (holdout {ho_mse:.4f}), direction cosine {cos:.3f} "
            f"(n={int(moving.sum())}), value explained var {v_ev:.3f}, returns mean {R.mean().item():.1f} "
            f"std {R.std().item():.1f}, log_std set to {args_cli.log_std}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    agent.save(args_cli.out)
    print(f"[bc] wrote {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
