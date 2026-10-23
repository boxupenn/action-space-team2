# Copyright (c) 2026, Action Space. SPDX-License-Identifier: BSD-3-Clause
"""Greedy pursuit baseline for AS-Defend-v0: head straight at the nearest live attacker.

Shared by ``scripts/evaluate.py`` (``--baseline chase``) and ``scripts/bc_warm_start.py``
(the behaviour-cloning teacher).
"""

import torch


def chase_actions(raw, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-agent world-frame unit vector toward the nearest live attacker, zero yaw rate."""
    actions = {}
    n = raw.cfg.num_attackers
    for agent in raw.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * n].reshape(raw.num_envs, n, 3)
        # the env zeroes a downed attacker's relative position and clears its alive flag (the last
        # n entries); without masking, argmin would "chase" the downed one and hover
        alive = obs[agent][:, 9 + 6 * n : 9 + 7 * n] > 0.5
        distance = torch.where(alive, torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], float("inf")))
        target = rel[torch.arange(raw.num_envs, device=raw.device), distance.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        direction = direction * alive.any(dim=-1, keepdim=True).float()  # nothing alive: hold position
        actions[agent] = torch.cat([direction, torch.zeros(raw.num_envs, 1, device=raw.device)], dim=-1)
    return actions
