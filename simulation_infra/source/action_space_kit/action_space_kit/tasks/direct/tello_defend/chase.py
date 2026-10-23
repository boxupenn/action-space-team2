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


def smart_actions(raw, obs: dict[str, torch.Tensor], v_max: float = 1.0) -> dict[str, torch.Tensor]:
    """Smarter scripted defender: one-to-one target assignment plus lead pursuit.

    Greedy chase sends both defenders after the same nearest attacker and flies at where the
    attacker *is*. This picks the assignment with the smallest summed distance (dead attackers
    cost a lot, so they are avoided) and aims at the intercept point, solving
    |r + v_a t| = v_max t for the time t (pure pursuit if no solution). Attacker world velocity
    is rebuilt from the observation: rel_vel + own velocity. action = unit vector * 1 (1 m/s).
    """
    import itertools

    n, E, dev = raw.cfg.num_attackers, raw.num_envs, raw.device
    agents = raw.cfg.possible_agents
    rel, att_vel, dist = {}, {}, {}
    alive = obs[agents[0]][:, 9 + 6 * n : 9 + 7 * n] > 0.5  # (E, n), same for every defender
    for a in agents:
        rel[a] = obs[a][:, 9 : 9 + 3 * n].reshape(E, n, 3)
        att_vel[a] = obs[a][:, 9 + 3 * n : 9 + 6 * n].reshape(E, n, 3) + obs[a][:, 3:6].unsqueeze(1)
        dist[a] = torch.where(alive, torch.linalg.norm(rel[a], dim=-1), torch.full((E, n), 1.0e3, device=dev))
    perms = torch.tensor(list(itertools.permutations(range(n), min(len(agents), n))), device=dev)  # (P, D)
    cost = torch.stack([dist[a][:, perms[:, i]] for i, a in enumerate(agents) if i < perms.shape[1]], dim=-1).sum(-1)
    best = perms[cost.argmin(dim=-1)]  # (E, D)
    actions = {}
    rows = torch.arange(E, device=dev)
    for i, a in enumerate(agents):
        if i < best.shape[1]:
            target = best[:, i]
            target = torch.where(alive[rows, target], target, dist[a].argmin(dim=-1))
        else:  # more defenders than attackers: extra ones take the nearest
            target = dist[a].argmin(dim=-1)
        r = rel[a][rows, target]
        va = att_vel[a][rows, target]
        qa = (va * va).sum(-1) - v_max**2
        qb = 2.0 * (r * va).sum(-1)
        qc = (r * r).sum(-1)
        disc = qb * qb - 4.0 * qa * qc
        t_pursuit = r.norm(dim=-1) / v_max
        # qa < 0 whenever the attacker is slower than us, so the quadratic has one positive root
        t_lead = (-qb - torch.sqrt(disc.clamp(min=0.0))) / (2.0 * qa.clamp(max=-1e-6))
        t = torch.where((disc >= 0) & (t_lead > 0), t_lead, t_pursuit)
        aim = r + va * t.unsqueeze(-1)
        direction = aim / torch.clamp(aim.norm(dim=-1, keepdim=True), min=1e-6)
        direction = direction * alive.any(dim=-1, keepdim=True).float()
        actions[a] = torch.cat([direction, torch.zeros(E, 1, device=dev)], dim=-1)
    return actions


def smart2_actions(raw, obs: dict[str, torch.Tensor], v_max: float = 1.0) -> dict[str, torch.Tensor]:
    """Like ``smart_actions`` but the assignment uses only what ONE defender observes, so a
    per-agent policy can clone it: attackers are ranked by their azimuth around the protected
    site (position relative to the site = rel_pos + own_rel_site, both in the observation) and
    defender ``i`` takes the ``i``-th live attacker in that order. Both defenders see the same
    attackers, so they compute the same ranking without knowing each other's position.
    Defenders left over (3 defenders vs 2 attackers) are goalkeepers on the attacker nearest the
    site. Lead pursuit as in ``smart_actions``."""
    n, E, dev = raw.cfg.num_attackers, raw.num_envs, raw.device
    agents = raw.cfg.possible_agents
    rows = torch.arange(E, device=dev)
    actions = {}
    for i, a in enumerate(agents):
        o = obs[a]
        alive = o[:, 9 + 6 * n : 9 + 7 * n] > 0.5
        rel = o[:, 9 : 9 + 3 * n].reshape(E, n, 3)
        va = o[:, 9 + 3 * n : 9 + 6 * n].reshape(E, n, 3) + o[:, 3:6].unsqueeze(1)
        site_rel = rel + o[:, 0:3].unsqueeze(1)  # attacker position relative to the site
        az = torch.atan2(site_rel[..., 1], site_rel[..., 0])
        az = torch.where(alive, az, torch.full_like(az, 10.0))  # dead attackers sort last
        order = az.argsort(dim=-1)  # (E, n) attacker indices by azimuth, live ones first
        n_alive = alive.sum(dim=-1)
        rank = torch.clamp(torch.full((E,), i, device=dev, dtype=torch.long), max=(n_alive - 1).clamp(min=0))
        target = order[rows, rank]
        # defenders beyond the live attacker count are goalkeepers: go for the attacker nearest
        # the site, i.e. the one whose interception is failing ("send another drone")
        site_dist = torch.where(alive, site_rel[..., :2].norm(dim=-1), torch.full_like(az, 1.0e3))
        target = torch.where(i >= n_alive, site_dist.argmin(dim=-1), target)
        r = rel[rows, target]
        v = va[rows, target]
        qa = (v * v).sum(-1) - v_max**2
        qb = 2.0 * (r * v).sum(-1)
        qc = (r * r).sum(-1)
        disc = qb * qb - 4.0 * qa * qc
        t_pursuit = r.norm(dim=-1) / v_max
        t_lead = (-qb - torch.sqrt(disc.clamp(min=0.0))) / (2.0 * qa.clamp(max=-1e-6))
        t = torch.where((disc >= 0) & (t_lead > 0), t_lead, t_pursuit)
        aim = r + v * t.unsqueeze(-1)
        direction = aim / torch.clamp(aim.norm(dim=-1, keepdim=True), min=1e-6)
        direction = direction * alive.any(dim=-1, keepdim=True).float()
        actions[a] = torch.cat([direction, torch.zeros(E, 1, device=dev)], dim=-1)
    return actions


TEACHERS = {"chase": chase_actions, "smart": smart_actions, "smart2": smart2_actions}
