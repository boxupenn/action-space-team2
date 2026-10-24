"""Multi-agent defend task: Tello defenders intercept incoming attackers.

Each defender is an agent with the same four-channel velocity action as the single-drone
tasks, so trained policies remain flyable on real hardware. Attackers are scripted by
default and dive towards the protected site.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from action_space_kit.assets import TELLO_CFG, TELLO_MASS
from action_space_kit.control import TelloVelocityController
from action_space_kit.tasks.direct.tello_defend.chase import assignment_from_obs
from action_space_kit.rl.terminal_observation import TerminalObservationMixin

from .tello_defend_env_cfg import TelloDefendEnvCfg


class TelloDefendEnv(TerminalObservationMixin, DirectMARLEnv):
    cfg: TelloDefendEnvCfg

    def __init__(self, cfg: TelloDefendEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        inertia_zz = float(self._defenders[0].root_physx_view.get_inertias()[0].reshape(3, 3)[2, 2])
        make_controller = lambda: TelloVelocityController(  # noqa: E731
            cfg=self.cfg.command,
            num_envs=self.num_envs,
            mass=TELLO_MASS,
            inertia_zz=inertia_zz,
            device=self.device,
        )
        self._defender_ctl = [make_controller() for _ in range(self.cfg.num_defenders)]
        self._attacker_ctl = [make_controller() for _ in range(self.cfg.num_attackers)]

        self._actions = {
            agent: torch.zeros(self.num_envs, 4, device=self.device) for agent in self.cfg.possible_agents
        }
        # attacker bookkeeping
        self._attacker_alive = torch.ones(self.num_envs, self.cfg.num_attackers, device=self.device, dtype=torch.bool)
        self._attacker_speed = torch.zeros(self.num_envs, self.cfg.num_attackers, device=self.device)
        self._breached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_distance = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)
        # defenders that already rammed an attacker this wave (ramming rule, cfg.defender_spent_on_capture)
        self._spent = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device, dtype=torch.bool)
        self._episode_credit = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)
        self._episode_min_sep = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._spawn_angle = torch.zeros(self.num_envs, self.cfg.num_attackers, device=self.device)  # diagnostics
        if self.cfg.closing_mode not in ("distance", "own_motion"):
            raise ValueError(f"closing_mode must be 'distance' or 'own_motion', got {self.cfg.closing_mode!r}")
        if self.cfg.target_mode not in ("nearest", "assigned"):
            raise ValueError(f"target_mode must be 'nearest' or 'assigned', got {self.cfg.target_mode!r}")
        # target_mode "assigned": every injective defender -> attacker map, (num_perms, num_defenders)
        if self.cfg.num_defenders <= self.cfg.num_attackers:
            perms = list(itertools.permutations(range(self.cfg.num_attackers), self.cfg.num_defenders))
        else:  # more defenders than attackers: no one-to-one map exists, fall back to nearest
            perms = []
        self._target_perms = torch.tensor(perms, dtype=torch.long, device=self.device)
        self._yaw_mask = torch.tensor([1.0, 1.0, 1.0, 0.0], device=self.device)
        # Episode outcome, latched in _get_dones before Isaac Lab resets the environment.
        # Reading _breached after env.step() is too late: the reset has already cleared it.
        self._episode_captures = torch.zeros(self.num_envs, device=self.device)
        self.outcome_breached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.outcome_captures = torch.zeros(self.num_envs, device=self.device)
        self.outcome_credit = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)  # captures per defender
        self.outcome_min_sep = torch.full((self.num_envs,), float("inf"), device=self.device)  # min defender-defender distance
        # diagnostics latched with the outcome: attacker spawn azimuths, speeds, episode time
        self.outcome_spawn_angle = torch.zeros(self.num_envs, self.cfg.num_attackers, device=self.device)
        self.outcome_attacker_speed = torch.zeros(self.num_envs, self.cfg.num_attackers, device=self.device)
        self.outcome_time = torch.zeros(self.num_envs, device=self.device)
        self.outcome_valid = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # -- scene ------------------------------------------------------------------

    def _setup_scene(self):
        self._defenders = []
        self._attackers = []
        for i in range(self.cfg.num_defenders):
            cfg = TELLO_CFG.replace(prim_path=f"/World/envs/env_.*/Defender_{i}")
            obj = RigidObject(cfg)
            self._defenders.append(obj)
        for i in range(self.cfg.num_attackers):
            cfg = TELLO_CFG.replace(prim_path=f"/World/envs/env_.*/Attacker_{i}")
            obj = RigidObject(cfg)
            self._attackers.append(obj)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Sensors must exist before cloning: the clone creates their per-env prims, and
        # attaching render annotators after the fact fails inside Replicator with
        # "Unable to write from unknown dtype".
        self._setup_extra_sensors()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for i, obj in enumerate(self._defenders):
            self.scene.rigid_objects[f"defender_{i}"] = obj
        for i, obj in enumerate(self._attackers):
            self.scene.rigid_objects[f"attacker_{i}"] = obj

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        marker_cfg = CUBOID_MARKER_CFG.copy()
        marker_cfg.markers["cuboid"].size = (self.cfg.site_radius, self.cfg.site_radius, 0.05)
        marker_cfg.prim_path = "/Visuals/site"
        self._site_markers = VisualizationMarkers(marker_cfg)

    def _setup_extra_sensors(self) -> None:
        """Hook for subclasses to add sensors. Called before the scene is cloned."""
        return

    # -- stepping ---------------------------------------------------------------

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent, action in actions.items():
            self._actions[agent] = action.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        for i, drone in enumerate(self._defenders):
            actions = self._actions[f"defender_{i}"]
            if not self.cfg.yaw_action:
                actions = actions * self._yaw_mask
            if self.cfg.defender_spent_on_capture:
                actions = actions * (~self._spent[:, i]).float().unsqueeze(-1)  # a spent defender just hovers
            force, torque = self._defender_ctl[i].compute(
                actions=actions,
                lin_vel_w=drone.data.root_lin_vel_w,
                quat_w=drone.data.root_quat_w,
                ang_vel_b=drone.data.root_ang_vel_b,
                projected_gravity_b=drone.data.projected_gravity_b,
                dt=self.physics_dt,
            )
            drone.set_external_force_and_torque(forces=force, torques=torque, body_ids=[0])

        attacker_actions = self._scripted_attacker_actions()
        for i, drone in enumerate(self._attackers):
            force, torque = self._attacker_ctl[i].compute(
                actions=attacker_actions[i],
                lin_vel_w=drone.data.root_lin_vel_w,
                quat_w=drone.data.root_quat_w,
                ang_vel_b=drone.data.root_ang_vel_b,
                projected_gravity_b=drone.data.projected_gravity_b,
                dt=self.physics_dt,
            )
            drone.set_external_force_and_torque(forces=force, torques=torque, body_ids=[0])

    def _scripted_attacker_actions(self) -> list[torch.Tensor]:
        """Fly each attacker at the protected site, with optional lateral weaving."""
        actions = []
        for i, drone in enumerate(self._attackers):
            to_site = self.scene.env_origins - drone.data.root_pos_w
            to_site[:, 2] += self.cfg.site_height * 0.5
            direction = to_site / torch.clamp(torch.linalg.norm(to_site, dim=-1, keepdim=True), min=1e-6)
            if self.cfg.attacker_evasion > 0.0:
                phase = self.episode_length_buf.float() * self.step_dt * 2.0 + i
                lateral = torch.stack([-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])], dim=-1)
                direction = direction + self.cfg.attacker_evasion * torch.sin(phase).unsqueeze(-1) * lateral
                direction = direction / torch.clamp(torch.linalg.norm(direction, dim=-1, keepdim=True), min=1e-6)
            speed = (self._attacker_speed[:, i] / self.cfg.command.max_lin_speed).unsqueeze(-1)
            action = torch.cat([direction * speed, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
            # a downed attacker stops moving
            action = action * self._attacker_alive[:, i].unsqueeze(-1).float()
            actions.append(action.clamp(-1.0, 1.0))
        return actions

    # -- MDP --------------------------------------------------------------------

    def _attacker_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attacker positions and velocities, shaped (num_envs, num_attackers, 3)."""
        pos = torch.stack([d.data.root_pos_w for d in self._attackers], dim=1)
        vel = torch.stack([d.data.root_lin_vel_w for d in self._attackers], dim=1)
        return pos, vel

    def _get_observations(self) -> dict[str, torch.Tensor]:
        attacker_pos, attacker_vel = self._attacker_states()
        site = self.scene.env_origins.clone()
        site[:, 2] += self.cfg.site_height * 0.5

        obs = {}
        defender_pos = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)  # (envs, D, 3)
        for i, drone in enumerate(self._defenders):
            own_rel_site = drone.data.root_pos_w - site
            # per teammate: position relative to me (clipped) and an active flag (0 once it has rammed
            # and is out -- on the floor in reality -- so policies and the teacher stop avoiding it)
            mates = torch.cat(
                [
                    torch.cat(
                        [
                            (defender_pos[:, j] - drone.data.root_pos_w).clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos),
                            (~self._spent[:, j]).float().unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    for j in range(self.cfg.num_defenders)
                    if j != i
                ],
                dim=-1,
            )
            rel_pos = attacker_pos - drone.data.root_pos_w.unsqueeze(1)
            rel_vel = attacker_vel - drone.data.root_lin_vel_w.unsqueeze(1)
            # A downed attacker is zeroed out and flagged, not pushed to a huge sentinel
            # value: a 100 m spike dominates the running normalisation and destabilises
            # the policy. The flag tells the network the slot is empty.
            alive = self._attacker_alive.float().unsqueeze(-1)
            rel_pos = rel_pos.clamp(-self.cfg.obs_clip_pos, self.cfg.obs_clip_pos) * alive
            rel_vel = rel_vel.clamp(-self.cfg.obs_clip_vel, self.cfg.obs_clip_vel) * alive
            obs[f"defender_{i}"] = torch.cat(
                [
                    own_rel_site,
                    drone.data.root_lin_vel_w,
                    drone.data.projected_gravity_b,
                    rel_pos.reshape(self.num_envs, -1),
                    rel_vel.reshape(self.num_envs, -1),
                    alive.squeeze(-1),
                    mates,  # teammates' positions relative to me, so the policy can keep clear of them
                ],
                dim=-1,
            )
        if self.cfg.obs_assignment:
            n, D = self.cfg.num_attackers, self.cfg.num_defenders
            for i in range(D):
                a = f"defender_{i}"
                target, goalkeeper = assignment_from_obs(obs[a], i, n, D, self.cfg.command.max_lin_speed)
                onehot = torch.nn.functional.one_hot(target, n).float() * (~goalkeeper).float().unsqueeze(-1)
                obs[a] = torch.cat([obs[a], onehot, goalkeeper.float().unsqueeze(-1)], dim=-1)
        return obs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        attacker_pos, _ = self._attacker_states()
        site = self.scene.env_origins.clone()
        site[:, 2] += self.cfg.site_height * 0.5

        # interceptions: any live attacker within capture radius of any defender
        captured = torch.zeros_like(self._attacker_alive)
        defender_credit = torch.zeros(self.num_envs, self.cfg.num_defenders, device=self.device)
        for i, drone in enumerate(self._defenders):
            dist = torch.linalg.norm(attacker_pos - drone.data.root_pos_w.unsqueeze(1), dim=-1)
            hit = (dist < self.cfg.capture_radius) & self._attacker_alive
            if self.cfg.defender_spent_on_capture:
                hit &= ~self._spent[:, i].unsqueeze(-1)
            captured |= hit
            defender_credit[:, i] = hit.float().sum(dim=-1)
        self._attacker_alive &= ~captured
        if self.cfg.defender_spent_on_capture:
            self._spent |= defender_credit > 0
        self._episode_credit += defender_credit
        if self.cfg.num_defenders > 1:
            dpos = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)
            pair = torch.cdist(dpos, dpos)  # (envs, D, D)
            pair = pair + torch.eye(self.cfg.num_defenders, device=self.device) * 1.0e3
            if self.cfg.defender_spent_on_capture and not self.cfg.collision_count_spent:
                spent_pair = self._spent.unsqueeze(1) | self._spent.unsqueeze(2)
                pair = torch.where(spent_pair, torch.full_like(pair, 1.0e3), pair)
            self._episode_min_sep = torch.minimum(self._episode_min_sep, pair.flatten(1).min(dim=-1).values)

        # breach: a live attacker inside the protected cylinder
        radial = torch.linalg.norm(attacker_pos[:, :, :2] - site[:, :2].unsqueeze(1), dim=-1)
        inside = (radial < self.cfg.site_radius) & self._attacker_alive
        self._breached = inside.any(dim=-1)

        # every attacker down this step: the outcome the task actually wants
        cleared = captured.any(dim=-1) & ~self._attacker_alive.any(dim=-1)
        self._episode_captures += captured.float().sum(dim=-1)

        # the attacker each defender's shaping terms refer to: distance and unit direction to it
        target_dist, target_dir = self._select_targets(attacker_pos)
        any_alive = self._attacker_alive.any(dim=-1)

        rewards = {}
        for i, drone in enumerate(self._defenders):
            nearest = target_dist[:, i]
            # With no live attacker left there is no distance to close. Carrying the
            # previous value keeps `closing` at zero instead of swinging to the clamp,
            # which otherwise punishes the defender for the capture it just made.
            nearest = torch.where(any_alive, nearest, self._prev_distance[:, i])
            if self.cfg.closing_mode == "own_motion":
                # metres closed by the defender's own motion this step; an attacker flying at a
                # stationary defender pays nothing
                closing = torch.sum(drone.data.root_lin_vel_w * target_dir[:, i], dim=-1) * self.step_dt
                closing = closing * any_alive.float()
            else:
                closing = self._prev_distance[:, i] - nearest
            self._prev_distance[:, i] = nearest
            effort = torch.sum(torch.square(self._actions[f"defender_{i}"]), dim=-1)
            proximity = torch.exp(-nearest.clamp(min=0.0, max=self.cfg.spawn_radius * 2.0) / self.cfg.spawn_radius)
            # Rates are scaled by step_dt; per-step deltas are not.
            #
            # proximity/alive/effort are rates (value per second), so without step_dt
            # their income over a 400-step episode dwarfs the capture bonus, and
            # clearing the field -- which ends the episode -- costs more future reward
            # than it pays, making stalling optimal.
            #
            # closing is already a per-step distance delta in metres. Scaling it by
            # step_dt as well shrinks the dense signal ~60x, leaving PPO with almost
            # nothing to follow between sparse captures. It telescopes over an episode
            # to (start distance - end distance), so at scale 5 a full 5 m approach is
            # worth ~25, comparable to one capture.
            shaping = (
                self.cfg.rew_scale_proximity * proximity
                + self.cfg.rew_scale_effort * effort
                + self.cfg.rew_scale_alive
            ) * self.step_dt + self.cfg.rew_scale_closing * closing.clamp(-1.0, 1.0)
            events = (
                self.cfg.rew_scale_capture * defender_credit[:, i]
                + self.cfg.rew_scale_breach * self._breached.float()
                + self.cfg.rew_scale_cleared * cleared.float()
            )
            rewards[f"defender_{i}"] = shaping + events
        return rewards

    def _select_targets(self, attacker_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick each defender's target attacker for the shaping terms.

        Returns the distance to it, (num_envs, num_defenders), ``inf`` where no attacker is
        alive, and the unit direction from the defender to it, (num_envs, num_defenders, 3).
        ``cfg.target_mode`` selects the nearest live attacker or a one-to-one assignment.
        """
        defender_pos = torch.stack([d.data.root_pos_w for d in self._defenders], dim=1)
        rel = attacker_pos.unsqueeze(1) - defender_pos.unsqueeze(2)  # (envs, defenders, attackers, 3)
        dist = torch.linalg.norm(rel, dim=-1)
        dist = torch.where(self._attacker_alive.unsqueeze(1), dist, torch.full_like(dist, float("inf")))
        nearest = dist.argmin(dim=-1)  # (envs, defenders)
        if self.cfg.target_mode == "assigned" and len(self._target_perms):
            # cheapest injective map; a downed attacker costs a lot so it is avoided when possible
            cost_dist = torch.where(torch.isinf(dist), torch.full_like(dist, 1.0e3), dist)
            rows = torch.arange(self.cfg.num_defenders, device=self.device).unsqueeze(0).expand_as(self._target_perms)
            cost = cost_dist[:, rows, self._target_perms].sum(dim=-1)  # (envs, perms)
            target = self._target_perms[cost.argmin(dim=-1)]  # (envs, defenders)
            assigned_alive = self._attacker_alive.gather(1, target)
            target = torch.where(assigned_alive, target, nearest)
        else:
            target = nearest
        target_dist = dist.gather(2, target.unsqueeze(-1)).squeeze(-1)  # (envs, defenders)
        target_rel = rel.gather(2, target[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        target_dir = target_rel / torch.clamp(torch.linalg.norm(target_rel, dim=-1, keepdim=True), min=1e-6)
        return target_dist, target_dir

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        all_down = ~self._attacker_alive.any(dim=-1)
        if self.cfg.respawn_waves:
            # a cleared field starts the next wave; only a breach ends the episode
            if all_down.any():
                self._respawn_attackers(all_down.nonzero(as_tuple=False).flatten())
            done = self._breached
        else:
            done = self._breached | all_down
        finished = done | time_out
        self.outcome_valid = finished
        self.outcome_breached = torch.where(finished, self._breached, self.outcome_breached)
        self.outcome_captures = torch.where(finished, self._episode_captures, self.outcome_captures)
        self.outcome_credit = torch.where(finished.unsqueeze(-1), self._episode_credit, self.outcome_credit)
        self.outcome_min_sep = torch.where(finished, self._episode_min_sep, self.outcome_min_sep)
        self.outcome_spawn_angle = torch.where(finished.unsqueeze(-1), self._spawn_angle, self.outcome_spawn_angle)
        self.outcome_attacker_speed = torch.where(finished.unsqueeze(-1), self._attacker_speed, self.outcome_attacker_speed)
        self.outcome_time = torch.where(finished, self.episode_length_buf.float() * self.step_dt, self.outcome_time)
        terminated = {agent: done for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _respawn_attackers(self, env_ids: torch.Tensor) -> None:
        """Send a fresh wave into the listed environments, leaving defenders as they are."""
        if len(env_ids) == 0:
            return
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]
        for i, drone in enumerate(self._attackers):
            angle = sample_uniform(0.0, 6.2832, (num,), self.device)
            self._spawn_angle[env_ids, i] = angle
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.spawn_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.spawn_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.5, 3.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._attacker_ctl[i].reset(env_ids)
        self._attacker_alive[env_ids] = True
        self._attacker_speed[env_ids] = sample_uniform(
            self.cfg.attacker_speed[0], self.cfg.attacker_speed[1], (num, self.cfg.num_attackers), self.device
        )
        self._prev_distance[env_ids] = self.cfg.spawn_radius
        self._spent[env_ids] = False

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._defenders[0]._ALL_INDICES
        super()._reset_idx(env_ids)
        num = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        # defenders start on a ring around the site
        base_angle = sample_uniform(0.0, 6.2832, (num,), self.device)
        for i, drone in enumerate(self._defenders):
            if self.cfg.defender_spawn_even:
                angle = base_angle + i * 6.2832 / self.cfg.num_defenders + sample_uniform(
                    -self.cfg.defender_spawn_jitter, self.cfg.defender_spawn_jitter, (num,), self.device
                )
            else:
                angle = sample_uniform(0.0, 6.2832, (num,), self.device)
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.defender_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.defender_radius * torch.sin(angle)
            heights = self.cfg.defender_spawn_heights
            if heights and i < len(heights):
                state[:, 2] = origins[:, 2] + heights[i] + sample_uniform(-0.1, 0.1, (num,), self.device)
            else:
                state[:, 2] = origins[:, 2] + sample_uniform(1.0, 2.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._defender_ctl[i].reset(env_ids)

        # attackers enter from a wider ring
        for i, drone in enumerate(self._attackers):
            angle = sample_uniform(0.0, 6.2832, (num,), self.device)
            self._spawn_angle[env_ids, i] = angle
            state = drone.data.default_root_state[env_ids].clone()
            state[:, 0] = origins[:, 0] + self.cfg.spawn_radius * torch.cos(angle)
            state[:, 1] = origins[:, 1] + self.cfg.spawn_radius * torch.sin(angle)
            state[:, 2] = origins[:, 2] + sample_uniform(1.5, 3.0, (num,), self.device)
            state[:, 7:] = 0.0
            drone.write_root_pose_to_sim(state[:, :7], env_ids)
            drone.write_root_velocity_to_sim(state[:, 7:], env_ids)
            self._attacker_ctl[i].reset(env_ids)

        self._attacker_alive[env_ids] = True
        self._attacker_speed[env_ids] = sample_uniform(
            self.cfg.attacker_speed[0], self.cfg.attacker_speed[1], (num, self.cfg.num_attackers), self.device
        )
        self._breached[env_ids] = False
        self._episode_captures[env_ids] = 0.0
        self._episode_credit[env_ids] = 0.0
        self._episode_min_sep[env_ids] = float("inf")
        self._spent[env_ids] = False
        for agent in self.cfg.possible_agents:
            self._actions[agent][env_ids] = 0.0
        self._prev_distance[env_ids] = self.cfg.spawn_radius

    def _debug_vis_callback(self, event):
        site = self.scene.env_origins.clone()
        self._site_markers.visualize(site)
