"""Configuration for the multi-agent defend task: the hackathon scenario."""

from __future__ import annotations

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from action_space_kit.control import TelloCommandCfg

NUM_DEFENDERS = 3  # real-world spec (2026-10-23): 3 defenders vs 2 attackers
NUM_ATTACKERS = 2

# obs per defender: own pos rel. site (3) + own vel (3) + gravity (3)
#                 + per attacker (rel pos 3, rel vel 3, visible flag 1)
OBS_TEAMMATES = True  # per teammate [rel xyz, active]; False = the kit's 23-dim layout the organisers' runner feeds (2026-10-24)
OBS_ASSIGNMENT = True  # append the decentralised target assignment to the observation (chase.assign_latched / assignment_from_obs)
_OBS_DIM = 9 + 7 * NUM_ATTACKERS + (4 * (NUM_DEFENDERS - 1) if OBS_TEAMMATES else 0) + (NUM_ATTACKERS + 1 if OBS_ASSIGNMENT else 0)
# 9 own + per attacker 7 [+ per teammate [rel xyz, active]] + [one-hot assigned attacker, goalkeeper flag]
# The exported defender_i.pt takes the 23 kit numbers and computes the 3 assignment features itself (export_skrl_policy.py --obs23).


@configclass
class TelloDefendEnvCfg(DirectMARLEnvCfg):
    """Defenders intercept attackers before they reach the protected site.

    Attackers are scripted by default (``trainable_attackers = False``), which is the
    fastest path to a working defender policy. Set it True to train both sides.
    """

    # env
    decimation = 2
    episode_length_s = 20.0

    # multi-agent spaces -- every agent uses the four SDK rc channels
    possible_agents = [f"defender_{i}" for i in range(NUM_DEFENDERS)]
    # Bounded on purpose: the four SDK rc channels live in [-1, 1], and an unbounded
    # space leaves skrl nothing to clip sampled actions against.
    action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in possible_agents}
    observation_spaces = {agent: _OBS_DIM for agent in possible_agents}
    state_space = -1  # concatenate agent observations

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # scene: 1024 envs x 4 drones fits comfortably on a 24 GB GPU shared by several teams
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=12.0, replicate_physics=True)

    # Tello SDK velocity interface (shared by both sides). Real-Tello numbers (2026-10-24): full rc stick in
    # normal mode is ~2 m/s (forum measurements; the 8 m/s spec is sport mode), tilt-limited acceleration
    # ~3 m/s^2, ~0.25 s stick-to-velocity lag. Actions are fractions of max_lin_speed. Measure rc -> m/s on
    # site and set env.command.max_lin_speed to it; the geometry only depends on the defender/attacker ratio.
    command: TelloCommandCfg = TelloCommandCfg(max_lin_speed=2.0, max_lin_accel=3.0, response_time=0.25)

    # - scenario geometry
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    site_radius = 1.3  # [m] protected perimeter (cylinder rule; unused while breach_x > 0, still drawn as the site square)
    # ARENA (user's figure 2026-10-24): room 6.6 x 3.5 m, tag-10 wall = the short wall; the BASE is a 13 cm box
    # 0.86 m out of that wall and 0.94 m from the long wall on the viewer's left (tag x = -0.74 + 0.94). An
    # attacker that crosses the line 1.05 m from the tag-10 wall (towards it) wins: breach = attacker sim x < breach_x.
    # Sim frame = (tag z out of the wall, tag x right, up); site_xy is the base relative to the env origin.
    site_xy = [0.86, 0.20]  # [m]
    breach_x = 1.30  # [m] the base line, 1.3 m from the tag-10 wall (user 2026-10-24); 0 = the cylinder rule (site_radius) instead
    site_height = 2.5  # [m] height of the protected cylinder
    spawn_radius = 4.7  # [m] from the base: attackers take off in the far (left) half of the 6.6 m room, 4.6-6.3 m from the wall
    spawn_radius_jitter = 0.9  # [m]
    attacker_spawn_sector_deg = 60.0  # attackers enter within +-this azimuth of +x (the room side of the tag wall); 180 = full ring
    attacker_spawn_z = [1.2, 1.8]  # [m] above the floor (formation flies at ~1.2-1.7 m)
    defender_radius = 2.5  # [m] where defenders start, around the site
    capture_radius = 0.35  # [m] distance counting as an interception (sphere; used when capture_box is empty)
    # Organisers' hit box (2026-10-24): a ram counts when |dx|, |dy|, |dz| between the two drones are all
    # within these half-extents (metres). Set to [] to use the capture_radius sphere.
    # ponytail: read as half-extents of "15 cm x 15 cm x 3 cm"; if they meant full extents, halve these.
    capture_box = [0.15, 0.15, 0.03]
    obs_clip_pos = 12.0  # [m] relative positions are clipped to this before normalisation
    obs_clip_vel = 3.0  # [m/s] relative velocities are clipped to this
    trainable_attackers = False

    # - scripted attacker behaviour (curriculum stage 0 = straight dive)
    attacker_speed = [0.3, 0.7]  # [m/s]  swarm_formation --fast caps rc at 35 of a ~2 m/s stick
    # After launch a real attacker takes off, climbs and searches for a tag before it moves: it hovers in
    # place (visible, alive) for a time sampled from this range, seconds. [0, 0] = moves at once.
    attacker_hover = [1.0, 4.0]
    # Take-off (2026-10-24 afternoon): everybody starts on the floor at takeoff_z. An attacker climbs at
    # attacker_climb_speed to its cruise height (attacker_spawn_z) and is invisible / uncapturable until it is
    # above attacker_visible_z (the runner ignores bus attackers below attacker_min_height_cm); it moves off once
    # at cruise height and past its hover time. Defenders climb to their launch heights by themselves (the
    # teacher's go-home rule). takeoff_z 0 = everybody starts airborne at its height (the earlier behaviour).
    takeoff_z = 0.3  # [m]
    attacker_climb_speed = 0.5  # [m/s]
    attacker_visible_z = 0.5  # [m]  (runner attacker_min_height_cm 50)
    # Launch points relative to the site (metres, site frame): defenders one per index, attackers a list of
    # candidate points each attacker picks from at random (+-spawn_xy_jitter). Empty attacker list = random
    # ring at spawn_radius (the kit default). Fill both from the organisers' arena figure; the defender
    # points are also where the teacher / clone flies back to once the field is cleared.
    # From the organisers' map.yaml (2026-10-24): tag-10 wall at z=0, floor 1.25 m below tag 10, right wall at
    # x~2.8 m, floor tags out to z~5 m, left side mapped only to x~-0.5. Sim frame = (tag z, tag x, up).
    # Launch spots = the three crosses of the figure (tag frame x, z): 1.37 m out / mid-room, 0.99 m out next to the
    # base, 0.50 m out behind the line. defender_2 (goalkeeper role when idle) is the one by the base.
    defender_spawn_xy = [[1.37, 1.0], [0.50, 1.52], [0.99, 0.33]]
    attacker_spawn_xy = []
    # Room box from map.yaml, sim frame [x out of the wall, y right, z above the floor]: attackers only
    # spawn inside it and the teacher never commands a velocity through its faces (the runner's fence does
    # the same on the real drone). [] = no box.
    arena_bounds = [[0.3, 6.3], [-0.64, 2.66], [0.5, 2.25]]  # 6.6 x 3.5 m room; tag x -0.74 (left wall) .. 2.76 (right wall, tag 8)
    spawn_xy_jitter = 0.2  # [m]
    # Seconds between a cleared field and the next wave's first launch (respawn_waves only): the
    # defenders fly home in the gap, so the clone learns the return.
    wave_gap = 3.0
    # Staggered launches: attacker 0 enters at t=0, every further attacker after a delay sampled
    # uniformly from this range (seconds). A pending attacker is parked at its spawn point, is not
    # observable (alive flag 0) and cannot be captured or breach -- the arena case where the
    # organisers launch one attacker first or one crashes early, leaving a single live attacker and
    # two goalkeepers. [0, 0] = everyone at t=0 (the kit's original behaviour).
    attacker_launch_delay = [0.0, 3.0]
    attacker_evasion = 0.0  # 0 = straight line, 1 = strong lateral weave

    # - reward scales (per defender)
    # Balance matters here: a breach happens in nearly every early episode, so a huge
    # penalty drowns the shaping terms and training drifts below a do-nothing baseline
    # (measured: -45.8 return for zero actions). Keep the penalty comparable to what a
    # defender can earn by doing its job.
    rew_scale_closing = 5.0  # getting nearer its target attacker
    rew_scale_proximity = 2.0  # continuous pull towards the nearest attacker
    rew_scale_capture = 30.0  # intercepting one
    rew_scale_cleared = 30.0  # bonus for downing the last attacker of a wave
    respawn_waves = False  # arena 2026-10-24: a cleared field ends the round, everybody re-appears at its launch spot
    # (takes off again); nobody flies back. True = the kit's endless waves (defenders stay airborne).
    # Ramming rule (real spec 2026-10-23): an interception is a physical collision and the
    # defender that makes it is out too. True: that defender's commands are zeroed for the rest
    # of the wave and it cannot capture again (reset on a new wave / episode).
    defender_spent_on_capture = True
    # Defender-defender separation: the per-episode minimum pairwise distance is latched in
    # outcome_min_sep; a minimum below this counts as a defender collision in evaluate.py.
    defender_collision_radius = 0.3
    collision_count_spent = False  # a rammed defender is on the floor in reality: pairs with it do not count
    # Defenders start evenly spaced around the site (2*pi/num_defenders apart, +-defender_spawn_jitter
    # rad), as they will be placed on the real field. False: independent random angles (the kit's
    # default), which puts two of three within 0.3 m in ~6% of episodes.
    defender_spawn_even = True
    defender_spawn_jitter = 0.3  # [rad]
    # smart2 teacher knobs (chase.py reads them when present): teammate repulsion radius / gain
    teacher_sep_radius = 0.7  # [m]  arena sweep 2026-10-24 (box judging): 100% / 0%, near-misses 9%, covered by the runner guard
    teacher_sep_gain = 2.5
    teacher_sep_vertical = 0.0  # vertical repulsion is a vertical manoeuvre: it misses the 3 cm box (99.3% -> 100% without)
    teacher_cruise_bias = 0.0  # [m] altitude offset while cruising; 0.2 tested slightly worse (100% / 0.1%, 0.5% near-misses) -> off
    teacher_standby_up = 0.8  # [m] goalkeeper standby height above the site centre (~2.05 m: above the 1.2-1.8 m attackers)
    teacher_standby_ring = 0.8  # [m] goalkeeper ring, used only when defender_spawn_xy is empty (else a goalkeeper waits at its own launch point)
    # smart2 target assignment: "azimuth" = defender i takes the i-th attacker by azimuth around the
    # site (ignores where the defenders are); "time" = every defender computes, from its own
    # observation (own + teammates' positions), the assignment that minimises the summed lead-pursuit
    # intercept time over the live attackers; the leftover defender is the goalkeeper. Same inputs ->
    # same answer on every drone, no communication.
    teacher_assign = "time"
    # "latched" (2026-10-24, 23-dim runner): teammate-free. Attackers ranked by azimuth, defender i takes
    # rank i when the whole wave is visible or when it holds no target; the target is then kept until that
    # attacker is down (a teammate's ram does not re-shuffle a pursuit in progress); unassigned = goalkeeper.
    # Goalkeepers engage once a live attacker is within teacher_alert_radius of the site.
    teacher_alert_radius = 2.2  # [m]  goalkeepers engage inside this of the base (figure arena, 300 ep: 99.8% / 0.3%, 1.3% near-misses; 3.0 m: 100% / 0% but 4.3%)
    # ... AND only while no active teammate is within teacher_cover_m of that attacker (someone is about to
    # ram it): otherwise two defenders crowd one attacker, repel each other and both miss the 3 cm box.
    # The dispatch is part of the assignment feature (chase.assignment_from_obs). None = old rule.
    teacher_cover_m = 1.2  # [m]
    teacher_alert_force = 1.7  # [m] inside this the goalkeeper goes regardless of cover (the base line is 1.3 m out); 0 = off
    teacher_aim_up = 0.02  # [m] aim offset above the attacker: inside the 3 cm box, but from above (props vs frame)
    # Ram from above (2026-10-24, 15x15x3 cm box): while below the lead-pursuit aim point the defender climbs
    # at least teacher_climb_slope x horizontal distance (1 = 45 deg), so it gets above early and the
    # terminal run is the plain straight line, descending onto the props. 0 = plain lead pursuit.
    # (Tried and rejected: a vertical P-loop lags a descending attacker, 72-83%; an aim point that slides
    # down over the last 2 m lags the 0.25 s response, 39-50%. The last metre must be a straight line.)
    teacher_climb_slope = 0.0
    # Terminal slow-down: within teacher_terminal_dist of the aim point the stick is scaled to
    # teacher_terminal_speed (full speed passes straight through the 3 cm box; the attackers are slow). 0 = off.
    teacher_terminal_dist = 0.0  # [m]  (0.5 x inside 1 m: 99.7% / 0.7% and 5.7% near-misses -> off)
    teacher_terminal_speed = 0.5
    # Optional creep (20% speed) within teacher_align_dist while still > teacher_align_tol below the
    # attacker; 0 = off (a 0.9 m/s attacker flies past a creeping defender).
    teacher_align_dist = 0.0  # [m]
    teacher_align_tol = 0.05  # [m]
    obs_teammates = OBS_TEAMMATES
    # Append [one-hot of my assigned attacker (n), goalkeeper flag (1)] to the observation, computed
    # per defender from its own observation by chase.assignment_from_obs. The discrete allocation is
    # then an input, and the cloned network only has to learn the continuous pursuit (2026-10-24: the
    # clone of the time-assignment teacher without this reached cosine 0.95 / 98.2%; the rule itself
    # is 100% / 0%). The real runner computes the same feature from the same numbers.
    obs_assignment = OBS_ASSIGNMENT
    # Defenders start at index-specific heights (metres above the floor), as they will be placed and
    # launched on the real field; empty list = uniform 1-2 m (the kit's default).
    defender_spawn_heights = [1.45, 1.5, 1.55]  # the runner climbs to 1.5 m after take-off (user 2026-10-24); teacher 100% / 0% here vs 99.8% / 0.1% at 1.9-2.2 m
    """Launch a fresh wave when the field is cleared, instead of ending the episode.

    Ending on success truncates the value bootstrap, which makes a short successful
    episode look worse than a long passive one -- measured as a policy that started at
    +137 return and unlearned its way to -260 while episodes grew longer.
    """
    rew_scale_breach = -20.0  # an attacker reaching the site (shared)
    rew_scale_effort = -0.01
    rew_scale_alive = 0.1

    # - shaping variants (WINDOWS_TRAINING.md section 10); the defaults reproduce the original reward
    closing_mode = "distance"
    """What ``rew_scale_closing`` pays for.

    ``"distance"``: the per-step decrease of the distance to the target attacker. A stationary
    defender earns it too, because the attacker flies towards the site on its own.
    ``"own_motion"``: the defender's own velocity projected onto the direction to its target,
    times step_dt (metres closed by its own motion), so only the defender moving at its target pays.
    """
    yaw_action = True
    """Whether the defenders' fourth action channel (yaw rate) is applied.

    ``False`` forces the yaw-rate command to zero (the action space stays 4-D, the channel is
    ignored), so a defender keeps the heading it spawned with. It was the section 10 workaround
    for the controller-frame bug (the world-frame force used to be applied in the body frame, so
    a yawed drone's horizontal loop rotated with its heading); the controller now returns a
    body-frame force and yaw no longer disturbs the velocity loop (section 11), so the default
    ``True`` is the normal setting and ``False`` only reproduces the section 10 runs.
    """
    target_mode = "nearest"
    """Which attacker the closing / proximity terms refer to.

    ``"nearest"``: the nearest live attacker; both defenders may chase the same one.
    ``"assigned"``: a one-to-one assignment, the permutation of attackers that minimises the summed
    defender-to-attacker distance over live attackers, so the defenders split the attackers. A
    defender whose assigned attacker is down falls back to the nearest live one.
    """
