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
_OBS_DIM = 9 + 7 * NUM_ATTACKERS + 4 * (NUM_DEFENDERS - 1)  # + per teammate: relative position (3) and active flag (1)


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

    # Tello SDK velocity interface (shared by both sides)
    command: TelloCommandCfg = TelloCommandCfg()

    # - scenario geometry
    num_defenders = NUM_DEFENDERS
    num_attackers = NUM_ATTACKERS
    site_radius = 1.5  # [m] protected perimeter around the env origin
    site_height = 2.5  # [m] height of the protected cylinder
    spawn_radius = 6.0  # [m] where attackers enter from
    defender_radius = 2.5  # [m] where defenders start, around the site
    capture_radius = 0.35  # [m] distance counting as an interception
    obs_clip_pos = 12.0  # [m] relative positions are clipped to this before normalisation
    obs_clip_vel = 3.0  # [m/s] relative velocities are clipped to this
    trainable_attackers = False

    # - scripted attacker behaviour (curriculum stage 0 = straight dive)
    attacker_speed = [0.4, 0.9]  # [m/s]
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
    respawn_waves = True
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
    teacher_sep_radius = 1.0  # [m]  (sweep 2026-10-24 with index spawn heights: 100% / 0.1%, 0.1% near-misses, p5 sep 0.52 m)
    teacher_sep_gain = 2.5
    teacher_sep_vertical = 0.8  # share of the repulsion that goes vertical (lower index dives, higher climbs)
    teacher_cruise_bias = 0.0  # [m] altitude offset while cruising; 0.2 tested slightly worse (100% / 0.1%, 0.5% near-misses) -> off
    teacher_standby_up = 1.0  # [m] goalkeeper standby height above the site centre
    # Defenders start at index-specific heights (metres above the floor), as they will be placed and
    # launched on the real field; empty list = uniform 1-2 m (the kit's default).
    defender_spawn_heights = [1.0, 1.5, 2.0]
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
