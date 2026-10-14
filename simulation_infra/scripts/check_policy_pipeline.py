"""Isaac integration check: inference parity, timeout terminals, and gated touchdown."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import json

import action_space_kit.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
from action_space_kit.rl.policy_export import norm_path_for
from action_space_kit.rl.run_config import restore_run_config
from action_space_kit.rl.sb3_shared import Sb3CentralVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drone_deployment_infra"))
from core.room_policy import RoomPolicy


def feed(u):
    fields = {
        "position": u.pub_pos,
        "velocity": u.kf.v,
        "position_variance": u.kf.a,
        "velocity_variance": u.kf.c,
        "target": u._spots,
        "yaw": u.pub_yaw,
        "fix_age": u.fix_age,
        "fresh": u.fresh,
        "acquired": u.acquired,
        "lost": u.lost,
        "height": u.height_meas,
        "taking_off": u.taking_off,
        "landed": u.landed,
    }
    return {k: v[0].detach().cpu().tolist() for k, v in fields.items()}


def main():
    cfg = load_cfg_from_registry("AS-Tello-Room-Single-v0", "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry("AS-Tello-Room-Single-v0", "sb3_cfg_entry_point")
    restore_run_config(args.checkpoint, cfg, agent_cfg)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 2.0  # force multiple timeout/reset cases for any single-drone checkpoint
    base = Sb3CentralVecEnv(gym.make("AS-Tello-Room-Single-v0", cfg=cfg))
    u = base.unwrapped_env
    normal = VecNormalize.load(str(norm_path_for(args.checkpoint)), base)
    normal.training = False
    normal.norm_reward = False
    env = VecFrameStack(normal, n_stack=agent_cfg["frame_stack"])
    model = PPO.load(args.checkpoint, env, device="cpu")
    adapter = RoomPolicy(args.checkpoint)
    obs = env.reset()
    u.episode_length_buf[:] = 0
    records, resets = [], 0
    done = np.array([True])
    for i in range(90):
        sample = feed(u)
        sample["reset"] = i == 0 or bool(done[0])
        if sample["reset"]:
            adapter.reset()
        result = adapter.step(sample)
        raw = u._get_observations()["drone_0"][0].cpu().numpy()
        np.testing.assert_allclose(result["observation"], raw, atol=2e-6)
        action, _ = model.predict(obs, deterministic=True)
        np.testing.assert_allclose(result["action"], action, rtol=1e-4, atol=2e-5)
        sample["expected_observation"] = raw.tolist()
        sample["expected_action"] = action.tolist()
        records.append(sample)
        obs, reward, done, infos = env.step(action)
        assert np.isfinite(reward).all()
        np.testing.assert_array_equal(result["rc"], u.last_rc[0].cpu().numpy()) if not done[0] else None
        if done[0]:
            resets += 1
            assert infos[0]["TimeLimit.truncated"]
            assert not np.array_equal(infos[0]["terminal_observation"], obs[0])
    assert resets >= 2
    Path("outputs/policy_replay.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"[ok] 90 simulator/adapter observation, action and RC comparisons; {resets} timeout resets")

    # Controlled touchdown: the policy requests land while zero sticks hold over the target.
    # Remove measurement/link randomness to isolate the gate+firmware integration.
    cfg.episode_length_s = 75.0
    cfg.fix_noise = cfg.fix_bias = 0.0
    cfg.tag_sensor.pixel_noise = cfg.tag_sensor.dropout = cfg.tag_sensor.blur_max_loss = 0.0
    cfg.link_loss_range = (0.0, 0.0)
    cfg.spawn_yaw_noise = 0.0
    successes = 0
    for target_index in range(4):
        cfg.fixed_spot_index = target_index  # keep reset heading consistent with the controlled pad
        env.reset()
        u.episode_length_buf[:] = 0
        u._spots[0, 0] = u._spot_table[target_index]
        u.taking_off[:] = False
        u._fix_bias[:] = 0
        u._link_loss[:] = 0
        pose = u._drones[0].data.root_state_w.clone()
        pose[:, :3] = u._spots[:, 0] + u._wp_offset + u.scene.env_origins
        pose[:, 7:] = 0
        u._drones[0].write_root_pose_to_sim(pose[:, :7])
        u._drones[0].write_root_velocity_to_sim(pose[:, 7:])
        u.pub_pos[:] = u._spots + u._wp_offset
        for tick in range(int(12 / u.step_dt)):
            _, _, done, _ = env.step(np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], np.float32))
            if tick < int(cfg.land_dwell_s / u.step_dt):
                assert not u.landing.any(), "Land sent before dwell completed"
            if done[0]:
                successes += int(u.outcome_success[0])
                break
        assert bool(done[0]) and bool(u.outcome_success[0]), f"Controlled landing failed for pad {target_index}"
    print(f"[ok] Gated, on-target touchdown on all {successes} landing pads")
    env.close()
    print("POLICY PIPELINE PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Print before shutdown: Kit can stall in close(), hiding the actual assertion.
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.app.close()
