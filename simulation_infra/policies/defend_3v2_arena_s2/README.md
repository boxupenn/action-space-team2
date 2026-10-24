# defend_3v2_arena_s2 — arena rules (2026-10-24 afternoon), airborne start

3 defenders vs 2 attackers, 34-dim obs (own 9, attackers 2x7, teammates 2x[rel xyz, active], assignment feature 3),
cloned by DAgger (8 rounds, 800 steps x 512 envs per round, 40 epochs, seed 2) from the `smart2` teacher with
`teacher_assign="time"` plus the goalkeeper cover rule (`teacher_alert_radius` 2.2 m, `teacher_cover_m` 1.2 m).

Rules / geometry in the sim (`tello_defend_env_cfg.py`): hit box 15 x 15 x 3 cm (half-extents), a ram takes both
drones out, attackers enter 2-3 m in front of tag 10 within +-60 deg at 1.2-1.8 m, hover 1-4 s (take-off + tag
search), fly at 0.3-0.7 m/s to the 1.3 m arc (= breach); defenders launch at (0.2, 2.0) / (1.6, 1.6) / (-0.4, 1.6)
(tag frame x, z) at 1.9 / 2.05 / 2.2 m, Tello dynamics 2 m/s full stick, 3 m/s^2, 0.25 s lag, room box from
map.yaml. Cloning fit: holdout mse 0.0002, direction cosine 1.000 (all three).

| Eval (waves off, 1000 episodes unless noted) | interception / breach | captures d0 / d1 / d2 | intercept time | pair sep min / p5 | pairs < 0.30 m |
|---|---|---|---|---|---|
| teacher | 99.8% / 0.4% | 0.77 / 0.54 / 0.68 | 1.9 s | 0.30 / 0.55 m | 0.0% |
| **clone** | **99.8% / 0.5%** | 0.81 / 0.51 / 0.67 | 1.9 s | 0.11 / 0.53 m | 0.1% |
| clone, attackers move at once (no hover; untrained case), 500 ep | 72.3% / 39.4% | | 2.1 s | 0.05 / 0.43 m | 2.2% |

Files: `agent_skrl_ippo.pt` (skrl checkpoint), `defender_{0,1,2}.pt` (TorchScript, obs (1,34) -> (1,4), normalisation
inside), `defender_{0,1,2}.npz` (the same nets as numpy arrays for a laptop without torch), `bc_dagger8.log`.
Deployment: `drone_deployment_infra/swarm_ops/policies/defend.py` + `config/defend.yaml` (`policy_dir`).

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 1000 env.respawn_waves=False \
  --checkpoint policies/defend_3v2_arena_s2/agent_skrl_ippo.pt
python -u scripts/record_clip.py --pilot policy --checkpoint policies/defend_3v2_arena_s2/agent_skrl_ippo.pt \
  --frames 900 --eye 4.6 -3.4 3.0 --lookat 1.5 0.9 1.0 --out outputs/videos/arena/student_arena_waves.mp4
```
