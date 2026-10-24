# defend_3v2_ram6_s2 — FINAL (2026-10-24): 3 defenders vs 2 attackers, ramming rule, 31-dim observation, de-conflicted

DAgger 10 rounds (60 epochs, seed 2) from the `smart2` teacher with the final de-confliction defaults: teammate
repulsion 1.0 m / gain 2.5, 80 % vertical (lower index dives, higher climbs), no cruise bias, goalkeeper standby 1.0 m
above the site, defenders spawn 120 deg apart at 1.0 / 1.5 / 2.0 m. Ramming rule (spent defender), capture radius 0.20 m.
Trained on the Linux box 00:18-00:34, `WINDOWS_TRAINING.md` sections 12.10-12.11.

**Observation (31)**: the 23 of the original task, then per teammate (ascending index) [rel x, y, z (sim world, clipped
+-12 m), active flag (1 flying, 0 rammed/landed/lost)]. Action [vx, vy, vz, yaw_rate] in [-1, 1], world frame, 1.0 = 1 m/s.

| Eval (default difficulty, spent rule, waves off) | interception / breach | captures d0 / d1 / goalkeeper | pair separation p5 / median | episodes with a pair < 0.30 m |
|---|---|---|---|---|
| teacher, 0.20 m, 1000 ep | 99.9% / 0.1% | 1.00 / 1.00 / 0.00 | 0.51 / 1.11 m | 0.3% |
| **this clone, 0.20 m, 1000 ep** | **99.6% / 0.5%** | 0.99 / 0.99 / 0.00 | 0.50 / 1.12 m | **0.3%** |
| this clone, 0.15 m, 400 ep | 93.4% / 11.2% | 0.93 / 0.94 / 0.00 | 0.29 / 0.83 m | 6.0% |
| previous `defend_3v2_ram4_A` (29-dim, weak repulsion), 0.20 m | 99.8% / 0.3% | | 0.26 / 0.93 m | 7.8% |

Cloning fit: holdout mse 0.0015 / 0.0032 / 0.0021, direction cosine 0.998 / 0.994 / 0.995 (best so far).
Files: `agent_skrl_ippo.pt`, `defender_{0,1,2}.pt` (TorchScript, obs (1,31) -> (1,4)), `bc_dagger10.log`.

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 1000 env.respawn_waves=False env.capture_radius=0.2 \
  --checkpoint policies/defend_3v2_ram6_s2/agent_skrl_ippo.pt
```
