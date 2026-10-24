# defend_3v2_ram8_s2 — FINAL (2026-10-24 01:25): 100 % / 0 % over 2000 episodes

3 defenders vs 2 attackers, ramming rule (a defender that rams is out), capture radius 0.20 m (contact), default difficulty
(attackers 0.4-0.9 m/s from 6 m, defenders 1.0 m/s). DAgger 10 rounds (60 epochs, seed 2) from the `smart2` teacher with the
intercept-time assignment carried in the observation.

**Observation (34)**: 9 own [pos rel. site, world velocity, gravity in body frame] + per attacker (2) [rel pos 3, rel vel 3] + alive flags (2)
+ per teammate (2) [rel pos 3, active 1] + **assignment feature** [one-hot of my assigned attacker (2), goalkeeper flag (1)].
The assignment feature is computed on each drone from its own observation (`chase.py::assignment_from_obs`: rebuild the 3x2
lead-pursuit intercept-time table, pick the ordered pair of defenders with the smallest total time, leftover = goalkeeper); same
inputs -> same answer on every drone, no communication. Action [vx, vy, vz, yaw_rate] in [-1, 1], world frame, 1.0 = 1 m/s.

| Eval (spent rule, waves off) | interception / breach | captures d0 / d1 / d2 | intercept time | pair sep min / p5 | pairs < 0.30 m |
|---|---|---|---|---|---|
| teacher, 0.20 m, 2000 ep | 100.0% / 0.0% | 0.66 / 0.67 / 0.68 | 3.1 s | 0.65 / 1.81 m | 0.0% |
| **this clone, 0.20 m, 2000 ep** | **100.0% / 0.0%** | 0.67 / 0.65 / 0.68 | 3.1 s | **0.65 / 1.72 m** | **0.0%** |
| this clone, 0.15 m, 400 ep | 96.0% / 8.0% | | 3.7 s | 0.07 / 0.18 m | 10.8% |
| this clone, 0.20 m, waves on, 400 ep | 467% / 1.8% | 3.18 / 3.02 / 3.15 | | 0.05 / 0.36 m | 3.6% |
| chase baseline, 0.20 m | 86.8-94% / 10-23% | | | | 26-42% |

Cloning fit: holdout mse 0.0027 / 0.0014 / 0.0019, direction cosine 0.996 / 0.998 / 0.997.
Files: `agent_skrl_ippo.pt` (skrl IPPO, 3 agents), `defender_{0,1,2}.pt` (TorchScript, obs (1,34) -> (1,4)), `bc_dagger10.log`.

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 2000 env.respawn_waves=False env.capture_radius=0.2 \
  --checkpoint policies/defend_3v2_ram8_s2/agent_skrl_ippo.pt
```
Deployment: `drone_deployment_infra/swarm_ops/policies/defend.py` builds this observation (numpy port of the assignment, verified
against the torch reference) and rotates the world-frame velocity by the drone's heading before `rc`. Open: on-site rc->m/s calibration.
