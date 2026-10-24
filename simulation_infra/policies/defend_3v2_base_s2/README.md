# defend_3v2_base_s2 — the arena of the organisers' figure (2026-10-24, afternoon)

3 defenders vs 2 attackers, 34-dim obs (own 9, attackers 2x7, teammates 2x[rel xyz, active], assignment feature 3),
cloned by DAgger (8 rounds, 800 steps x 512 envs per round, 40 epochs, seed 2) from the `smart2` teacher
(`teacher_assign="time"`, goalkeeper cover rule: alert 2.2 m, cover 1.2 m, force 1.7 m).

Arena (`tello_defend_env_cfg.py`, sim frame = tag z out of the wall, tag x right, up): room 6.6 x 3.5 m, the BASE a 13 cm
box 0.86 m out of the tag-10 wall at tag x 0.20; an attacker crossing the base line 1.30 m from the wall wins
(`breach_x`). Everybody takes off from the floor: attackers at the far (left) end, 4.6-6.3 m from the wall, climb at
0.5 m/s, are visible / rammable above 1.0 m, hover 1-4 s (tag search), then fly at 0.3-0.7 m/s at the base; defenders
take off at the three crosses (tag x, z) = (1.0, 1.37), (1.52, 0.50), (0.33, 0.99) and climb to 1.9 / 2.05 / 2.2 m.
Hit box 15 x 15 x 3 cm (half-extents), a ram takes both drones out, a cleared field ends the round (no waves).
Tello dynamics 2 m/s full stick, 3 m/s^2, 0.25 s lag. Cloning fit: holdout mse 0.0006, direction cosine 0.999.

| Eval (1000 episodes) | interception / breach | captures d0 / d1 / d2 | intercept time | pair sep min / p5 | pairs < 0.30 m |
|---|---|---|---|---|---|
| teacher | 99.8% / 0.1% | 0.99 / 0.18 / 0.83 | 6.5 s | 0.09 / 0.38 m | 3.8% |
| **clone** | **99.3% / 1.2%** | 0.99 / 0.19 / 0.81 | 6.7 s | 0.04 / 0.34 m | 4.0% |

The clone's breaches (12 / 1000) are all the same case: after the first ram the two remaining defenders crowd the second
attacker (min pair distance 0.04-0.24 m) and both miss; on the real drones the runner's 0.6 m separation guard holds the
higher-index one. Files: `agent_skrl_ippo.pt` (skrl checkpoint), `defender_{0,1,2}.pt` (TorchScript, obs (1,34) ->
(1,4), normalisation inside), `defender_{0,1,2}.npz` (the same nets as numpy arrays, for a laptop without torch),
`bc_dagger8.log`. Deployment: `drone_deployment_infra/swarm_ops/policies/defend.py` + `config/defend.yaml`.

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 1000 \
  --checkpoint policies/defend_3v2_base_s2/agent_skrl_ippo.pt
python -u scripts/record_clip.py --pilot policy --checkpoint policies/defend_3v2_base_s2/agent_skrl_ippo.pt \
  --frames 1200 --eye 3.3 -5.0 3.6 --lookat 2.6 1.0 1.0 --out outputs/videos/arena/student_base_rounds.mp4
```
