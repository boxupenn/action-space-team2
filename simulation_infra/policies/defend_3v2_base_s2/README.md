# defend_3v2_base_s2 — the arena of the organisers' figure (2026-10-24, afternoon)

3 defenders vs 2 attackers, 34-dim obs (own 9, attackers 2x7, teammates 2x[rel xyz, active], assignment feature 3),
cloned by DAgger (10 rounds, 800 steps x 512 envs per round, 60 epochs, seed 2) from the `smart2` teacher
(`teacher_assign="time"`, goalkeeper cover rule: alert 2.2 m, cover 1.2 m, force 1.7 m).

Arena (`tello_defend_env_cfg.py`, sim frame = tag z out of the wall, tag x right, up): room 6.6 x 3.5 m, the BASE a 13 cm
box 0.86 m out of the tag-10 wall at tag x 0.20; an attacker crossing the base line 1.30 m from the wall wins
(`breach_x`). Everybody takes off from the floor: attackers at the far (left) end, 4.6-6.3 m from the wall, climb at
0.5 m/s, are visible / rammable above 1.0 m, hover 1-4 s (tag search), then fly at 0.3-0.7 m/s at the base; defenders
take off at the three crosses (tag x, z) = (1.0, 1.37), (1.52, 0.50), (0.33, 0.99) and climb to 1.45 / 1.5 / 1.55 m
(the runner's `launch_height_cm`); attackers are rammable above 0.5 m (runner `attacker_min_height_cm` 50).
Hit box 15 x 15 x 3 cm (half-extents), a ram takes both drones out, a cleared field ends the round (no waves).
Tello dynamics 2 m/s full stick, 3 m/s^2, 0.25 s lag. Cloning fit: holdout mse 0.0005, direction cosine 0.999.

| Eval (1000 episodes) | interception / breach | captures d0 / d1 / d2 | intercept time | pair sep min / p5 | pairs < 0.30 m |
|---|---|---|---|---|---|
| teacher (300 ep) | 100.0% / 0.0% | 1.00 / 0.19 / 0.81 | 4.7 s | 0.07 / 0.47 m | 1.0% |
| **clone** | **99.7% / 0.1%** | 0.99 / 0.18 / 0.82 | 4.8 s | 0.09 / 0.46 m | 1.2% |
| earlier clone, 2 m launch, DAgger-8 / 40 epochs | 99.3% / 1.2% | | 6.7 s | 0.04 / 0.34 m | 4.0% |
| earlier clone, 2 m launch, DAgger-10 / 60 epochs | 99.8% / 0.3% | | 6.4 s | 0.06 / 0.37 m | 3.2% |

The one breach in 1000 is the second attacker leaking after the first ram; on the real drones the runner's 0.6 m separation
guard also holds the higher-index defender when two crowd one attacker. Files: `agent_skrl_ippo.pt` (skrl checkpoint), `defender_{0,1,2}.pt` (TorchScript, obs (1,34) ->
(1,4), normalisation inside), `defender_{0,1,2}.npz` (the same nets as numpy arrays, for a laptop without torch),
`bc_dagger10.log`. Deployment: `drone_deployment_infra/swarm_ops/policies/defend.py` + `config/defend.yaml`.

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 1000 \
  --checkpoint policies/defend_3v2_base_s2/agent_skrl_ippo.pt
python -u scripts/record_clip.py --pilot policy --checkpoint policies/defend_3v2_base_s2/agent_skrl_ippo.pt \
  --frames 1200 --eye 3.3 -5.0 3.6 --lookat 2.6 1.0 1.0 --out outputs/videos/arena/student_base_rounds.mp4
```
