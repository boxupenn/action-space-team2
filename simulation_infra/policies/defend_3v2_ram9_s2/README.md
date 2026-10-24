# defend_3v2_ram9_s2 — FINAL (2026-10-24 01:56): staggered launches + goalkeeper standby ring

Same as `defend_3v2_ram8_s2` (34-dim obs with the on-board assignment feature, ramming rule, 0.20 m) plus two robustness changes
for the arena: the sim now launches the second attacker 0-3 s late (invisible until launched: the "one attacker in the air, two
goalkeepers" case), and goalkeeper i holds on a 0.8 m ring around the site at azimuth 2*pi*i/3 (1.0 m up) instead of all on the
same point. DAgger 10 rounds (60 epochs, seed 2); cloning fit holdout mse 0.0020 / 0.0010 / 0.0021, cosine 0.997 / 0.999 / 0.997.

| Eval (default difficulty, spent rule, waves off, 0.20 m) | interception / breach | captures d0 / d1 / d2 | intercept time | pair sep min / p5 | pairs < 0.30 m |
|---|---|---|---|---|---|
| teacher, attackers together, 1000 ep | 100.0% / 0.0% | 0.62 / 0.67 / 0.70 | 3.1 s | 0.76 / 1.81 m | 0.0% |
| teacher, 2nd attacker 0-3 s late, 2000 ep | 100.0% / 0.0% | 0.67 / 0.66 / 0.67 | 4.0 s | 0.25 / 0.51 m | 1.3% |
| **clone, attackers together, 400 ep** | **100.0% / 0.0%** | 0.67 / 0.63 / 0.70 | 3.0 s | **0.67 / 1.86 m** | **0.0%** |
| **clone, 2nd attacker 0-3 s late, 2000 ep** | **100.0% / 0.0%** | 0.69 / 0.65 / 0.66 | 4.0 s | 0.16 / 0.48 m | 2.2% |
| clone, 0.15 m, 400 ep | 97.2% / 4.8% | | 4.5 s | 0.12 / 0.27 m | 6.8% |
| clone, waves on, 400 ep | 359% / 0.2% | 2.44 / 2.40 / 2.34 | | 0.06 / 0.33 m | 3.6% |

Residual: with a late second attacker a few episodes still bring two live defenders within 0.30 m (centre distance; 2.2 %); the
runner's 0.6 m separation guard covers those. Files: `agent_skrl_ippo.pt`, `defender_{0,1,2}.pt` (TorchScript, obs (1,34) -> (1,4)), `bc_dagger10.log`.

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 2000 env.respawn_waves=False env.capture_radius=0.2 \
  --checkpoint policies/defend_3v2_ram9_s2/agent_skrl_ippo.pt            # add env.attacker_launch_delay=[0.0,0.0] for simultaneous launches
```
