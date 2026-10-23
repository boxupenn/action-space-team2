# defend_3v2_smart2_dagger4 — AS-Defend-v0, 3 defenders vs 2 attackers, default difficulty

Behaviour-cloned (DAgger, 4 rounds, 1.92 M samples per agent) from the `smart2` scripted
teacher (`action_space_kit/tasks/direct/tello_defend/chase.py`: azimuth-rank target assignment,
lead pursuit, 3rd defender = goalkeeper on the attacker nearest the site). Trained 2026-10-23 on
the Linux box (`WINDOWS_TRAINING.md` section 12.7).

| File | What |
|---|---|
| `agent_skrl_ippo.pt` | skrl IPPO checkpoint (3 agents, obs 23 -> action 4, linear output, log_std -1.0); `evaluate.py --checkpoint`, `skrl/train.py --checkpoint` |
| `defender_{0,1,2}.pt` | TorchScript per defender from `export_skrl_policy.py --agent defender_N`: obs (1,23) -> action (1,4) in [-1,1] = [vx, vy, vz, yaw_rate] world frame |
| `bc_dagger4.log` | the cloning log |

Eval (default difficulty, 200 episodes, `env.respawn_waves=False`): **100.0% interception / 0.0% breach**
(mean actions), 99.8% / 0.5% sampled; waves on 404.7% / 1.2% (teacher 411% / 0.8%; chase 96.5% / 7.0%; zero 6.2% / 100%).

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 200 env.respawn_waves=False \
  --checkpoint policies/defend_3v2_smart2_dagger4/agent_skrl_ippo.pt
```
Reproduce: `python -u scripts/bc_warm_start.py --headless --num_envs 512 --steps 800 --epochs 40 --teacher smart2 --value_coef 0 --dagger_rounds 4 --out <ckpt>`
(defaults = default difficulty, waves on during collection). Deployment: rotate the world-frame velocity by yaw before `rc` (section 11.1).
