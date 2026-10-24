# defend_3v2_ram_dagger4 — AS-Defend-v0, 3 defenders vs 2 attackers, RAMMING rule

Same recipe as `defend_3v2_smart2_dagger4` (DAgger 4 rounds from the `smart2` teacher) but trained under
the ramming rule of the real spec (2026-10-23 evening): `env.defender_spent_on_capture=True` (a defender
that rams is out for the rest of the wave), `env.capture_radius=0.2` (contact), goalkeeper (defender_2)
stands by 0.5 m above the site until an attacker is within 0.6 x spawn_radius, intercept point aimed
0.15 m above the attacker. Trained 2026-10-23 on the Linux box (`WINDOWS_TRAINING.md` section 12.9).

| File | What |
|---|---|
| `agent_skrl_ippo.pt` | skrl IPPO checkpoint (3 agents, obs 23 -> action 4, linear output, log_std -1.0) |
| `defender_{0,1,2}.pt` | TorchScript per defender (`export_skrl_policy.py --agent`): obs (1,23) -> [vx, vy, vz, yaw_rate] in [-1,1], world frame |
| `bc_dagger4.log` | cloning log (direction cosine 0.990-0.995) |

Eval (default difficulty, spent rule, 200 episodes, waves off): capture radius 0.20 m **99.5% / 0.5%**
(defender_0 0.98, defender_1 0.97, goalkeeper 0.04 captures per episode); 0.15 m 93.2% / 11.5% (goalkeeper 0.14:
it steps in when a ram misses); waves on 332% / 1.6% (teacher 330% / 1.6%). Teacher at 0.20: 100% / 0%; chase 86.8% / 23.5%.

```bash
python -u scripts/evaluate.py --task AS-Defend-v0 --headless --algorithm IPPO --episodes 200 env.respawn_waves=False env.capture_radius=0.2 \
  --checkpoint policies/defend_3v2_ram_dagger4/agent_skrl_ippo.pt
```
Open: sim speeds (defender 1.0 m/s, attackers 0.4-0.9 m/s) are kit assumptions; measure the real rc->m/s on site and re-clone with
`env.command.max_lin_speed=<measured> env.attacker_speed=[lo,hi]` if the ratio differs.
