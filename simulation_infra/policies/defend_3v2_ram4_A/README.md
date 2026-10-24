# defend_3v2_ram4_A — 3 defenders vs 2 attackers, ramming rule, 29-dim observation

DAgger 8 rounds (60 epochs, 1.92 M + 8 x 0.38 M samples per agent) from the `smart2` teacher with teammate
repulsion (0.8 m, gain 1.5), evenly spaced defender spawns, ramming rule (spent defender, capture radius 0.20 m).
**Observation is 29-dim**: the 23 of the earlier policies + the two teammates' positions relative to me (sim
world frame, ascending defender index, clipped +-12 m). Trained 2026-10-23 23:05-23:18 on the Linux box.

Eval (default difficulty, spent rule, waves off, 1000 episodes, radius 0.20 m): **99.8% interception / 0.3% breach**,
captures per episode defender_0 0.99, defender_1 0.98, goalkeeper 0.02; defender separation p5 0.26 m, median 0.93 m,
episodes with a pair closer than 0.30 m: 7.8% (teacher: 100% / 0.1%, 3.8%). At 0.15 m: 88.4% / 19.2%.

Files: `agent_skrl_ippo.pt` (skrl IPPO), `defender_{0,1,2}.pt` (TorchScript, obs (1,29) -> action (1,4)), `bc_dagger8.log`.
