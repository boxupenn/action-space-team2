"""Self-check for the 23-dim defend export: chase.assign_latched latches a pursuit across a teammate's
ram, and the Obs23Policy wrapper scripts to TorchScript and keeps that latch across calls.
No Kit app needed: python -u scripts/test_defend_obs23.py  (pass: SELFTEST PASSED)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_skrl_policy as ex  # noqa: E402

chase = ex._load_chase()
assign_latched = chase.assign_latched
n = 2


def obs(own, atk):  # own site-relative position, attackers [(rel xyz) | None] -> (1, 23)
    rel, vel, alive = [], [], []
    for a in atk:
        rel += list(a) if a else [0.0, 0.0, 0.0]
        vel += [0.0, 0.0, 0.0]
        alive.append(1.0 if a else 0.0)
    return torch.tensor([list(own) + [0.0] * 3 + [0.0, 0.0, -1.0] + rel + vel + alive])


def main():
    me = (2.5, 0.0, 0.0)
    both = obs(me, [(1.0, 3.0, 0.0), (1.0, -3.0, 0.0)])  # slot 0 at +az, slot 1 at -az (around the site)
    prev = torch.full((1,), -1, dtype=torch.long)
    # whole wave visible: defender 0 takes the lower azimuth (slot 1), defender 1 slot 0, defender 2 goalkeeper
    t0, g0, p0 = assign_latched(both, 0, n, prev)
    t1, g1, p1 = assign_latched(both, 1, n, prev)
    t2, g2, p2 = assign_latched(both, 2, n, prev)
    assert (t0.item(), g0.item(), p0.item()) == (1, False, 1), (t0, g0, p0)
    assert (t1.item(), g1.item(), p1.item()) == (0, False, 0), (t1, g1, p1)
    assert (t2.item(), g2.item(), p2.item()) == (0, True, -1), (t2, g2, p2)
    # slot 1 rammed by defender 0: defender 1 keeps slot 0 (latched), defender 0 becomes goalkeeper
    one = obs(me, [(1.0, 3.0, 0.0), None])
    t1b, g1b, p1b = assign_latched(one, 1, n, p1)
    assert (t1b.item(), g1b.item(), p1b.item()) == (0, False, 0), (t1b, g1b, p1b)
    t0b, g0b, p0b = assign_latched(one, 0, n, p0)
    assert (g0b.item(), p0b.item()) == (True, -1), (t0b, g0b, p0b)
    # un-latched (fresh) defender 0 facing a lone attacker takes it; defender 1 waits
    t0c, g0c, _ = assign_latched(one, 0, n, prev)
    assert (t0c.item(), g0c.item()) == (0, False)
    _, g1c, _ = assign_latched(one, 1, n, prev)
    assert g1c.item() is True
    # field cleared: everyone unassigned
    none = obs(me, [None, None])
    _, gz, pz = assign_latched(none, 1, n, p1b)
    assert (gz.item(), pz.item()) == (True, -1)

    # TorchScript wrapper: a 26 -> 4 linear stand-in for the cloned network, latch survives across calls
    sd = {"net_container.0.weight": torch.zeros(8, 26), "net_container.0.bias": torch.zeros(8),
          "policy_layer.weight": torch.zeros(4, 8), "policy_layer.bias": torch.zeros(4)}
    sd["policy_layer.weight"][0, 0] = 1.0
    sd["net_container.0.weight"][0, 23] = 1.0  # action[0] = relu(one-hot slot 0)
    sd["net_container.0.weight"][1, 24] = 1.0
    sd["policy_layer.weight"][1, 1] = 1.0  # action[1] = one-hot slot 1
    sd["net_container.0.weight"][2, 25] = 1.0
    sd["policy_layer.weight"][2, 2] = 1.0  # action[2] = goalkeeper flag
    ex.assign_latched = assign_latched  # what main() does before scripting
    m = torch.jit.script(ex.Obs23Policy(ex.SkrlPolicy(sd, None, "relu"), 1, n))
    a = m(both)
    assert a[0, 0].item() == 1.0 and a[0, 2].item() == 0.0, a  # defender 1 -> slot 0
    a = m(one)
    assert a[0, 0].item() == 1.0 and a[0, 2].item() == 0.0, a  # still slot 0 after the teammate's ram
    a = m(none)
    assert a[0, 2].item() == 1.0, a  # cleared -> goalkeeper flag
    a = m(torch.cat([both, both]))  # batch != 1 runs un-latched and leaves the buffer alone
    assert a.shape == (2, 4) and m.prev.item() == -1
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
