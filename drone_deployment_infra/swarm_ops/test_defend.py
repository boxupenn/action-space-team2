"""python -m pytest test_defend.py  (from swarm_ops/). The policy smoke tests need torch."""
import glob
import os

import numpy as np
import pytest

from policies.defend import (OBS_DIM, OBS_DIM_ACTIVE, OBS_DIM_LEGACY, OBS_DIM_MATES, OBS_WIDTHS, assignment_feature,
                             build_obs, face_yaw_rc, fit_obs, grounded, heading_deg, intercept_time, load_cfg,
                             load_nets, load_tag_map, net_obs_dim, run_net, sep_yield, tag_to_sim, sim_to_tag,
                             telemetry_pose_to_sim, world_to_rc)
from policies.defend import TelemetryAttackers


def unit_to(p):
    p = np.asarray(p, float)
    return list(p / np.linalg.norm(p)) + [0.0]


@pytest.mark.parametrize("yaw, want", [
    (0.0, "forward"),       # drone faces +x: attacker ahead -> rc b > 0
    (90.0, "right"),        # drone faces +y: attacker at +x is on its right -> rc a > 0
    (180.0, "back"),        # drone faces -x: attacker behind -> rc b < 0
])
def test_rc_points_at_attacker(yaw, want):
    attacker_rel = [3.0, 0.0, 0.0]                     # straight ahead in the world, at yaw 0
    a, b, c, d = world_to_rc(unit_to(attacker_rel), yaw)
    assert c == 0 and d == 0
    if want == "forward":
        assert b > 90 and abs(a) < 2
    elif want == "right":
        assert a > 90 and abs(b) < 2
    else:
        assert b < -90 and abs(a) < 2


def test_rc_cap_and_yaw_sign():
    assert world_to_rc([1, 1, 1, 1], 0.0, cap=0.5) == (-50, 50, 50, -50)   # vy+ = left -> a<0; CCW+ -> d<0
    assert world_to_rc([0, 0, 0, 0], 37.0) == (0, 0, 0, 0)


def test_frames_roundtrip_and_heading():
    p = [0.3, -0.5, 1.7]
    assert np.allclose(sim_to_tag(tag_to_sim(p)), p)
    assert np.allclose(tag_to_sim(p), [1.7, 0.3, -0.5])   # x out of wall, y right, z up
    assert abs(heading_deg(np.eye(3))) < 1e-9             # camera looking away from the wall
    R_face_wall = np.diag([-1.0, 1.0, -1.0])             # turned 180 about the vertical (tag y)
    assert abs(abs(heading_deg(R_face_wall)) - 180.0) < 1e-9


def test_face_yaw_turns_toward_site():
    site = np.zeros(3)
    own = np.array([2.0, 0.0, 0.0])                       # 2 m out of the wall, site at the wall
    assert face_yaw_rc(own, 180.0, site) == 0             # already facing it
    assert face_yaw_rc(own, 90.0, site) < 0               # needs CCW (to 180) -> rc d negative (clockwise+)
    assert face_yaw_rc(own, -90.0, site) > 0


OWN_P, OWN_V, SITE = np.array([2.0, 0.0, 1.0]), np.array([0.1, 0.0, 0.0]), np.array([0.5, 0.0, 1.2])
ATT_P = np.array([[5.0, 1.0, 1.5], [99.0, 99.0, 99.0]])
ATT_V = np.array([[-0.5, 0.0, 0.0], [1.0, 1.0, 1.0]])
MATES = [OWN_P + [1.0, -2.0, 0.5], OWN_P + [30.0, 0.0, 0.0]]          # ascending index, self excluded


def test_obs_layout_legacy_23():
    obs = build_obs(OWN_P, OWN_V, ATT_P, ATT_V, [1.0, 0.0], SITE)
    assert obs.shape == (OBS_DIM_LEGACY,) and obs.dtype == np.float32
    assert np.allclose(obs[0:3], [1.5, 0.0, -0.2]) and np.allclose(obs[3:6], OWN_V)
    assert np.allclose(obs[6:9], [0, 0, -1])
    assert np.allclose(obs[9:12], [3.0, 1.0, 0.5]) and np.allclose(obs[12:15], 0)      # dead -> zero, not clipped
    assert np.allclose(obs[15:18], [-0.6, 0.0, 0.0]) and np.allclose(obs[18:21], 0)
    assert np.allclose(obs[21:23], [1.0, 0.0])


def test_obs_layout_29_and_31():
    base = build_obs(OWN_P, OWN_V, ATT_P, ATT_V, [1.0, 0.0], SITE)
    obs29 = build_obs(OWN_P, OWN_V, ATT_P, ATT_V, [1.0, 0.0], SITE, MATES)
    assert obs29.shape == (OBS_DIM_MATES,) == (29,)
    assert np.allclose(obs29[:23], base)                               # prefix unchanged
    assert np.allclose(obs29[23:26], [1.0, -2.0, 0.5])
    assert np.allclose(obs29[26:29], [12.0, 0.0, 0.0])                 # clipped like rel_pos
    obs31 = build_obs(OWN_P, OWN_V, ATT_P, ATT_V, [1.0, 0.0], SITE, MATES, [True, False])
    assert obs31.shape == (OBS_DIM_ACTIVE,) == (31,)
    assert np.allclose(obs31[:23], base)
    assert np.allclose(obs31[23:27], [1.0, -2.0, 0.5, 1.0])            # rel x y z, active
    assert np.allclose(obs31[27:31], [12.0, 0.0, 0.0, 0.0])            # inactive keeps its position, flag 0
    obs34 = build_obs(OWN_P, OWN_V, ATT_P, ATT_V, [1.0, 0.0], SITE, MATES, [True, False], 0)
    assert obs34.shape == (OBS_DIM,) == (34,) and np.allclose(obs34[:31], obs31)
    assert obs34[31:].tolist() in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    # one 34-dim build serves every net width
    assert np.allclose(fit_obs(obs34, 34), obs34)
    assert np.allclose(fit_obs(obs34, 31), obs31)
    assert np.allclose(fit_obs(obs34, 29), obs29)
    assert np.allclose(fit_obs(obs34, 23), base)


def test_intercept_time():
    assert abs(intercept_time([3.0, 0.0, 0.0], [0.0, 0.0, 0.0]) - 3.0) < 1e-9          # still target: |r| / v_max
    assert abs(intercept_time([3.0, 0.0, 0.0], [-0.5, 0.0, 0.0]) - 2.0) < 1e-9         # closing head-on: 3 / 1.5
    assert abs(intercept_time([3.0, 0.0, 0.0], [0.5, 0.0, 0.0]) - 6.0) < 1e-9          # fleeing: 3 / 0.5
    assert intercept_time([3.0, 0.0, 0.0], [2.0, 0.0, 0.0]) > 1e5                      # fleeing faster than me: unreachable (as the sim: qa clamped to -1e-6)
    assert abs(intercept_time([3.0, 0.0, 0.0], [-2.0, 0.0, 0.0]) - 3.0) < 1e-9         # coming at me faster than me: no lead root (qa > 0) -> pursuit |r| / v_max, as the sim


def _obs31(own_v, rel, va, alive, mates, active):
    """Hand-assembled 31-dim obs in the sim layout."""
    rel, va = np.asarray(rel, float), np.asarray(va, float)
    rel_v = (va - own_v) * np.asarray(alive, float)[:, None]
    rel = rel * np.asarray(alive, float)[:, None]
    block = np.concatenate([np.append(m, a) for m, a in zip(mates, active)])
    return np.concatenate([np.zeros(3), own_v, [0, 0, -1], rel.ravel(), rel_v.ravel(), alive, block])


def test_assignment_feature_cases():
    own_v = np.zeros(3)
    # attacker 0 is 1 m from me, attacker 1 is 1 m from teammate 1; teammate 2 far away -> I chase 0, 2 is goalkeeper
    o = _obs31(own_v, [[1, 0, 0], [0, 4, 0]], [[0, 0, 0], [0, 0, 0]], [1, 1], [[0, 3, 0], [8, 8, 0]], [1, 1])
    assert assignment_feature(o, 0).tolist() == [1.0, 0.0, 0.0]
    # the same scene seen from defender 2 (teammates 0 and 1 relative to it): goalkeeper
    o2 = _obs31(own_v, [[-7, -8, 0], [-8, -4, 0]], [[0, 0, 0], [0, 0, 0]], [1, 1], [[-8, -8, 0], [-8, -5, 0]], [1, 1])
    assert assignment_feature(o2, 2).tolist() == [0.0, 0.0, 1.0]
    # my attacker is dead -> goalkeeper; teammate 1 inactive -> I take attacker 1 instead if that is cheapest
    o = _obs31(own_v, [[1, 0, 0], [0, 4, 0]], [[0, 0, 0], [0, 0, 0]], [0, 1], [[0, 3, 0], [8, 8, 0]], [1, 1])
    assert assignment_feature(o, 0).tolist() == [0.0, 0.0, 1.0]
    o = _obs31(own_v, [[1, 0, 0], [0, 4, 0]], [[0, 0, 0], [0, 0, 0]], [1, 1], [[0, 3, 0], [8, 8, 0]], [0, 1])
    assert assignment_feature(o, 0).tolist() == [1.0, 0.0, 0.0]


def test_assignment_matches_torch_reference():
    torch = pytest.importorskip("torch")
    import re
    src_path = os.path.join(os.path.dirname(__file__), "..", "..", "simulation_infra", "source", "action_space_kit",
                            "action_space_kit", "tasks", "direct", "tello_defend", "chase.py")
    if not os.path.isfile(src_path):
        pytest.skip("chase.py not in this checkout")
    src = open(src_path, encoding="utf-8").read()
    ns = {"torch": torch}
    for name in ("_intercept_time", "assignment_from_obs"):
        i = src.index(f"def {name}")
        j = src.find("\ndef ", i + 1)
        exec(src[i:j if j > 0 else len(src)], ns)      # the two pure-torch functions, without the Isaac imports
    rng = np.random.default_rng(0)
    E = 400
    o = np.zeros((E, 31))
    o[:, 3:6] = rng.uniform(-1, 1, (E, 3))
    o[:, 9:15] = rng.uniform(-6, 6, (E, 6))
    o[:, 15:21] = rng.uniform(-1.5, 1.5, (E, 6))
    o[:, 21:23] = (rng.uniform(0, 1, (E, 2)) > 0.2).astype(float)
    o[:, 23:31] = rng.uniform(-5, 5, (E, 8))
    o[:, [26, 30]] = (rng.uniform(0, 1, (E, 2)) > 0.2).astype(float)
    for i in range(3):
        tgt, gk = ns["assignment_from_obs"](torch.as_tensor(o, dtype=torch.float64), i, 2, 3, 1.0)
        for e in range(E):
            feat = assignment_feature(o[e], i)
            if bool(gk[e]):
                assert feat.tolist() == [0.0, 0.0, 1.0], (i, e)
            else:
                assert feat[int(tgt[e])] == 1.0 and feat[2] == 0.0, (i, e)


def test_sep_guard_higher_index_yields_with_hysteresis():
    me = np.array([1.0, 0.0, 0.0])
    near = [(0, me + [0.5, 0.0, 0.0], True), (2, me + [0.1, 0.0, 0.0], True)]   # index 0 close, 2 very close
    y, hit = sep_yield(1, me, near, False, 0.6, 0.8)
    assert y and hit[0] == 0                                           # only lower indices count
    assert sep_yield(0, me, near, False, 0.6, 0.8) == (False, None)    # defender 0 never yields
    assert sep_yield(1, me, [(0, me + [0.7, 0, 0])], False, 0.6, 0.8)[0] is False   # 0.7 > guard: no entry
    assert sep_yield(1, me, [(0, me + [0.7, 0, 0])], True, 0.6, 0.8)[0] is True     # 0.7 < release: keep holding
    assert sep_yield(1, me, [(0, me + [0.9, 0, 0])], True, 0.6, 0.8)[0] is False    # released
    assert sep_yield(1, me, [(0, me + [0.1, 0, 0], False)], False, 0.6, 0.8) == (False, None)   # inactive ignored


class _Status:
    """Stand-in for the umbrella Drone's status accessors."""
    num = 7

    def __init__(self, age=0.1, h=120.0, ground=False):
        self.age, self.h, self.ground = age, h, ground

    def status_age(self):
        return self.age

    def height_cm(self):
        return self.h

    def on_ground(self):
        return self.ground


def test_grounded_triggers():
    since = {}
    assert grounded(_Status(), 100.0, since) is False                  # flying: 1.2 m, fresh status
    assert grounded(_Status(age=3.5), 100.0, since) is True            # status stream stopped
    assert grounded(_Status(h=20.0), 100.0, since) is False            # just dropped low: not yet
    assert grounded(_Status(h=20.0), 101.1, since) is True             # low for > 1 s: on the floor
    assert grounded(_Status(h=120.0), 101.2, since) is False           # back up: timer cleared
    assert grounded(_Status(h=20.0), 101.3, since) is False            # ... so a new low starts over
    since.clear()
    assert grounded(_Status(h=None, ground=True), 110.0, since) is False   # Tello h<=5 with no tof: first read
    assert grounded(_Status(h=None, ground=True), 111.1, since) is True


def test_telemetry_pose_to_sim():
    ident = {10: (np.eye(3), np.zeros(3))}
    # drone 2 m in front of tag 10, 0.5 m right of it, 0.3 m below (telemetry: y down, z into the wall)
    p = telemetry_pose_to_sim({"tag": 10, "xyz": [0.5, 0.3, -2.0], "yaw": 0.0, "age": 0.1}, ident)
    assert np.allclose(p, [2.0, 0.5, -0.3])                           # sim: x out of wall, y right, z up
    assert telemetry_pose_to_sim({"tag": 8, "xyz": [0, 0, -1]}, ident) is None    # unmapped tag
    assert telemetry_pose_to_sim(None, ident) is None
    # a mapped side tag: the tag seen 0.7 m right of and level with tag 10 (map t), same orientation
    side = {4: (np.eye(3), np.array([0.7, 0.0, 0.0]))}
    p = telemetry_pose_to_sim({"tag": 4, "xyz": [0.0, 0.0, -1.5]}, side)
    assert np.allclose(p, [1.5, 0.7, 0.0])
    assert 10 in load_tag_map()                                        # the real map.yaml parses


def test_telemetry_datagram_v1():
    feed = TelemetryAttackers(port=None, tag_map={10: (np.eye(3), np.zeros(3))})
    msg = {"v": 1, "src": "sim", "seq": 7, "t": 0.0, "hz": 10.0, "drones": [
        {"name": "TELLO-5", "slot": 5, "team": "defense", "fly": True, "pose": {"tag": 10, "xyz": [0, 0, -1], "yaw": 0, "age": 0.0}},
        {"name": "TELLO-3", "slot": 2, "team": "attack", "fly": True, "pose": {"tag": 10, "xyz": [-1.0, 0.2, -3.0], "yaw": 5, "age": 0.1}},
        {"name": "TELLO-2", "slot": 1, "team": "attack", "fly": True, "pose": {"tag": 10, "xyz": [0.5, 0.3, -2.0], "yaw": 0, "age": 0.05},
         "truth": {"tag": 10, "xyz": [0.52, 0.31, -2.01], "yaw": 0}},
        {"name": "TELLO-4", "slot": 3, "team": "attack", "fly": False, "pose": None},
    ]}
    feed.ingest(msg, now=100.0)
    pos, vel, alive = feed.state(100.1)
    assert np.allclose(pos[0], [2.0, 0.5, -0.3]) and np.allclose(pos[1], [3.0, -1.0, -0.2])   # by slot: TELLO-2, TELLO-3
    assert alive.tolist() == [1.0, 1.0] and np.allclose(vel, 0)
    assert np.allclose(feed.truth[0], [2.01, 0.52, -0.31])
    msg["drones"][2]["pose"]["xyz"] = [0.5, 0.3, -1.5]                 # TELLO-2 moved 0.5 m toward the wall in 0.25 s
    msg["seq"] = 8
    feed.ingest(msg, now=100.25)
    pos, vel, alive = feed.state(100.3)
    assert np.allclose(pos[0], [1.5, 0.5, -0.3]) and np.allclose(vel[0], [-2.0, 0.0, 0.0])
    msg["drones"][2]["fly"], msg["seq"] = False, 10                    # landed / rammed -> not alive, position kept
    feed.ingest(msg, now=100.4)
    pos, vel, alive = feed.state(100.45)
    assert alive.tolist() == [0.0, 1.0] and np.allclose(pos[0], [1.5, 0.5, -0.3]) and feed.lost == 1
    assert feed.state(101.6)[2].tolist() == [0.0, 0.0]                 # feed silent > 1 s: nobody alive
    assert feed.ingest({"v": 2, "drones": []}, 102.0) is None          # other versions skipped
    named = TelemetryAttackers(port=None, names=["TELLO-3", "TELLO-4"], tag_map={10: (np.eye(3), np.zeros(3))})
    named.ingest(msg, 100.0)
    assert named.who == ["TELLO-3", "TELLO-4"] and named.state(100.1)[2].tolist() == [1.0, 0.0]


def _policy_dirs():
    cfg = load_cfg()
    root = os.path.dirname(cfg["policy_dir"])
    dirs = {cfg["policy_dir"]}
    dirs |= {d for d in glob.glob(os.path.join(root, "defend_3v2_ram*")) if os.path.isfile(os.path.join(d, "defender_0.pt"))}
    return sorted(dirs)


@pytest.mark.parametrize("policy_dir", _policy_dirs(), ids=os.path.basename)
def test_policy_shapes_and_closing(policy_dir):
    pytest.importorskip("torch")
    cfg = load_cfg()
    nets = load_nets(policy_dir, 3)
    assert all(net_obs_dim(n) in OBS_WIDTHS for n in nets)
    site = cfg["site"]
    own = site + np.array([1.5, 0.0, 0.0])
    mates = [own + [0.0, 1.0, 0.0], own + [0.0, -1.0, 0.0]]
    att_p = np.array([site + [4.0, 0.0, 0.3], site + [3.5, -2.0, 0.5]])
    att_v = np.array([[-0.6, 0.0, 0.0], [-0.5, 0.3, 0.0]])
    obs = build_obs(own, np.zeros(3), att_p, att_v, [1.0, 1.0], site, mates, [True, True], 0)
    assert obs.shape == (OBS_DIM,)
    closing = 0
    for net in nets:
        act = run_net(net, obs)
        assert act.shape == (4,) and np.all(np.abs(act) <= 1.0) and np.all(np.isfinite(act))
        to_att = att_p[0][:2] - own[:2]
        closing += float(np.dot(act[:2], to_att / np.linalg.norm(to_att))) > 0
    assert closing >= 1, "at least one defender should move toward the attackers"
