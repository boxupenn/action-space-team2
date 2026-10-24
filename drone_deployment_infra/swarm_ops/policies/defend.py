"""defend: fly the sim-trained 3-defender policy (simulation_infra/policies/defend_3v2_smart2_dagger4)
on real Tellos. Config: config/defend.yaml.

Per defender, every tick: tag-10 position + heading from the drone's own camera (core/positional.py,
any mapped tag in view), velocity by differencing fixes, attackers from a scripted path or a UDP
feed -> the 23-dim sim observation -> defender_N.pt (TorchScript) -> world-frame [vx, vy, vz, yaw]
in [-1, 1] -> rotated by minus the drone's yaw into body-frame `rc a b c d`. Observation = 23 (own, gravity,
attackers) + per teammate 4 (position relative to me in the sim frame, active flag) + 3 (one-hot of my assigned
attacker from the shared intercept-time rule, goalkeeper flag) = 34; 31/29/23-dim nets get their prefix (fit_obs).
Separation guard: within sep_guard_m of a lower-index teammate the higher index holds horizontally
until sep_release_m (config/defend.yaml).

Frames. Tag frame T (map.yaml): x right along the wall, y up, z out of the tag face into the room.
Sim world frame S (z up): S = (T_z, T_x, T_y), i.e. sim x points out of the wall, sim y right along
the wall, sim z up. Heading = angle of the camera's forward axis in S, CCW from sim x, degrees;
a drone looking at the wall has heading 180.

Safety: no tag fix for FIX_FRESH_S -> hover (rc 0 0 0 0); none for NO_FIX_LAND_S -> land. Fence and
tof height limits zero any rc that pushes further out. The policy's yaw channel is replaced by a
controller that keeps the camera on the site (= the tags): it only sees where it is while a tag is in
view. Without --fly nothing is sent; obs and rc are printed every tick.

Offline demo, no drones, no video (kinematic defenders, scripted attackers):
    python defend_offline.py --seconds 12
Attackers from drone_gui's telemetry (the organisers' feed), still no defenders:
    python defend_offline.py --seconds 12 --attackers telemetry
"""
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque

import numpy as np

from .base import Policy, clip, hold_height

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "..", "..", "config", "defend.yaml")
sys.path.insert(0, os.path.join(HERE, "..", "..", "core"))

N_ATT, N_DEF = 2, 3
OBS_DIM_LEGACY = 23                         # defend_3v2_smart2_dagger4 nets: no teammate block
OBS_DIM_MATES = 23 + 3 * (N_DEF - 1)        # 29 (defend_3v2_ram4_*): + teammates' relative positions
OBS_DIM_ACTIVE = 23 + 4 * (N_DEF - 1)       # 31 (defend_3v2_ram5/6_*): + per teammate rel position and active flag
OBS_DIM = OBS_DIM_ACTIVE + N_ATT + 1        # 34 (defend_3v2_ram8_*): + one-hot of my assigned attacker, goalkeeper flag
OBS_WIDTHS = (OBS_DIM_LEGACY, OBS_DIM_MATES, OBS_DIM_ACTIVE, OBS_DIM)
V_MAX = 2.0                                 # m/s at |action| = 1 (sim TelloCommandCfg.max_lin_speed, arena nets): assignment rule
MATE_INACTIVE_S = 6.0                       # teammate with no fix this long (= NO_FIX_LAND_S) counts as inactive
MATE_LOW_CM, MATE_LOW_S = 30, 1.0           # ... or on the floor: height below this for this long (rammed)
MATE_STATUS_STALE_S = 3.0                   # ... or its status stream stopped (link lost / powered off)
OBS_CLIP_POS, OBS_CLIP_VEL = 12.0, 3.0      # as the sim env clips before normalisation
FIX_FRESH_S = 0.5                           # fix older than this: hover
NO_FIX_LAND_S = 6.0                         # no fix this long while airborne: land
PREFLIGHT_FIX_S = 8.0                       # preflight waits this long for video + a first fix
VEL_WINDOW_S = 0.4                          # ponytail: velocity = finite difference over this window;
                                            # a Kalman filter if the fix jitter (~cm) makes it too noisy
KP_YAW, MAX_RC_YAW, YAW_DEADBAND_DEG = 1.2, 60, 4.0
MIN_RC_XY = 14                              # a Tello ignores smaller horizontal rc (swarm_formation.py)
GRAVITY_B = np.array([0.0, 0.0, -1.0])      # level flight; sim's projected_gravity_b


# ----------------------------------------------------------------- pure functions (tested)
def tag_to_sim(p):
    """Tag-frame point/vector -> sim world frame (x out of the wall, y right, z up)."""
    p = np.asarray(p, float)
    return np.array([p[2], p[0], p[1]])


def sim_to_tag(p):
    p = np.asarray(p, float)
    return np.array([p[1], p[2], p[0]])


def heading_deg(R_cam_to_tag):
    """Heading in the sim frame (deg, CCW from sim x) of a camera whose cam->tag rotation is R."""
    f = tag_to_sim(np.asarray(R_cam_to_tag, float) @ np.array([0.0, 0.0, 1.0]))
    return math.degrees(math.atan2(f[1], f[0]))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def world_to_rc(action, yaw_deg, cap=1.0):
    """Sim action [vx, vy, vz, yaw_rate] (world frame, [-1, 1]; yaw_rate CCW+) -> rc (a, b, c, d):
    a right+, b forward+, c up+, d clockwise+, each in [-100*cap, 100*cap]."""
    vx, vy, vz, w = (max(-1.0, min(1.0, float(v))) for v in action)
    th = math.radians(yaw_deg)
    fwd = vx * math.cos(th) + vy * math.sin(th)
    left = -vx * math.sin(th) + vy * math.cos(th)
    lim = 100.0 * cap
    return tuple(clip(v * 100.0, lim) for v in (-left, fwd, vz, -w))


def kick_xy(a, b):
    """Scale a tiny non-zero horizontal rc up to MIN_RC_XY (else the Tello does not move)."""
    mag = math.hypot(a, b)
    if 0.5 < mag < MIN_RC_XY:
        a, b = int(round(a * MIN_RC_XY / mag)), int(round(b * MIN_RC_XY / mag))
    return a, b


def intercept_time(r, v, v_max=V_MAX):
    """Time to meet a target at relative position r moving at v when flying straight at v_max (lead
    pursuit): smallest positive root of |r + v t| = v_max t; pure-pursuit fallback |r| / v_max.
    Port of chase.py::_intercept_time."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    qa, qb, qc = float(v @ v) - v_max ** 2, 2.0 * float(r @ v), float(r @ r)
    disc = qb * qb - 4.0 * qa * qc
    t_pursuit = float(np.linalg.norm(r)) / v_max
    t_lead = (-qb - math.sqrt(max(disc, 0.0))) / (2.0 * min(qa, -1e-6))
    return t_lead if disc >= 0 and t_lead > 0 else t_pursuit


def assignment_feature(obs31, i, v_max=V_MAX, alert_radius=None, cover_m=None, alert_force=0.0):
    """[one-hot of the attacker assigned to defender i (N_ATT), goalkeeper flag] from the 31-dim obs
    alone: rebuild the (defenders x attackers) intercept-time table (own position = origin, teammates
    from their block), pick the ordered assignment with the smallest total (dead attacker / inactive
    defender = 1e3), ties -> first permutation. Port of chase.py::assignment_from_obs; every drone
    computes the same table from the same numbers, so no communication is needed."""
    import itertools
    o = np.asarray(obs31, float)
    n, D = N_ATT, N_DEF
    own_v = o[3:6]
    rel = o[9:9 + 3 * n].reshape(n, 3)
    va = o[9 + 3 * n:9 + 6 * n].reshape(n, 3) + own_v
    alive = o[9 + 6 * n:9 + 7 * n] > 0.5
    block = o[9 + 7 * n:9 + 7 * n + 4 * (D - 1)].reshape(D - 1, 4)
    def_pos, def_active = np.zeros((D, 3)), np.ones(D, bool)
    for k, j in enumerate(j for j in range(D) if j != i):
        def_pos[j], def_active[j] = block[k, :3], block[k, 3] > 0.5
    cost = np.full((D, n), 1.0e3)
    for j in range(D):
        for m in range(n):
            if alive[m] and def_active[j]:
                cost[j, m] = intercept_time(rel[m] - def_pos[j], va[m], v_max)
    perms = list(itertools.permutations(range(D), min(D, n)))
    totals = [sum(cost[p[m], m] for m in range(len(p))) for p in perms]
    chosen = perms[int(np.argmin(totals))]
    feat = np.zeros(n + 1)
    if i in chosen and alive[chosen.index(i)]:
        feat[chosen.index(i)] = 1.0
    elif alert_radius is not None and cover_m is not None:
        # goalkeeper dispatch (chase.py cover rule): the live attacker nearest the site, once inside
        # alert_radius of it and no active teammate within cover_m of it; else stand by
        site_rel = rel + o[0:3]
        site_d = np.where(alive, np.linalg.norm(site_rel[:, :2], axis=-1), 1.0e3)
        c = int(np.argmin(site_d))
        mate_d = [np.linalg.norm(rel[c] - def_pos[j]) for j in range(D) if j != i and def_active[j]]
        if site_d[c] < alert_radius and (not mate_d or min(mate_d) > cover_m or site_d[c] < alert_force):
            feat[c] = 1.0
        else:
            feat[n] = 1.0
    else:
        feat[n] = 1.0                           # goalkeeper
    return feat


def build_obs(own_p, own_v, att_p, att_v, alive, site, mates_p=(), mates_active=None, defender_index=None,
              alert_radius=None, cover_m=None, alert_force=0.0):
    """The sim observation (tello_defend_env.py::_get_observations), all in the sim frame: own pos - site
    (3), own vel (3), gravity_b (3), per attacker rel pos (3) and rel vel (3), alive (2), then per
    teammate (ascending defender index, self excluded) its position relative to me (3), clipped +-12 m,
    followed by its active flag (1 flying, 0 rammed/landed/lost) when mates_active is given, then (with
    defender_index) the assignment feature: one-hot of my attacker (N_ATT) and a goalkeeper flag.
    No mates_p -> the legacy 23-dim layout; mates_p alone -> 29; with mates_active -> 31; + index -> 34."""
    own_p = np.asarray(own_p, float)
    alive = np.asarray(alive, float).reshape(N_ATT, 1)
    rel_p = np.clip(np.asarray(att_p, float) - own_p, -OBS_CLIP_POS, OBS_CLIP_POS) * alive
    rel_v = np.clip(np.asarray(att_v, float) - own_v, -OBS_CLIP_VEL, OBS_CLIP_VEL) * alive
    mates = []
    for k, m in enumerate(mates_p):
        mates.append(np.clip(np.asarray(m, float) - own_p, -OBS_CLIP_POS, OBS_CLIP_POS))
        if mates_active is not None:
            mates.append(np.array([1.0 if mates_active[k] else 0.0]))
    obs = np.concatenate([own_p - site, own_v, GRAVITY_B, rel_p.ravel(), rel_v.ravel(), alive.ravel()]
                         + mates).astype(np.float32)
    assert obs.shape == (OBS_DIM_LEGACY + (4 if mates_active is not None else 3) * len(mates_p),)
    if defender_index is not None and mates_active is not None:
        obs = np.concatenate([obs, assignment_feature(obs, defender_index, V_MAX, alert_radius, cover_m, alert_force)]).astype(np.float32)
    return obs


def fit_obs(obs, width):
    """The full 34-dim obs (or 31) re-laid for a net of input `width`: 34 as is, 31 drops the assignment
    feature, 29 also drops the active flags, 23 the whole teammate block."""
    obs = np.asarray(obs, np.float32)
    if width == OBS_DIM and obs.size == OBS_DIM:
        return obs
    obs31 = obs[:OBS_DIM_ACTIVE]
    if width == OBS_DIM_ACTIVE:
        return obs31
    if width == OBS_DIM_MATES:
        block = obs31[OBS_DIM_LEGACY:].reshape(-1, 4)[:, :3].ravel()
        return np.concatenate([obs31[:OBS_DIM_LEGACY], block])
    return obs31[:OBS_DIM_LEGACY]


def grounded(dr, now, low_since):
    """A drone whose Tello status says it is down: stream stopped for MATE_STATUS_STALE_S, or on the ground /
    below MATE_LOW_CM for MATE_LOW_S. low_since: dict drone num -> when it first read low (updated here)."""
    if hasattr(dr, "status_age") and dr.status_age() > MATE_STATUS_STALE_S:
        return True
    h = dr.height_cm() if hasattr(dr, "height_cm") else None
    low = (hasattr(dr, "on_ground") and dr.on_ground() is True) or (h is not None and h < MATE_LOW_CM)
    if not low:
        low_since.pop(dr.num, None)
        return False
    return now - low_since.setdefault(dr.num, now) >= MATE_LOW_S


def sep_yield(i, own_p, others, yielding, guard_m, release_m):
    """Separation guard: others = [(index, p)] or [(index, p, active)]; inactive teammates are ignored.
    The HIGHER index yields (holds horizontal) while any lower-index teammate is within guard_m, until
    all are beyond release_m (hysteresis). Returns the new yielding flag and the closest offending
    (index, distance) or None."""
    lim = release_m if yielding else guard_m
    close = [(float(np.linalg.norm(np.asarray(o[1], float) - own_p)), o[0]) for o in others
             if o[0] < i and (len(o) < 3 or o[2])]
    close = [(dist, j) for dist, j in close if dist < lim]
    if not close:
        return False, None
    dist, j = min(close)
    return True, (j, dist)


def face_yaw_rc(own_p, yaw_deg, site):
    """rc d that turns the camera toward the site (the tag wall)."""
    want = math.degrees(math.atan2(site[1] - own_p[1], site[0] - own_p[0]))
    e = wrap_deg(want - yaw_deg)                       # CCW+ error
    return 0 if abs(e) < YAW_DEADBAND_DEG else clip(-KP_YAW * e, MAX_RC_YAW)   # rc d is clockwise+


def fence_action(action, own_p, lo, hi):
    """Zero every velocity component that would push a drone outside [lo, hi] further out (sim frame)."""
    a = list(action)
    for i in range(3):
        if (own_p[i] <= lo[i] and a[i] < 0) or (own_p[i] >= hi[i] and a[i] > 0):
            a[i] = 0.0
    return a


# ----------------------------------------------------------------- config / policy nets
def load_cfg(path=CFG_PATH):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["policy_dir"] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), cfg["policy_dir"]))
    cfg["site"] = tag_to_sim(cfg["site_xyz"])
    f = cfg["fence"]
    cfg["fence_lo"] = tag_to_sim([f["x"][0], f["y"][0], f["z"][0]])
    cfg["fence_hi"] = tag_to_sim([f["x"][1], f["y"][1], f["z"][1]])
    return cfg


class NpzNet:
    """defender_i.npz (export_skrl_policy.py --npz): the same MLP in numpy, for a laptop without torch.
    Keys: mean, std, W0, b0, ... (hidden layers), Wh, bh (head), act (elu|relu|tanh), squash (tanh or empty)."""

    def __init__(self, path):
        z = np.load(path)
        self.mean, self.std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
        self.layers = [(z[f"W{k}"], z[f"b{k}"]) for k in range(int(z["n_hidden"]))]
        self.head = (z["Wh"], z["bh"])
        self.act, self.squash = str(z["act"]), str(z["squash"])

    def __call__(self, obs):
        x = np.clip((np.asarray(obs, np.float32) - self.mean) / self.std, -5.0, 5.0)
        for W, b in self.layers:
            x = W @ x + b
            if self.act == "elu":
                x = np.where(x > 0, x, np.expm1(x))
            elif self.act == "relu":
                x = np.maximum(x, 0.0)
            else:
                x = np.tanh(x)
        y = self.head[0] @ x + self.head[1]
        if self.squash == "tanh":
            y = np.tanh(y)
        return np.clip(y, -1.0, 1.0)


def load_nets(policy_dir, n):
    try:
        import torch
    except ImportError:
        torch = None
    nets = []
    for i in range(n):
        pt, npz = (os.path.join(policy_dir, f"defender_{i}.{ext}") for ext in ("pt", "npz"))
        if torch is not None and os.path.isfile(pt):
            nets.append(torch.jit.load(pt, map_location="cpu").eval())
        elif os.path.isfile(npz):
            nets.append(NpzNet(npz))              # ponytail: numpy twin of the TorchScript, no torch needed
        else:
            raise FileNotFoundError(f"{pt} (needs torch) or {npz}")
    return nets


def net_obs_dim(net):
    """Input width of an exported net (its normalisation buffer `mean`); OBS_DIM if it has none."""
    try:
        return int(np.asarray(net.mean).size)
    except AttributeError:
        return OBS_DIM


def run_net(net, obs):
    """obs: the full observation (or already the net's width)."""
    obs = np.asarray(obs, np.float32)
    if obs.size != net_obs_dim(net):
        obs = fit_obs(obs, net_obs_dim(net))
    if isinstance(net, NpzNet):
        return net(obs)
    import torch
    with torch.no_grad():
        return net(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1))[0].numpy()


# ----------------------------------------------------------------- attacker feeds (sim frame out)
class ScriptedAttackers:
    """Each attacker flies straight from `start` at `speed` toward the site; down within
    capture_radius of a defender or on arrival; back at `start` after respawn_s (0 = stays down)."""

    def __init__(self, spec, site, capture_radius=0.5, respawn_s=0.0):
        self.site = site
        self.spec = list(spec)[:N_ATT]
        self.capture_r, self.respawn_s = float(capture_radius), float(respawn_s)
        self.t0 = None
        self.down_t = [None] * N_ATT                  # when it went down
        self.born = [float(s.get("delay", 0.0)) for s in self.spec] + [None] * (N_ATT - len(self.spec))

    def state(self, now, defenders_p):
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        pos, vel, alive = np.zeros((N_ATT, 3)), np.zeros((N_ATT, 3)), np.zeros(N_ATT)
        for i, s in enumerate(self.spec):
            if self.down_t[i] is not None:
                if self.respawn_s > 0 and now - self.down_t[i] >= self.respawn_s:
                    self.down_t[i], self.born[i] = None, t
                else:
                    continue
            if t < self.born[i]:
                continue
            start = tag_to_sim(s["start"])
            to_site = self.site - start
            dist = float(np.linalg.norm(to_site))
            u = to_site / (dist or 1.0)
            travelled = min(float(s["speed"]) * (t - self.born[i]), dist)
            pos[i], vel[i], alive[i] = start + u * travelled, u * float(s["speed"]), 1.0
            if travelled >= dist or any(np.linalg.norm(pos[i] - p) < self.capture_r for p in defenders_p):
                self.down_t[i], alive[i] = now, 0.0
        return pos, vel, alive


class UdpAttackers:
    """What the real tracker must provide: JSON datagrams {"attackers": [[x,y,z], [x,y,z]]} in the
    TAG frame (metres), null for an attacker that is down/unseen, at >= 5 Hz. Velocity is differenced
    here; an attacker not updated for STALE_S counts as not alive."""
    STALE_S = 1.0

    def __init__(self, port):
        self.hist = [deque(maxlen=20) for _ in range(N_ATT)]     # (t, p_sim)
        self.lock = threading.Lock()
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", int(port)))
        s.settimeout(0.5)
        self.sock = s
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
                pts = json.loads(data.decode())["attackers"]
            except (socket.timeout, ValueError, KeyError, UnicodeDecodeError):
                continue
            now = time.time()
            with self.lock:
                for i, p in enumerate(pts[:N_ATT]):
                    if p is not None and len(p) == 3:
                        self.hist[i].append((now, tag_to_sim(p)))

    def state(self, now, defenders_p=None):
        pos, vel, alive = np.zeros((N_ATT, 3)), np.zeros((N_ATT, 3)), np.zeros(N_ATT)
        with self.lock:
            for i, h in enumerate(self.hist):
                if h and now - h[-1][0] < self.STALE_S:
                    pos[i], vel[i], alive[i] = h[-1][1], velocity(h, now), 1.0
        return pos, vel, alive


class BusAttackers:
    """THE ARENA SOURCE (2026-10-24): the organisers fly the attackers with swarm_formation.py under the
    same umbrella, which publishes "formation/drone<N>" on the shared telemetry bus every tick:
    {state, p (map frame: x right, y up, z out of tag 10, m; None before the first fix), psi_deg,
    fix_age, height_cm, rc}. Attackers = `names` (e.g. ["drone4", "drone5"]) or, empty, the first
    N_ATT "formation/" topics by name. Alive while its topic is younger than STALE_S, has a position,
    its height reads above LOW_CM and its state is not a landing one; a rammed attacker (on the floor,
    link lost, or publishing stopped) keeps its last position with alive 0. Velocity is differenced."""
    STALE_S = 1.0
    LOW_CM = 30
    DOWN_STATES = ("LAND", "DONE", "EMERG")

    def __init__(self, bus, names=None):
        self.bus = bus
        self.names = list(names or [])[:N_ATT]
        self.hist = {}                                # name -> deque of (t, p_sim)
        self.flag = {}

    def pick(self):
        if len(self.names) < N_ATT:
            seen = sorted(k.split("/", 1)[1] for k in self.bus.since("formation/"))
            for n in seen:
                if n not in self.names and len(self.names) < N_ATT:
                    self.names.append(n)
        return self.names

    def state(self, now, defenders_p=None):
        pos, vel, alive = np.zeros((N_ATT, 3)), np.zeros((N_ATT, 3)), np.zeros(N_ATT)
        for i, n in enumerate(self.pick()):
            hit = self.bus.latest(f"formation/{n}")
            h = self.hist.setdefault(n, deque(maxlen=20))
            up = False
            if hit is not None:
                data, age = hit
                p = data.get("p")
                if p is not None and len(p) == 3:
                    t = time.time() - age                 # the bus ages against the wall clock
                    if not h or t > h[-1][0]:
                        h.append((t, tag_to_sim(p)))
                hc = data.get("height_cm")
                st = str(data.get("state") or "").upper()
                up = (age < self.STALE_S and p is not None and (hc is None or hc > self.LOW_CM)
                      and not any(st.startswith(s) for s in self.DOWN_STATES))
            self.flag[n] = up
            if h:
                pos[i], vel[i] = h[-1][1], velocity(h, now)   # last known even when down
                alive[i] = 1.0 if up else 0.0
        return pos, vel, alive

    def status(self):
        return f"bus attackers {self.names} alive {[self.flag.get(n) for n in self.names]}"


def load_tag_map(path=None):
    """config/map.yaml -> {tag id: (R, t)} with X_tag10 = R @ X_tag + t (TagMap.poses convention)."""
    import yaml
    path = path or os.path.join(HERE, "..", "..", "config", "map.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return {int(k): (np.asarray(v["R"], float), np.asarray(v["t"], float)) for k, v in data["tags"].items()}


def telemetry_pose_to_sim(pose, tag_map):
    """drone_telemetry v1 `pose` (camera in ITS tag's frame: x right, y DOWN, z INTO the wall, metres)
    -> sim frame via config/map.yaml. None if the tag is not mapped."""
    if pose is None or pose.get("tag") is None or int(pose["tag"]) not in tag_map:
        return None
    x, y, z = (float(v) for v in pose["xyz"])
    R, t = tag_map[int(pose["tag"])]
    p_tag = np.array([x, -y, -z])              # -> map tag frame: x right, y up, z out of the face
    return tag_to_sim(R @ p_tag + t)


class TelemetryAttackers:
    """Attackers from drone_gui's fleet telemetry (drone_deployment_infra/drone_telemetry/README.md, UDP
    JSON v1, default port 9100, one datagram per tick with every drone). Attackers = `attacker_names`
    if given, else the `team == "attack"` entries by slot, first N_ATT. Alive = fly and a pose from a
    mapped tag younger than STALE_S. Velocity is differenced here (`vel` is body-frame, no heading)."""
    STALE_S = 0.8                              # pose age; the GUI keeps the last pose and grows age
    FEED_STALE_S = 1.0                         # no datagram this long -> nobody alive

    def __init__(self, port=9100, names=None, tag_map=None):
        self.names = list(names or [])
        self.tag_map = tag_map if tag_map is not None else load_tag_map()
        self.hist = [deque(maxlen=20) for _ in range(N_ATT)]     # (t, p_sim)
        self.flag = [False] * N_ATT            # alive at the last datagram
        self.truth = [None] * N_ATT            # sim-source datagrams: true p_sim, for checking
        self.who = [None] * N_ATT
        self.last_t, self.seq, self.lost, self.unmapped = 0.0, None, 0, set()
        self.lock = threading.Lock()
        if port:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", int(port)))
            s.settimeout(0.5)
            self.sock = s
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65535)
                msg = json.loads(data.decode())
            except (socket.timeout, ValueError, UnicodeDecodeError):
                continue
            self.ingest(msg, time.time())

    def pick(self, drones):
        if self.names:
            by = {d.get("name"): d for d in drones}
            return [by.get(n) for n in self.names[:N_ATT]]
        att = sorted((d for d in drones if d.get("team") == "attack"), key=lambda d: d.get("slot", 99))
        return (att + [None] * N_ATT)[:N_ATT]

    def ingest(self, msg, now):
        """One v1 datagram (dict). Returns the list of (name, p_sim or None) it was read as."""
        if msg.get("v") != 1 or "drones" not in msg:
            return None
        if self.seq is not None and msg.get("seq", 0) > self.seq + 1:
            self.lost += msg["seq"] - self.seq - 1
        self.seq = msg.get("seq")
        out = []
        with self.lock:
            self.last_t = now
            for i, d in enumerate(self.pick(msg["drones"])):
                self.flag[i], self.truth[i], self.who[i] = False, None, d and d.get("name")
                if d is None:
                    out.append((None, None))
                    continue
                pose = d.get("pose")
                p = telemetry_pose_to_sim(pose, self.tag_map)
                if pose is not None and p is None and pose.get("tag") not in self.unmapped:
                    self.unmapped.add(pose.get("tag"))
                    print(f"[defend] telemetry: {d.get('name')} sees tag {pose.get('tag')} which is not in map.yaml"
                          " -> no position", flush=True)
                if p is not None:
                    self.hist[i].append((now, p))
                    self.flag[i] = bool(d.get("fly")) and float(pose.get("age", 99)) < self.STALE_S
                if d.get("truth"):
                    self.truth[i] = telemetry_pose_to_sim(d["truth"], self.tag_map)
                out.append((d.get("name"), p))
        return out

    def state(self, now, defenders_p=None):
        pos, vel, alive = np.zeros((N_ATT, 3)), np.zeros((N_ATT, 3)), np.zeros(N_ATT)
        with self.lock:
            fresh = now - self.last_t < self.FEED_STALE_S
            for i, h in enumerate(self.hist):
                if h:
                    pos[i], vel[i] = h[-1][1], velocity(h, now)   # last known even when down
                    alive[i] = 1.0 if (fresh and self.flag[i]) else 0.0
        return pos, vel, alive

    def status(self):
        with self.lock:
            age = time.time() - self.last_t if self.last_t else None
            return (f"telemetry seq {self.seq} lost {self.lost} age {age if age is None else round(age, 2)}s "
                    f"attackers {self.who} alive {self.flag}")


def velocity(hist, now):
    """Finite-difference velocity from (t, p) samples: newest vs the oldest within VEL_WINDOW_S."""
    t1, p1 = hist[-1]
    for t0, p0 in hist:
        if now - t0 <= VEL_WINDOW_S and t1 - t0 > 1e-3:
            return (p1 - p0) / (t1 - t0)
    return np.zeros(3)


# ----------------------------------------------------------------- own-position feeds
class TagFeed:
    """Position + heading of one drone from its own camera: H.264 over UDP 11111 (bound to the
    drone's adapter, like swarm_formation.py) -> core/positional.Positioner (any mapped tag)."""

    def __init__(self, d):
        from positional import Positioner       # core/; needs `pip install -e ../cal_and_map`
        self.d = d
        self.pos = Positioner(f"tello_{d.num}")
        self.hist = deque(maxlen=30)            # (t, p_sim)
        self.yaw = None
        self.lock = threading.Lock()
        self.frame, self.frame_id, self.frames, self.fixes = None, 0, 0, 0
        self.stop = threading.Event()
        self.started = False

    def start(self):
        if self.started:
            return True
        self.started = True
        threading.Thread(target=self._video, daemon=True).start()
        threading.Thread(target=self._perceive, daemon=True).start()
        for _ in range(3):
            if self._streamon() == "ok":
                return True
        return False

    def _streamon(self):
        try:
            return self.d.send("streamon", 5)
        except PermissionError:
            # the organisers' unmodified swarm_policies.py blocks every command but rc/takeoff/land:
            # streamon moves nothing, send it on the drone's own socket (same lock, same adapter)
            with self.d.lock:
                self.d.sock.settimeout(5)
                self.d.sock.sendto(b"streamon", ("192.168.10.1", 8889))
                try:
                    r = self.d.sock.recv(1024).decode(errors="ignore").strip().lower()
                except (socket.timeout, OSError):
                    r = "timeout"
            self.d.log(f"streamon (raw) -> {r}")
            return r

    def _video(self):
        import av
        vs = self.d._sock(11111)
        vs.settimeout(1.0)
        codec = av.CodecContext.create("h264", "r")
        while not self.stop.is_set():
            try:
                data = vs.recv(65536)
            except socket.timeout:
                continue
            try:
                for pkt in codec.parse(data):
                    for fr in codec.decode(pkt):
                        self.frame, self.frame_id = fr.to_ndarray(format="bgr24"), self.frame_id + 1
                        self.frames += 1
            except Exception:
                pass
        vs.close()

    def _perceive(self):
        last = 0
        while not self.stop.is_set():
            if self.frame is None or self.frame_id == last:
                time.sleep(0.005)
                continue
            last, img, t = self.frame_id, self.frame, time.time()
            loc = self.pos.locate(img)
            if loc is None:
                continue
            p = tag_to_sim(loc["xyz"])
            with self.lock:
                self.hist.append((t, p))
                self.yaw = heading_deg(loc["R"])
                self.fixes += 1

    def est(self, now):
        """(p_sim, v_sim, yaw_deg, age_s) of the latest fix, or None if there never was one."""
        with self.lock:
            if not self.hist:
                return None
            t, p = self.hist[-1]
            return p, velocity(self.hist, now), self.yaw, now - t

    def close(self):
        self.stop.set()


class KinematicFeed:
    """Offline stand-in: the drone is where the commanded world velocity took it (perfect tracking)."""

    def __init__(self, p0_tag, yaw_deg=180.0):
        self.p, self.v, self.yaw, self.t = tag_to_sim(p0_tag), np.zeros(3), yaw_deg, None

    def apply(self, action_world, yaw_rc, now):
        if self.t is not None:
            dt = now - self.t
            self.v = np.asarray(action_world[:3], float)
            self.p = self.p + self.v * dt
            self.yaw = wrap_deg(self.yaw - yaw_rc / 100.0 * 100.0 * dt)   # rc 100 ~ 100 deg/s, clockwise+
        self.t = now

    def est(self, now):
        return self.p, self.v, self.yaw, 0.0


# ----------------------------------------------------------------- the policy
class DefendPolicy(Policy):
    name = "defend"

    def __init__(self, cfg_path=CFG_PATH, feeds=None, attackers=None, policy_dir=None):
        self.cfg = load_cfg(cfg_path)
        if policy_dir:                           # --defend-dir / --policy-dir: another policy version
            self.cfg["policy_dir"] = os.path.abspath(policy_dir)
        self.feeds = feeds or {}                 # drone num -> feed with est(now)
        self.attackers = attackers
        self.nets = None
        self.idx = {}                            # drone num -> defender index
        self.last_fix_t = {}
        self.last_action = {}
        self.last_p = {}                         # drone num -> last known sim position (teammate block)
        self.yielding = {}                       # drone num -> separation guard active
        self.low_since = {}                      # drone num -> since when its height reads "on the floor"
        self.ready = {}                          # drone num -> reached its launch height (then the net flies it)

    def setup(self, drones):
        super().setup(drones)
        if len(drones) > 3:
            raise ValueError("defend: the shipped policy has 3 defenders; give at most 3 drones")
        self.idx = {d.num: i for i, d in enumerate(drones)}
        self.by_num = {d.num: d for d in drones}
        self.nets = load_nets(self.cfg["policy_dir"], len(drones))
        dims = [net_obs_dim(n) for n in self.nets]
        if len(drones) != N_DEF or any(dm not in OBS_WIDTHS for dm in dims):
            raise ValueError(f"defend: nets take obs dims {dims} for {len(drones)} drones; this runner builds "
                             f"{OBS_WIDTHS} for exactly {N_DEF} drones")
        if self.attackers is None:
            if self.cfg["attackers"] == "bus":
                self.attackers = BusAttackers(self.bus, self.cfg.get("attacker_names") or None)
            elif self.cfg["attackers"] == "udp":
                self.attackers = UdpAttackers(self.cfg["udp_port"])
            elif self.cfg["attackers"] == "telemetry":
                self.attackers = TelemetryAttackers(self.cfg.get("telemetry_port", 9100),
                                                    self.cfg.get("attacker_names") or None)
            else:
                self.attackers = ScriptedAttackers(self.cfg["scripted"], self.cfg["site"],
                                                   self.cfg.get("capture_radius", 0.5), self.cfg.get("respawn_s", 0.0))
        for d in drones:
            if d.num not in self.feeds:
                self.feeds[d.num] = TagFeed(d)       # map + camera calib must exist
            feed = self.feeds[d.num]
            if hasattr(feed, "start") and not feed.start():   # video + streamon now: the organisers' umbrella
                d.log("WARNING: streamon not confirmed")       # may never call our preflight() hook
        print(f"[defend] site(sim)={np.round(self.cfg['site'], 2).tolist()} attackers={self.cfg['attackers']} "
              f"nets={self.cfg['policy_dir']} (obs {dims}) sep guard {self.cfg.get('sep_guard_m', 0.6)}/"
              f"{self.cfg.get('sep_release_m', 0.8)} m", flush=True)

    def describe(self, d):
        return f"defend{self.idx.get(d.num, '?')}"

    def preflight(self, d):
        """streamon, then wait for video and (flying) a tag fix: the drone must face a mapped tag."""
        feed = self.feeds[d.num]
        if not hasattr(feed, "start"):
            return True
        if not feed.start() and not feed.frames:
            d.log("FAIL: streamon refused")
            return False
        t0 = time.time()
        while time.time() - t0 < PREFLIGHT_FIX_S and not feed.fixes:
            time.sleep(0.2)
        if not feed.frames:
            d.log("FAIL: no video frames")
            return False
        if not feed.fixes:
            d.log(f"no tag fix in {PREFLIGHT_FIX_S:.0f} s (point the camera at tag 10)"
                  + (" -> FAIL" if getattr(d, "fly", False) else " (dry-run: continuing)"))
            return not getattr(d, "fly", False)
        p, _, yaw, _ = feed.est(time.time())
        d.log(f"fix: sim p={np.round(p, 2).tolist()} yaw={yaw:.0f} ({feed.fixes} fixes, {feed.frames} frames)")
        return True

    def step(self, d, t):
        now = time.time()
        feed = self.feeds[d.num]
        est = feed.est(now)
        fly = getattr(d, "fly", False)
        if est is None or est[3] > FIX_FRESH_S:
            age = est[3] if est else float("inf")
            self.last_fix_t.setdefault(d.num, now)
            if now - self.last_fix_t[d.num] > NO_FIX_LAND_S:
                d.log(f"no tag fix for {NO_FIX_LAND_S:.0f} s")
                return None                      # land it
            if not fly:
                print(f"[{d.name}] defend{self.idx[d.num]} NO FIX (age {age:.1f}s, frames {getattr(feed, 'frames', '-')}) "
                      f"-> rc 0 0 0 0", flush=True)
            return (0, 0, 0, 0)
        own_p, own_v, yaw, _ = est
        self.last_fix_t[d.num] = now
        # The umbrella's takeoff leaves the drone at ~1 m; the nets were cloned from the launch heights
        # (above the attackers, ram from above). Climb there first, camera on the tags, then hand over.
        heights = self.cfg.get("launch_height_cm") or []
        i = self.idx[d.num]
        if fly and not self.ready.get(d.num) and i < len(heights):
            h = d.height_cm() if hasattr(d, "height_cm") else None
            if h is not None and abs(h - heights[i]) <= 10:
                self.ready[d.num] = True
                d.log(f"at launch height {h:.0f} cm -> policy")
            else:
                return (0, 0, hold_height(d, heights[i]), face_yaw_rc(own_p, yaw, self.cfg["site"]))
        if self.cfg["fence_hi"][0] < own_p[0] or own_p[0] < -0.1:
            d.log(f"WARNING: fix behind the wall / past the fence (sim x {own_p[0]:.2f}); map frame mirrored?")

        self.last_p[d.num] = own_p
        # teammates, ascending defender index: (index, position, active). Position = latest fix, else last
        # known, else far away (clipped to 12 m). Inactive = landed / landing / status stream stopped / on the
        # floor for 1 s / no fix for MATE_INACTIVE_S (it is still in the air until it lands itself):
        # a rammed drone keeps its last position in the obs with active 0, and its lost feed is no error.
        mates = []
        for n, j in sorted(self.idx.items(), key=lambda kv: kv[1]):
            if n == d.num:
                continue
            o = self.feeds[n].est(now)
            if o is not None:
                self.last_p[n] = o[0]
            dr = self.by_num[n]
            active = not (getattr(dr, "landed", False) or getattr(dr, "land_thread", None) is not None
                          or grounded(dr, now, self.low_since) or o is None or o[3] > MATE_INACTIVE_S)
            mates.append((j, self.last_p.get(n, own_p + np.array([OBS_CLIP_POS, 0.0, 0.0])), active))
        att_p, att_v, alive = self.attackers.state(now, [own_p] + [m[1] for m in mates if m[2]])
        obs = build_obs(own_p, own_v, att_p, att_v, alive, self.cfg["site"], [m[1] for m in mates],
                        [m[2] for m in mates], i, self.cfg.get("alert_radius"), self.cfg.get("cover_m"), self.cfg.get("alert_force", 0.0))
        action = run_net(self.nets[i], obs)
        action = fence_action(action, own_p, self.cfg["fence_lo"], self.cfg["fence_hi"])
        a, b, c, _ = world_to_rc(action, yaw, self.cfg["speed_cap"])
        a, b = kick_xy(a, b)
        was = self.yielding.get(d.num, False)
        guard, release = self.cfg.get("sep_guard_m", 0.6), self.cfg.get("sep_release_m", 0.8)
        self.yielding[d.num], hit = sep_yield(i, own_p, mates, was, guard, release)
        if self.yielding[d.num]:
            a = b = 0                            # hold horizontal; the lower-index teammate keeps flying
            if not was:
                d.log(f"SEP GUARD: defender{hit[0]} at {hit[1]:.2f} m -> holding until > {release} m")
        elif was:
            d.log("SEP GUARD released")
        h = d.height_cm() if hasattr(d, "height_cm") else None
        lo_cm, hi_cm = self.cfg["height_cm"]
        if h is not None and ((h <= lo_cm and c < 0) or (h >= hi_cm and c > 0)):
            c = 0
        # ponytail: the policy's yaw_rate is dropped; the camera must stay on the tags to know where
        # it is. The policy never observes yaw, so its world-frame velocity is unaffected.
        yaw_rc = face_yaw_rc(own_p, yaw, self.cfg["site"])
        rc = (a, b, c, yaw_rc)
        self.last_action[d.num] = (action, rc)
        if hasattr(feed, "apply"):
            feed.apply(action, yaw_rc, now)
        if not fly:
            truth = getattr(self.attackers, "truth", None)
            extra = (f" truth={[None if t is None else np.round(t, 1).tolist() for t in truth]}"
                     if truth and any(t is not None for t in truth) else "")
            print(f"[{d.name}] defend{self.idx[d.num]} p={np.round(own_p, 2).tolist()} yaw={yaw:.0f} "
                  f"att={[(np.round(p, 2).tolist() if al else (np.round(p, 2).tolist(), 'down') if np.any(p) else None) for p, al in zip(att_p, alive)]}{extra}\n"
                  f"    obs={[round(float(x), 2) for x in obs]}\n"
                  f"    action={[round(float(x), 2) for x in action]} -> rc {rc[0]:+d} {rc[1]:+d} {rc[2]:+d} {rc[3]:+d}",
                  flush=True)
        return rc


# ----------------------------------------------------------------- offline demo
class _FakeDrone:
    fly = landed = False

    def __init__(self, num):
        self.num, self.name = num, f"drone{num}"

    def height_cm(self):
        return None

    def log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)


def offline(seconds=12.0, hz=13.5, cfg_path=CFG_PATH, policy_dir=None, attackers=None, stagger=None):
    starts = [[-1.0, -0.3, 1.5], [0.0, -0.3, 1.5], [1.0, -0.3, 1.5]]   # tag frame, 1.5 m off the wall
    drones = [_FakeDrone(n) for n in (4, 5, 6)]
    pol = DefendPolicy(cfg_path, feeds={d.num: KinematicFeed(s) for d, s in zip(drones, starts)},
                       policy_dir=policy_dir)
    if attackers:
        pol.cfg["attackers"] = attackers
    if stagger is not None and len(pol.cfg["scripted"]) > 1:
        pol.cfg["scripted"][1]["delay"] = float(stagger)      # the arena case: the 2nd attacker launches late
    pol.setup(drones)
    att = pol.attackers
    t0, min_sep, prev = time.time(), float("inf"), None
    while time.time() - t0 < seconds:
        tick = time.time()
        live = [d for d in drones if not d.landed]
        # a defender that rammed (scripted attacker went down next to it) is spent: down, like the sim
        # (zeroed commands, active 0) and like a real Tello after a collision
        if hasattr(att, "spec"):
            pos, _, alive = att.state(tick, [pol.feeds[d.num].p for d in live])
            if prev is not None:
                for m in range(N_ATT):
                    if prev[1][m] and not alive[m] and live:
                        near = min(live, key=lambda d: np.linalg.norm(pol.feeds[d.num].p - prev[0][m]))
                        if np.linalg.norm(pol.feeds[near.num].p - prev[0][m]) < att.capture_r + 0.3:
                            near.landed = True
                            near.log(f"rammed attacker {m} -> spent (down)")
            prev = (pos.copy(), alive.copy())
            live = [d for d in drones if not d.landed]
        for d in live:
            pol.step(d, tick - t0)
        ps = [pol.feeds[d.num].p for d in live]
        min_sep = min([min_sep] + [float(np.linalg.norm(a - b)) for k, a in enumerate(ps) for b in ps[k + 1:]])
        time.sleep(max(0.0, 1.0 / hz - (time.time() - tick)))
    if hasattr(att, "status"):
        print(f"[defend] {att.status()}", flush=True)
        return
    ps = [pol.feeds[d.num].p for d in drones if not d.landed]
    spacing = min([float("inf")] + [float(np.linalg.norm(a - b)) for k, a in enumerate(ps) for b in ps[k + 1:]])
    print(f"[defend] offline done: attackers down {sum(t is not None for t in att.down_t)}/{len(att.spec)}, "
          f"spent {sum(d.landed for d in drones)}/3, "
          + (f"final spacing of the live ones >= {spacing:.2f} m, " if spacing < float("inf") else "")
          + f"closest live pair during the run {min_sep:.2f} m", flush=True)


