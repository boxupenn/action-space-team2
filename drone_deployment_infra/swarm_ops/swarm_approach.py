#!/usr/bin/env python3
"""Swarm formation in front of AprilTag 10, using a pre-built tag MAP.

Each drone localizes from ANY mapped tag (map.yaml + tags.yaml), so it knows where it
is in tag 10's frame. All drones get non-crossing slots on an arc in front of tag 10,
fly there keeping safe spacing, hold, and land.
SAFETY RULE: a drone only moves sideways while it sees at least one mapped tag. If it
loses all tags it freezes in place (keeps looking); after 3 s without a tag it lands
straight down where it is. Searching = rotating in place only.

DEFAULT IS DRY-RUN: no takeoff / no motion commands; shows what it WOULD send.
Add --fly to fly.

Needs, next to this script (or pass paths):
  dongles.json         adapter <-> drone map
  map.yaml, tags.yaml  tag map (tag 10 = origin) and tag sizes
  camera/tello_<n>.yaml  camera calibration per drone (K + dist)
  libs: numpy opencv-python pupil-apriltags av pyyaml

  python swarm_approach.py --only 4                 # dry-run: check localization
  python swarm_approach.py --only 1,3,5 --fly

Keys (camera windows): q = land all, SPACE = EMERGENCY motor stop all. Ctrl+C = land all.
"""
import argparse
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from itertools import permutations

import cv2
import numpy as np
import yaml

# ======================= SETTINGS (edit here) =======================
ARC_RADIUS_M = 1.0           # distance from tag-10 center to each slot
ARC_RADIUS_MAX_M = 2.0       # grow up to this if N drones don't fit
MIN_SPACING_M = 0.7          # slot-to-slot spacing
MAX_VIEW_ANGLE_DEG = 55      # slots stay within this angle from tag-10 normal
HEIGHT_OFFSET_M = 0.0        # slot height relative to tag-10 center (+ = higher)

KP_POS = 40.0                # rc units per meter of position error
KP_YAW = 1.0                 # rc units per degree of yaw error
MAX_RC_XY = 15               # speed caps (rc range is -100..100); low = less blur
MAX_RC_Z = 20
MAX_RC_YAW = 25
DEADBAND_POS_M = 0.05
DEADBAND_YAW_DEG = 3.0
POSE_ALPHA = 0.6             # low-pass on pose (1 = no filtering)

D_SAFE_M = 0.6               # below this: slow approach + push apart (< MIN_SPACING)
D_STOP_M = 0.4               # below this: no approach, only push apart
K_REP = 60.0                 # rc units per meter inside D_SAFE
LOST_DRIFT_MPS = 0.15        # un-fixed drone's no-go bubble grows this fast
MIN_TAG_DIST_M = 0.6         # never get closer to tag 10 than this

TOL_POS_M = 0.10             # "in slot" tolerance
TOL_YAW_DEG = 5.0
LOCKED_NOFIX_S = 8.0         # LOCKED may turn/climb IN PLACE without a tag this long
ALIGN_H_M = 0.10             # LOCKED: at slot height within this...
ALIGN_YAW_DEG = 8.0          # ...and facing tag 10 within this -> may start approach
SETTLE_S = 2.0               # all in tolerance this long -> HOLD

# --- localization (relaxed detection, guarded by consistency checks) ---
MIN_DECISION_MARGIN = 15     # decode confidence floor (guards against misread ids)
MIN_TAG_SIDE_PX = 15
MAX_SINGLE_ERR_PX = 3.0      # single-tag reprojection error limit
JOINT_MAX_RMS_PX = 3.0       # multi-tag joint fit must agree with the map this well
JUMP_MAX_M = 0.5             # new fix this far from prediction = suspicious...
JUMP_MAX_DEG = 25.0
JUMP_ACCEPT_AFTER_S = 1.5    # ...unless it persists this long (then re-initialize)
FIX_FRESH_S = 0.5            # a fix older than this is not "seen now"
EST_MAX_AGE_S = 12.0         # predicted pose usable this long without a fix

# --- search ---
SEARCH_HEIGHT_CM = 160       # height above floor while searching
KP_HEIGHT = 0.5              # rc units per cm of height error while searching
CLIMB_TIMEOUT_S = 6.0
LOOK_S = 2.0                 # at search height: hold still and look first
SEARCH_YAW_RC = 25           # rotation speed during each search turn
SEARCH_STEP_DEG = 20         # turn this much, then...
SEARCH_LOOK_S = 0.8          # ...hold still this long to look (no motion blur)
GLIMPSE_HOLD_S = 2.0         # any glimpse of a mapped tag: freeze up to this long
SEARCH_MAX_TURN_DEG = 400    # searched this much without any tag -> land all
WAIT_ALL_S = 6.0             # localized drones wait for the others at most this long

# --- no tag = no movement ---
LOST_LAND_S = 3.0            # localized drone sees no tag this long -> it freezes, then lands

MIN_BATTERY_FLY = 40
LAND_BATTERY = 15
LOOP_HZ = 20
RES = (960, 720)
# ====================================================================

DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
NOFLY = {"command", "streamon", "streamoff", "battery?", "height?"}
RC_RE = re.compile(r"^rc (-?\d+) (-?\d+) (-?\d+) (-?\d+)$")
TURN_RE = re.compile(r"^(cw|ccw) (\d+)$")
GO_RE = re.compile(r"^go (-?\d+) (-?\d+) (-?\d+) (\d+)$")


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ======================= FRAMES =======================
# Internal frame for every tag (and the map origin, tag 10): x right, y DOWN,
# z INTO the tag (OpenCV style). "In front of tag 10" = negative z.
# map.yaml uses x right, y UP, z OUT of the tag; C converts between them.
C_FLIP = np.diag([1.0, -1.0, -1.0])


def load_tag_files(map_path, tags_path):
    """Returns (origin_id, {id: (R_10k, t_10k)} in internal frames, {id: size_m})."""
    with open(tags_path) as f:
        tcfg = yaml.safe_load(f)
    sizes = {int(k): float(v) / 1000.0 for k, v in (tcfg.get("sizes_mm") or {}).items()
             if v is not None}
    with open(map_path) as f:
        m = yaml.safe_load(f)
    origin = int(m.get("origin_tag", tcfg.get("origin_tag", 10)))
    frame = str(m.get("frame", ""))
    if "y up" not in frame or "out of" not in frame:
        print(f"WARNING: map frame says {frame!r}; expected 'x right, y up, z out of the "
              f"tag face'. Check the conversion.")
    poses = {}
    for k, v in (m.get("tags") or {}).items():
        k = int(k)
        if k not in sizes:
            print(f"WARNING: tag {k} is in the map but has no size in {tags_path}; skipped")
            continue
        R = np.array(v["R"], np.float64).reshape(3, 3)
        U, _, Vt = np.linalg.svd(R)                       # re-orthonormalize
        R = U @ Vt
        t = np.array(v["t"], np.float64).reshape(3)
        poses[k] = (C_FLIP @ R @ C_FLIP, C_FLIP @ t)      # map frame -> internal frame
    if origin not in poses:
        raise SystemExit(f"origin tag {origin} missing from {map_path}")
    return origin, poses, sizes


def obj_points(size):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def rotz90(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def drone_in_frame(R, t):
    """Camera pose given camera<-frame (R, t): position p and yaw psi (deg) in that
    frame. psi=0 faces into the tag (+z); positive = turned clockwise seen from above."""
    p = -R.T @ t
    f = R.T @ np.array([0.0, 0.0, 1.0])
    return p, math.degrees(math.atan2(f[0], f[2]))


def chain_to_origin(R_ck, t_ck, R_0k, t_0k):
    """camera<-tag_k and tag_k pose in origin -> camera<-origin."""
    R_c0 = R_ck @ R_0k.T
    return R_c0, t_ck - R_c0 @ t_0k


def tag_candidates(corners, K, D, size):
    """All plausible camera<-tag poses for one tag, independent of the detector's
    corner order: 4 start corners x 2 windings x 2 IPPE solutions; keep front-face
    views with low error; fix in-plane 90-deg ambiguity (tag upright: y ~ down).
    Returns [(err, R, t)] sorted by error, near-duplicates removed."""
    obj = obj_points(size)
    out = []
    for wind in (corners, corners[::-1]):
        for k in range(4):
            pts = np.roll(wind, k, axis=0).reshape(-1, 1, 2).astype(np.float64)
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, pts, K, D,
                                                        flags=cv2.SOLVEPNP_IPPE)
            errs = np.array(errs).reshape(-1)
            for i in range(n):
                if not np.isfinite(errs[i]) or errs[i] > MAX_SINGLE_ERR_PX:
                    continue
                R, _ = cv2.Rodrigues(rvecs[i])
                t = tvecs[i].reshape(3)
                if R[2, 2] <= 0 or t[2] <= 0:
                    continue
                best = max(range(4), key=lambda j: (R @ rotz90(j))[1, 1])
                out.append((float(errs[i]), R @ rotz90(best), t))
    out.sort(key=lambda c: c[0])
    uniq = []
    for c in out:
        if all(np.linalg.norm(c[2] - u[2]) > 0.02 or
               math.degrees(math.acos(max(-1, min(1, (np.trace(c[1].T @ u[1]) - 1) / 2)))) > 3
               for u in uniq):
            uniq.append(c)
    return uniq


def match_corners(corners, R, t, K, D, size):
    """Reorder detected corners to match obj_points(size) for pose (R, t)."""
    proj, _ = cv2.projectPoints(obj_points(size), cv2.Rodrigues(R)[0], t, K, D)
    proj = proj.reshape(-1, 2)
    idx = [int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in proj]
    if len(set(idx)) != 4:
        return None
    return corners[idx]


def localize(dets, K, D, tag_map, sizes, prior):
    """dets: [(id, corners)]. prior: (p, psi) predicted in origin frame or None.
    Returns (p, psi, info) in origin frame, or (None, None, reason)."""
    per_tag = []
    for tid, c in dets:
        cands = tag_candidates(c, K, D, sizes[tid])
        if not cands:
            continue
        R_0k, t_0k = tag_map[tid]
        e0 = cands[0][0]
        near = [cc for cc in cands if cc[0] < 2.0 * e0 + 0.5]
        poses = []
        for err, R, t in near:
            R_c0, t_c0 = chain_to_origin(R, t, R_0k, t_0k)
            p, psi = drone_in_frame(R_c0, t_c0)
            poses.append((err, R, t, p, psi, R_c0))
        if prior is not None:        # temporal disambiguation
            pp, ppsi = prior
            pick = min(poses, key=lambda q: np.linalg.norm(q[3] - pp) + abs(wrap_deg(q[4] - ppsi)) / 30.0)
        elif len(poses) > 1:         # gravity: tag upright -> its normal is horizontal
            pick = min(poses, key=lambda q: abs(q[1][1, 2]))
        else:
            pick = poses[0]
        area = cv2.contourArea(c.astype(np.float32))
        per_tag.append((tid, c, pick, area))
    if not per_tag:
        return None, None, "no valid tag pose"
    best = max(per_tag, key=lambda q: q[3])         # biggest tag in the image
    _, _, (err, R, t, p, psi, R_c0), _ = best
    info = {"tags": sorted(q[0] for q in per_tag), "mode": "single", "rms": err}
    if len(per_tag) >= 2:                            # joint fit through the map
        obj_all, img_all = [], []
        for tid, c, (e, Rk, tk, *_), _ in per_tag:
            mc = match_corners(c, Rk, tk, K, D, sizes[tid])
            if mc is None:
                continue
            R_0k, t_0k = tag_map[tid]
            obj_all.append((R_0k @ obj_points(sizes[tid]).T).T + t_0k)
            img_all.append(mc)
        if len(obj_all) >= 2:
            obj_all = np.vstack(obj_all)
            img_all = np.vstack(img_all)
            _, t_c0 = chain_to_origin(R, t, *tag_map[best[0]])
            ok, rv, tv = cv2.solvePnP(obj_all, img_all, K, D, cv2.Rodrigues(R_c0)[0],
                                      t_c0.reshape(3, 1).copy(), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                proj, _ = cv2.projectPoints(obj_all, rv, tv, K, D)
                rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_all) ** 2, 1))))
                if rms <= JOINT_MAX_RMS_PX:
                    Rj, _ = cv2.Rodrigues(rv)
                    p, psi = drone_in_frame(Rj, tv.reshape(3))
                    info.update(mode="joint", rms=rms)
                else:
                    info.update(mode="single (map disagrees)", rms=rms)
    return p, psi, info


# ======================= SHARED: tag 10 height above the floor =======================
class Tag10Height:
    """Tag 10's center height above the floor (m), from drone height sensor + tag fixes.
    Fixes that include tag 10 itself are trusted first. Shared by all drones."""
    def __init__(self):
        self.lock = threading.Lock()
        self.with10, self.other = [], []

    def add(self, floor_h_m, p_y, tags):
        v = floor_h_m + p_y                 # internal y is DOWN: drone below center -> p_y > 0
        if not 0.2 < v < 3.0:
            return
        with self.lock:
            lst = self.with10 if 10 in tags else self.other
            lst.append(v)
            del lst[:-60]

    def get(self):
        with self.lock:
            if len(self.with10) >= 3:
                return float(np.median(self.with10)), "tag10"
            if len(self.other) >= 3:
                return float(np.median(self.other)), "other tags"
        return None, None


TAG10_H = Tag10Height()


# ======================= FORMATION (pure) =======================
def make_slots(n):
    """Slots on an arc around tag 10. Returns (list of (pos, yaw_deg), radius)."""
    r = ARC_RADIUS_M
    while r <= ARC_RADIUS_MAX_M + 1e-9:
        if n == 1:
            phis = [0.0]
        else:
            dphi = 2 * math.asin(min(1.0, MIN_SPACING_M / (2 * r)))
            if (n - 1) * dphi / 2 > math.radians(MAX_VIEW_ANGLE_DEG):
                r += 0.1
                continue
            phis = [(i - (n - 1) / 2) * dphi for i in range(n)]
        return [(np.array([r * math.sin(f), -HEIGHT_OFFSET_M, -r * math.cos(f)]),
                 math.degrees(-f)) for f in phis], r
    raise ValueError(f"{n} drones don't fit (spacing {MIN_SPACING_M} m, "
                     f"angle {MAX_VIEW_ANGLE_DEG} deg, radius <= {ARC_RADIUS_MAX_M} m)")


def bearing(p):
    return math.atan2(p[0], -p[2])


def match_slots(positions, slots):
    """Min-total-length drone->slot matching: straight paths never cross."""
    nums = list(positions)
    best, best_cost = {}, float("inf")
    for perm in permutations(range(len(slots)), len(nums)):
        cost = sum(math.hypot(slots[k][0][0] - positions[n][0], slots[k][0][2] - positions[n][2])
                   for n, k in zip(nums, perm))
        if cost < best_cost:
            best, best_cost = dict(zip(nums, perm)), cost
    return best


def control(p, psi, slot, others):
    """rc toward slot with separation. others: [(p_other, extra_radius)]."""
    target, _ = slot
    e = target - p
    yaw_t = math.degrees(math.atan2(-p[0], -p[2]))     # face tag 10 center from here
    e_yaw = wrap_deg(yaw_t - psi)
    if np.linalg.norm(e) < DEADBAND_POS_M:
        e = np.zeros(3)
    v = KP_POS * e
    scale, rep = 1.0, np.zeros(3)
    for q, extra in others:
        dvec = np.array([p[0] - q[0], 0.0, p[2] - q[2]])
        d = float(np.linalg.norm(dvec))
        d_safe, d_stop = D_SAFE_M + extra, D_STOP_M + extra
        if d < d_safe:
            u = dvec / d if d > 1e-6 else np.array([1.0, 0, 0])
            rep += K_REP * (d_safe - d) * u
            scale = min(scale, max(0.0, (d - d_stop) / (d_safe - d_stop)))
    v = v * scale + rep
    dist_h = math.hypot(p[0], p[2])
    if 1e-6 < dist_h < MIN_TAG_DIST_M:
        u = np.array([p[0], 0.0, p[2]]) / dist_h
        inward = -float(v @ u)
        if inward > 0:
            v = v + inward * u
        v = v + 10.0 * u
    nxz = math.hypot(v[0], v[2])
    if nxz > MAX_RC_XY:
        v[0], v[2] = v[0] * MAX_RC_XY / nxz, v[2] * MAX_RC_XY / nxz
    v[1] = float(np.clip(v[1], -MAX_RC_Z, MAX_RC_Z))
    s, c_ = math.sin(math.radians(psi)), math.cos(math.radians(psi))
    fwd, right, up = v[0] * s + v[2] * c_, v[0] * c_ - v[2] * s, -v[1]
    yaw = 0.0 if abs(e_yaw) < DEADBAND_YAW_DEG else float(np.clip(KP_YAW * e_yaw, -MAX_RC_YAW, MAX_RC_YAW))
    return tuple(int(round(x)) for x in (right, fwd, up, yaw)), float(np.linalg.norm(target - p)), e_yaw


def body_displacement(p, psi, target):
    """Origin-frame displacement -> Tello 'go' axes (cm): x forward, y LEFT, z up."""
    d = target - p
    s, c_ = math.sin(math.radians(psi)), math.cos(math.radians(psi))
    fwd, right, up = d[0] * s + d[2] * c_, d[0] * c_ - d[2] * s, -d[1]
    return int(round(fwd * 100)), int(round(-right * 100)), int(round(up * 100))


def seg_point_dist(a, b, q):
    """Horizontal distance from point q to segment a-b."""
    a2, b2, q2 = np.array([a[0], a[2]]), np.array([b[0], b[2]]), np.array([q[0], q[2]])
    ab = b2 - a2
    L = float(ab @ ab)
    u = 0.0 if L < 1e-9 else max(0.0, min(1.0, float((q2 - a2) @ ab) / L))
    return float(np.linalg.norm(a2 + u * ab - q2))


# ======================= DRONE I/O =======================
def iface_ip(iface):
    out = subprocess.run(["ip", "-j", "-4", "addr", "show", iface],
                         capture_output=True, text=True).stdout or "[]"
    for dev in json.loads(out):
        for a in dev.get("addr_info", []):
            if a.get("local", "").startswith("192.168.10."):
                return a["local"]
    return None


def active_ssid(iface):
    out = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list",
                          "ifname", iface, "--rescan", "no"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:].replace("\\:", ":")
    return None


def load_camera(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    K = c.get("K", c.get("camera_matrix"))
    D = c.get("dist", c.get("dist_coeffs", c.get("D")))
    if isinstance(K, dict):
        K = K.get("data")
    if isinstance(D, dict):
        D = D.get("data")
    if K is None or D is None:
        raise ValueError(f"{path}: no K/dist found (keys: {list(c)})")
    K = np.array(K, np.float64).reshape(3, 3)
    D = np.array(D, np.float64).reshape(-1)
    res = c.get("resolution")
    if res is not None:
        if isinstance(res, str):
            res = [int(x) for x in re.split(r"[x, ]+", res) if x]
        elif isinstance(res, dict):
            res = [res.get("width"), res.get("height")]
        if list(res)[:2] != list(RES):
            raise ValueError(f"{path}: calibrated at {res}, stream is {RES}")
    return K, D


class Drone:
    def __init__(self, num, iface, ssid, K, D, fly, tag_map, sizes, det_opts):
        self.num, self.iface, self.ssid, self.K, self.D = num, iface, ssid, K, D
        self.name, self.fly = f"drone{num}", fly
        self.tag_map, self.sizes, self.det_opts = tag_map, sizes, det_opts
        self.sock = self._sock(9000 + num)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None; self.frame_t = 0.0; self.frame_id = 0; self.frames = 0
        self.packets = 0
        self.overlay = None
        self.battery = None
        self.airborne = False
        self.landed = False
        self.landing = False        # land command issued (no more rc)
        # status stream
        self.st, self.st_t = {}, 0.0
        self.last_yaw, self.yaw_acc = None, 0.0
        # localization
        self.fix = None             # (p, psi, t, imu_yaw_at_fix) filtered
        self.fix_info = {}
        self.reason = "no frames"
        self.glimpse_t = 0.0
        self.jump_since = None
        self.imu_sign, self.imu_evidence, self.imu_learned = 1.0, 0.0, False
        self.aligned = False
        self.was_fix = False
        # mission state
        self.dstate = "SEARCH"      # SEARCH -> LOCKED -> APPROACH; no tag 3 s -> land
        self.sub, self.sub_t = "CLIMB", None
        self.turn_start, self.search_turned = 0.0, 0.0
        self.busy = False           # a blocking cw/go is running (no rc)
        self.blind_leg = None       # (p0, p1, t0, duration)
        self.blind_note = ""
        self.reacq_t = 0.0

    def _sock(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, self.iface.encode())
        s.bind(("", port))
        return s

    def log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)

    def _check(self, cmd):
        if cmd in NOFLY:
            return
        if self.fly:
            if cmd in ("takeoff", "land", "emergency"):
                return
            m = RC_RE.match(cmd)
            if m and all(abs(int(x)) <= 100 for x in m.groups()):
                return
        raise PermissionError(f"blocked command {cmd!r} (fly={self.fly})")

    def send(self, cmd, timeout=10, quiet=False):
        self._check(cmd)
        with self.lock:
            self.sock.setblocking(False)
            try:
                while True:
                    self.sock.recv(1024)
            except (BlockingIOError, OSError):
                pass
            self.sock.settimeout(timeout)
            self.sock.sendto(cmd.encode(), DRONE_ADDR)
            try:
                r = self.sock.recv(1024).decode(errors="ignore").strip().lower()
            except socket.timeout:
                r = "timeout"
        if not quiet:
            self.log(f"{cmd} -> {r}")
        return r

    def send_noreply(self, cmd):
        self._check(cmd)
        self.sock.sendto(cmd.encode(), DRONE_ADDR)

    def on_ground(self):
        if time.time() - self.st_t > 1.0:
            return None
        try:
            return int(self.st.get("h", 99)) <= 5   # height since takeoff, cm
        except ValueError:
            return None

    def land(self):
        if not self.airborne or self.landed:
            return
        try:
            self.send_noreply("rc 0 0 0 0")
            for attempt in range(3):
                self.send_noreply("land")
                t0 = time.time()
                while time.time() - t0 < 8:           # wait for touchdown (status h ~ 0)
                    if self.on_ground():
                        self.landed = True
                        self.log("landed")
                        return
                    time.sleep(0.2)
                if self.on_ground() is None and self.send("land", 10) == "ok":
                    self.landed = True                 # no status stream: trust the reply
                    return
            self.log("WARNING: land NOT confirmed - catch this drone!")
        except OSError as ex:
            self.log(f"land error: {ex}")

    # ---- status stream ----
    def state_loop(self):
        ss = self._sock(8890)
        ss.settimeout(1.0)
        while not self.stop.is_set():
            try:
                data = ss.recv(2048).decode(errors="ignore")
            except socket.timeout:
                continue
            vals = dict(kv.split(":", 1) for kv in data.strip().split(";") if ":" in kv)
            self.st, self.st_t = vals, time.time()
            try:
                y = float(vals["yaw"])
                if self.last_yaw is not None:
                    self.yaw_acc += abs(wrap_deg(y - self.last_yaw))
                self.last_yaw = y
            except (KeyError, ValueError):
                pass
        ss.close()

    def imu_yaw(self):
        if time.time() - self.st_t > 1.0:
            return None
        try:
            return float(self.st["yaw"])
        except (KeyError, ValueError):
            return None

    def height_cm(self):
        if time.time() - self.st_t > 1.0:
            return None
        try:
            tof = int(self.st.get("tof", -1))
        except ValueError:
            tof = -1
        if 10 <= tof <= 400:
            return tof
        try:
            h = int(self.st.get("h", -1))           # height since takeoff (fallback)
        except ValueError:
            return None
        return h if 0 < h <= 400 else None

    # ---- pose estimate: last fix + IMU heading (+ blind-leg progress) ----
    def est(self, now):
        """(p, psi, age_s, source) in origin frame, or None."""
        if self.fix is None:
            return None
        p, psi, t, y0 = self.fix
        age = now - t
        y = self.imu_yaw()
        if y is not None and y0 is not None:
            psi = wrap_deg(psi + self.imu_sign * wrap_deg(y - y0))
        src = "FIX" if age < FIX_FRESH_S else "PRED"
        if self.blind_leg is not None:
            p0, p1, t0, dur = self.blind_leg
            u = min(1.0, max(0.0, (now - t0) / dur)) if dur > 0 else 1.0
            p = p0 + u * (p1 - p0)
            src = "BLIND"
        return p, psi, age, src

    def has_fix(self, now):
        return self.fix is not None and now - self.fix[2] < FIX_FRESH_S

    # ---- video + perception ----
    def video_loop(self):
        import av
        vs = self._sock(11111)
        vs.settimeout(1.0)
        codec = av.CodecContext.create("h264", "r")
        while not self.stop.is_set():
            try:
                data = vs.recv(65536)
            except socket.timeout:
                continue
            self.packets += 1
            try:
                for pkt in codec.parse(data):
                    for fr in codec.decode(pkt):
                        self.frame = fr.to_ndarray(format="bgr24")
                        self.frame_t = time.time()
                        self.frame_id += 1
                        self.frames += 1
            except Exception:
                pass
        vs.close()

    def perception_loop(self):
        from pupil_apriltags import Detector
        det = Detector(families="tag36h11", nthreads=1, **self.det_opts)
        last = 0
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.005)
                continue
            last, img, t = self.frame_id, self.frame.copy(), self.frame_t
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            good = []
            for d in det.detect(gray):
                if d.tag_id not in self.tag_map:
                    continue                      # only mapped ids (guards misreads)
                c = np.array(d.corners, np.float64)
                side = min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
                ok = d.decision_margin >= MIN_DECISION_MARGIN and side >= MIN_TAG_SIDE_PX
                self.glimpse_t = t
                if ok:
                    good.append((d.tag_id, c))
                else:
                    self.reason = f"tag {d.tag_id} weak (margin {d.decision_margin:.0f}, {side:.0f}px)"
                cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 255, 0) if ok else (0, 0, 255), 2)
                cv2.putText(img, str(d.tag_id), tuple(int(v) for v in c.mean(0)), 0, 0.8,
                            (0, 255, 0) if ok else (0, 0, 255), 2)
            if good:
                prior = None
                if self.fix is not None and t - self.fix[2] < 2.0:
                    e = self.est(t)
                    prior = (self.fix[0], e[1] if self.imu_learned else self.fix[1])
                p, psi, info = localize(good, self.K, self.D, self.tag_map, self.sizes, prior)
                if p is None:
                    self.reason = info
                else:
                    self._accept(p, psi, info, t)
            else:
                weak = [r for r in [self.reason] if r.startswith("tag ") and "weak" in r]
                self.reason = weak[0] if weak and self.glimpse_t == t else "no mapped tag in view"
            self.overlay = img

    def _accept(self, p, psi, info, t):
        if self.fix is not None and t - self.fix[2] < 1.0:
            if np.linalg.norm(p - self.fix[0]) > JUMP_MAX_M:          # position-only check
                self.jump_since = self.jump_since or t
                if t - self.jump_since < JUMP_ACCEPT_AFTER_S:
                    self.reason = "fix rejected (jumped >0.5 m vs last fix)"
                    return
            self.jump_since = None
        else:
            self.jump_since = None
        y = self.imu_yaw()
        if self.fix is not None and y is not None and self.fix[3] is not None:
            d_tag = wrap_deg(psi - self.fix[1])          # learn IMU yaw sign
            d_imu = wrap_deg(y - self.fix[3])
            if abs(d_imu) > 3 and t - self.fix[2] < 2.0:
                self.imu_evidence += d_tag * d_imu
                if abs(self.imu_evidence) > 300:
                    self.imu_sign = 1.0 if self.imu_evidence > 0 else -1.0
                    self.imu_learned = True
        if self.fix is not None and t - self.fix[2] < FIX_FRESH_S and self.blind_leg is None:
            p0, s0, _, _ = self.fix
            p = POSE_ALPHA * p + (1 - POSE_ALPHA) * p0
            psi = wrap_deg(s0 + POSE_ALPHA * wrap_deg(psi - s0))
        self.fix, self.fix_info = (p, psi, t, y), info
        h = self.height_cm()
        if h is not None:
            TAG10_H.add(h / 100.0, float(p[1]), info.get("tags", []))
        self.reason = f"tags {info['tags']} {info['mode']} {info['rms']:.1f}px"

    def battery_loop(self):
        while not self.stop.wait(5):
            if self.busy:
                continue
            b = self.send("battery?", 3, quiet=True)
            if b.isdigit():
                self.battery = int(b)

    def close(self):
        self.stop.set()
        try:
            self.send("streamoff", 3, quiet=True)
        except Exception:
            pass
        self.sock.close()


# ======================= SETUP =======================
def load_drones(cfg, only, cam_dir, fly, tag_map, sizes, det_opts):
    with open(cfg) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("dongles", [])
    entries = [e for e in data if e.get("ssid")]
    if only:
        entries = [e for e in entries if int(e["number"]) in only]
        missing = only - {int(e["number"]) for e in entries}
        if missing:
            sys.exit(f"No config/ssid for drone(s) {sorted(missing)} in {cfg}")
    if not entries:
        sys.exit("No drones selected.")
    drones, bad = [], []
    for e in entries:
        n = int(e["number"])
        path = os.path.join(cam_dir, f"tello_{n}.yaml")
        if not os.path.exists(path):
            bad.append(f"drone{n}: missing calibration {path}")
            continue
        try:
            K, D = load_camera(path)
        except Exception as ex:
            bad.append(f"drone{n}: {ex}")
            continue
        drones.append(Drone(n, e["iface"], e["ssid"], K, D, fly, tag_map, sizes, det_opts))
    if bad:
        sys.exit("Cannot run:\n  " + "\n  ".join(bad))
    return drones


def parallel(drones, fn):
    res, ths = {}, []
    for d in drones:
        def w(d=d):
            try:
                res[d.num] = fn(d)
            except Exception as ex:
                d.log(f"error: {ex}")
                res[d.num] = False
        t = threading.Thread(target=w, daemon=True)
        t.start()
        ths.append(t)
    for t in ths:
        while t.is_alive():
            t.join(0.2)
    return res


def preflight(d, fly):
    if not iface_ip(d.iface):
        d.log(f"FAIL: {d.iface} has no 192.168.10.x (not connected?)")
        return False
    ssid = active_ssid(d.iface)
    if ssid != d.ssid:
        d.log(f"FAIL: {d.iface} on {ssid!r}, expected {d.ssid!r}")
        return False
    threading.Thread(target=d.state_loop, daemon=True).start()
    for _ in range(5):
        if d.send("command", 3) == "ok":
            break
        time.sleep(1)
    else:
        d.log("FAIL: no reply to command")
        return False
    b = d.send("battery?", 5)
    d.battery = int(b) if b.isdigit() else None
    if fly and (d.battery is None or d.battery < MIN_BATTERY_FLY):
        d.log(f"FAIL: battery {b}% < {MIN_BATTERY_FLY}%")
        return False
    threading.Thread(target=d.video_loop, daemon=True).start()
    for _ in range(3):
        r = d.send("streamon", 5)
        if r == "ok":
            break
        if "unknown" in r or "unkown" in r:
            d.log("FAIL: drone doesn't know streamon -> update firmware in Tello app")
            return False
        time.sleep(1)
    else:
        d.log("FAIL: streamon not confirmed after 3 tries")
        return False
    t0, resent = time.time(), False
    while d.frames < 5 and time.time() - t0 < 15:
        if not resent and time.time() - t0 > 6:
            d.send("streamon", 5)
            resent = True
        time.sleep(0.1)
    if d.frames < 5:
        d.log("FAIL: no video packets" if d.packets == 0 else
              f"FAIL: {d.packets} video packets but {d.frames} frames (weak Wi-Fi?)")
        return False
    h, w = d.frame.shape[:2]
    if (w, h) != RES:
        d.log(f"FAIL: frame {w}x{h}, calibration is {RES}")
        return False
    threading.Thread(target=d.perception_loop, daemon=True).start()
    threading.Thread(target=d.battery_loop, daemon=True).start()
    t0 = time.time()
    while not d.st and time.time() - t0 < 3:
        time.sleep(0.1)
    if not d.st:
        d.log("WARNING: no status packets -> no IMU heading/height (blind legs disabled)")
    d.log(f"PASS (battery {d.battery}%, video ok, status {'ok' if d.st else 'missing'})")
    return True


def draw_topdown(drones, slots, assign, tag_map, now):
    S, W, H = 110, 700, 600
    img = np.full((H, W, 3), 30, np.uint8)
    ox, oy = W // 2, 60

    def px(p):
        return int(ox + p[0] * S), int(oy - p[2] * S)
    for tid, (R, t) in tag_map.items():               # all mapped tags (face as a line)
        xa = R[:, 0] * 0.15
        cv2.line(img, px(t - xa), px(t + xa), (255, 255, 255), 3 if tid != 10 else 5)
        cv2.putText(img, str(tid), (px(t)[0] + 6, px(t)[1] - 6), 0, 0.45, (200, 200, 200), 1)
    for i, (sp, _) in enumerate(slots):
        cv2.circle(img, px(sp), 6, (0, 200, 255), 1)
        cv2.putText(img, f"s{i}", (px(sp)[0] + 8, px(sp)[1]), 0, 0.4, (0, 200, 255), 1)
    for d in drones:
        e = d.est(now)
        if e is None:
            continue
        p, psi, age, src = e
        col = {"FIX": (0, 255, 0), "PRED": (0, 200, 255), "BLIND": (255, 120, 0)}[src]
        if src != "FIX":
            cv2.circle(img, px(p), int((D_SAFE_M + LOST_DRIFT_MPS * age) * S), (0, 0, 140), 1)
        cv2.circle(img, px(p), 8, col, -1)
        hx = p + 0.25 * np.array([math.sin(math.radians(psi)), 0, math.cos(math.radians(psi))])
        cv2.line(img, px(p), px(hx), col, 2)
        cv2.putText(img, str(d.num), (px(p)[0] + 10, px(p)[1] - 10), 0, 0.5, col, 1)
        if d.num in assign:
            cv2.line(img, px(p), px(slots[assign[d.num]][0]), (90, 90, 90), 1)
        cv2.line(img, px(p), px(np.zeros(3)), (255, 255, 0), 1)
        mid = (np.asarray(p) + np.zeros(3)) / 2
        cv2.putText(img, f"{math.hypot(p[0], p[2]):.2f}m", px(mid), 0, 0.4, (255, 255, 0), 1)
    cv2.putText(img, "green=sees tag  yellow=no tag (frozen)", (10, H - 12),
                0, 0.45, (200, 200, 200), 1)
    return img


# ======================= MISSION =======================
def tag10_relative(p, psi):
    """Tag 10 relative to the drone: (distance m, bearing deg (+right), above m)."""
    fwd, left, up = body_displacement(p, psi, np.zeros(3))
    return math.hypot(fwd, left) / 100.0, math.degrees(math.atan2(-left, fwd)), up / 100.0


def draw_tag10_prediction(img, d, p, psi, size10):
    """Cyan outline where tag 10 SHOULD appear (from map + drone pose), or an arrow
    at the image edge pointing toward it when it's out of view. img is full-res."""
    s_, c_ = math.sin(math.radians(psi)), math.cos(math.radians(psi))
    z = np.array([s_, 0.0, c_]); y = np.array([0.0, 1.0, 0.0]); x = np.cross(y, z)
    Rtc = np.column_stack([x, y, z]); R = Rtc.T; t = -R @ p        # camera<-origin (level)
    corners = obj_points(size10)
    zc = (R @ corners.T).T[:, 2] + t[2]
    col = (255, 255, 0)
    if (zc > 0.1).all():
        pr, _ = cv2.projectPoints(corners, cv2.Rodrigues(R)[0], t, d.K, d.D)
        pr = pr.reshape(-1, 2)
        H, W = img.shape[:2]
        if ((pr[:, 0] > -W) & (pr[:, 0] < 2 * W) & (pr[:, 1] > -H) & (pr[:, 1] < 2 * H)).all():
            cv2.polylines(img, [pr.astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
            cx, cy = pr.mean(0)
            if 0 <= cx < W and 0 <= cy < H:
                cv2.putText(img, "10?", (int(cx) - 20, int(cy)), 0, 1.0, col, 2)
                return
    dist, brg, up = tag10_relative(p, psi)                         # off-screen: arrow
    H, W = img.shape[:2]
    ang = math.atan2(-up, math.sin(math.radians(brg)))
    c0 = np.array([W / 2, H / 2])
    tip = c0 + np.array([math.cos(ang), math.sin(ang)]) * np.array([W * 0.45, H * 0.42])
    cv2.arrowedLine(img, tuple(int(v) for v in c0), tuple(int(v) for v in tip), col, 4, tipLength=0.15)


def locked_climb(d, p):
    """Vertical rc to reach tag-10 height (+offset), using the height sensor when tag 10's
    floor height is known; else the tag pose. Returns (c, height_error_m or None)."""
    h10, _ = TAG10_H.get()
    h = d.height_cm()
    if h10 is not None and h is not None:
        err = (h10 + HEIGHT_OFFSET_M) - h / 100.0          # >0 = must climb
        c = 0 if abs(err) < 0.03 else int(np.clip(100 * KP_HEIGHT * err, -MAX_RC_Z, MAX_RC_Z))
        return c, err
    eh = -HEIGHT_OFFSET_M - p[1]                            # tag-based fallback (y down)
    c = 0 if abs(eh) < 0.03 else int(np.clip(-KP_POS * eh, -MAX_RC_Z, MAX_RC_Z))
    return c, -eh


def land_async(d, why, fly):
    """Stop this drone and land it straight down, without blocking the control loop."""
    if not fly:                                   # dry-run: just say it, keep running
        if not getattr(d, "dry_noted", False):
            d.log(f"{why} -> would LAND in place (dry-run)")
            d.dry_noted = True
        return
    if d.landing:
        return
    d.landing = True
    d.log(f"{why} -> LANDING in place")
    d.send_noreply("rc 0 0 0 0")
    d.land_thread = threading.Thread(target=d.land, daemon=True)
    d.land_thread.start()


def main():
    global HEIGHT_OFFSET_M
    here = os.path.dirname(os.path.abspath(__file__))
    cam_default = os.path.join(here, "camera")
    if not os.path.isdir(cam_default):
        cam_default = os.path.join(here, "..", "config", "camera")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(here, "dongles.json"))
    ap.add_argument("--map", default=os.path.join(here, "map.yaml"))
    ap.add_argument("--tags", default=os.path.join(here, "tags.yaml"))
    ap.add_argument("--camera-dir", default=cam_default)
    ap.add_argument("--only", default="", help="e.g. 4 or 1,3,4,5")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to hold in slots")
    ap.add_argument("--height", type=int, default=SEARCH_HEIGHT_CM, help="search height, cm")
    ap.add_argument("--search-speed", type=int, default=SEARCH_YAW_RC)
    ap.add_argument("--slot-offset", type=int, default=int(HEIGHT_OFFSET_M * 100),
                    help="slot height above tag 10's center, cm (+ = higher)")
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    args = ap.parse_args()
    HEIGHT_OFFSET_M = args.slot_offset / 100.0
    search_h = max(60, min(250, args.height))
    yaw_rc = max(10, min(60, args.search_speed))
    only = {int(x) for x in args.only.split(",") if x.strip()}
    fly = args.fly
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no motion commands)'}")

    origin, tag_map, sizes = load_tag_files(args.map, args.tags)
    with open(args.tags) as f:
        dcfg = (yaml.safe_load(f) or {}).get("detector") or {}
    det_opts = {"quad_decimate": float(dcfg.get("quad_decimate", 2.0)),
                "refine_edges": bool(dcfg.get("refine_edges", True)),
                "decode_sharpening": float(dcfg.get("decode_sharpening", 0.25))}
    print(f"Map: origin tag {origin}, tags {sorted(tag_map)} "
          f"({', '.join(f'{k}:{sizes[k]*1000:.0f}mm' for k in sorted(tag_map))})")

    drones = load_drones(args.config, only, args.camera_dir, fly, tag_map, sizes, det_opts)
    slots, radius = make_slots(len(drones))
    # camera sees ~+-21 deg vertically: keep tag 10's bottom edge in view at the slot
    half = sizes[origin] / 2.0
    lowest = math.degrees(math.atan2(HEIGHT_OFFSET_M + half, radius))
    print(f"Slot height: {args.slot_offset:+d} cm vs tag 10 center "
          f"(tag 10 bottom edge {lowest:.0f} deg below camera axis)")
    if lowest > 20:
        print("WARNING: slot is high enough that tag 10's bottom may leave the view; "
              "other tags must be visible or the drone will freeze/land. Try a lower --slot-offset.")
    print(f"Drones: {[d.name for d in drones]}  slots radius {radius:.2f} m"
          + ("  (WARNING: > 1 m to fit spacing)" if radius > ARC_RADIUS_M + 1e-9 else ""))

    res = parallel(drones, lambda d: preflight(d, fly))
    if not all(res.get(d.num) for d in drones):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")

    os.makedirs(os.path.join(here, "logs"), exist_ok=True)
    logf = open(os.path.join(here, time.strftime("logs/approach_%Y%m%d_%H%M%S.csv")), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "drone", "state", "src", "tags", "mode", "rms", "px", "py", "pz", "psi",
                  "pos_err", "yaw_err", "a", "b", "c", "d", "height_cm", "searched", "battery"])

    assign, members = {}, set()
    phase, settle_t, hold_t, first_lock_t = "RUN", None, None, None
    try:
        if fly:
            if input("All PASS. Space clear, drones >= 1 m apart? Type FLY: ").strip() != "FLY":
                sys.exit("Aborted, no takeoff.")
            for d in drones:
                d.airborne = True

            def height_dm(d):
                h = re.sub(r"[^0-9]", "", d.send("height?", 5))
                return int(h) if h else -1

            def to(d):
                time.sleep(0.5 * drones.index(d))
                for _ in range(2):
                    if d.send("takeoff", 20) == "ok":
                        return True
                    time.sleep(1.0)
                    h = height_dm(d)
                    if h >= 3:
                        d.log(f"takeoff reply lost but airborne (height {h} dm) -> ok")
                        return True
                    d.log(f"not airborne (height {h} dm), retrying takeoff")
                return False
            if not all(parallel(drones, to).values()):
                print("A takeoff failed -> landing all")
                return
        for d in drones:
            d.yaw_acc = 0.0

        period = 1.0 / LOOP_HZ
        while True:
            t_loop = now = time.time()
            active = [d for d in drones if not d.landed and not d.landing]
            if not active:
                print("No drones left in the formation -> waiting for landings to finish.")
                return
            for d in active:
                if d.battery is not None and d.battery < LAND_BATTERY:
                    land_async(d, f"battery {d.battery}%", fly)
            cmds, all_in = {}, True

            def others_of(d):
                out = []
                for o in drones:                         # landing drones still obstacles
                    if o is d or o.landed:
                        continue
                    e = o.est(now)
                    if e is not None and e[2] < EST_MAX_AGE_S:
                        out.append((e[0], 0.0 if e[3] == "FIX" else LOST_DRIFT_MPS * e[2]))
                return out

            for d in active:
                fx = d.has_fix(now)
                if d.was_fix and not fx:
                    d.log(f"lost tags in {d.dstate}: {d.reason}")
                elif fx and not d.was_fix and d.dstate != "SEARCH":
                    d.log(f"tags back: {d.reason}")
                d.was_fix = fx
                e = d.est(now)
                localized = e is not None and e[2] < EST_MAX_AGE_S
                h = d.height_cm()
                c_hold = 0 if h is None else int(np.clip(KP_HEIGHT * (search_h - h), -MAX_RC_Z, MAX_RC_Z))
                if d.busy:                                   # cw/go running: no rc
                    cmds[d.num] = None
                    all_in = False
                    continue

                # ---------- SEARCH: climb, look, turn-and-look ----------
                if d.dstate == "SEARCH":
                    if d.has_fix(now):
                        d.dstate = "LOCKED"
                        first_lock_t = first_lock_t or now
                        p0, s0, _, _ = d.est(now)
                        dist, brg, up = tag10_relative(p0, s0)
                        h10, src10 = TAG10_H.get()
                        d.log(f"LOCALIZED from tags {d.fix_info.get('tags')} "
                              f"({d.fix_info.get('mode')}, {d.fix_info.get('rms', 0):.1f}px, "
                              f"searched {d.search_turned:.0f} deg) -> tag 10 is {dist:.2f} m away, "
                              f"{brg:+.0f} deg {'right' if brg >= 0 else 'left'}, "
                              f"{up * 100:+.0f} cm {'above' if up >= 0 else 'below'}; "
                              + (f"tag 10 floor height {h10 * 100:.0f} cm ({src10})" if h10
                                 else "tag 10 floor height: not known yet"))
                    else:
                        all_in = False
                        yaw = 0
                        if d.sub_t is None:
                            d.sub_t = now
                        glimpse = now - d.glimpse_t < GLIMPSE_HOLD_S
                        if d.sub == "CLIMB":
                            if (h is not None and abs(search_h - h) < 15) or now - d.sub_t > CLIMB_TIMEOUT_S:
                                d.sub, d.sub_t = "LOOK", now
                        elif d.sub == "LOOK":
                            look = LOOK_S if d.search_turned == 0 else SEARCH_LOOK_S
                            if now - d.sub_t > look and (not glimpse or now - d.sub_t > GLIMPSE_HOLD_S):
                                d.sub, d.sub_t, d.turn_start = "TURN", now, d.yaw_acc
                        elif d.sub == "TURN":
                            turned = (d.yaw_acc - d.turn_start) if d.st else (now - d.sub_t) * 40.0
                            if d.glimpse_t > d.sub_t + 0.3 or turned >= SEARCH_STEP_DEG:
                                d.search_turned += max(0.0, turned)
                                d.sub, d.sub_t = "LOOK", now
                            else:
                                yaw = yaw_rc
                        cmds[d.num] = ((0, 0, c_hold, yaw), None, None)
                        if d.search_turned > SEARCH_MAX_TURN_DEG:
                            land_async(d, f"searched {d.search_turned:.0f} deg without any "
                                          f"mapped tag", fly)
                        continue

                # ---------- no tag in view after localizing: FREEZE, then land ----------
                if not d.has_fix(now):
                    all_in = False
                    lost = e[2] if e else 99.0
                    yaw = 0
                    if d.dstate == "LOCKED" and e is not None and d.imu_yaw() is not None:
                        # rotating IN PLACE toward tag 10 is collision-free: keep turning
                        p, psi, _, _ = e
                        ey = wrap_deg(math.degrees(math.atan2(-p[0], -p[2])) - psi)
                        if abs(ey) > DEADBAND_YAW_DEG:
                            yaw = int(np.clip(KP_YAW * ey, -MAX_RC_YAW, MAX_RC_YAW))
                    c = 0
                    if d.dstate == "LOCKED" and e is not None:
                        c, _ = locked_climb(d, e[0])               # vertical in place (sensor)
                    cmds[d.num] = ((0, 0, c, yaw), None, None)     # never sideways without a tag
                    limit = LOCKED_NOFIX_S if d.dstate == "LOCKED" else LOST_LAND_S
                    if lost > limit:
                        land_async(d, f"no tag for {lost:.1f}s", fly)
                    continue

                d.dry_noted = False
                # ---------- LOCKED: sees a tag; turn in place to face tag 10 ----------
                if d.dstate == "LOCKED":
                    all_in = False
                    p, psi, _, _ = e
                    want = math.degrees(math.atan2(-p[0], -p[2]))
                    ey = wrap_deg(want - psi)
                    yaw = 0 if abs(ey) < DEADBAND_YAW_DEG else int(np.clip(KP_YAW * ey, -MAX_RC_YAW, MAX_RC_YAW))
                    c, herr = locked_climb(d, p)
                    sees10 = 10 in (d.fix_info.get("tags") or [])
                    d.aligned = herr is not None and abs(herr) < ALIGN_H_M \
                        and abs(ey) < ALIGN_YAW_DEG and sees10
                    cmds[d.num] = ((0, 0, c, yaw), None, None)   # up/down + turn only
                    continue

                # ---------- APPROACH: sees a tag; go to slot, keep spacing ----------
                if d.num not in assign:
                    cmds[d.num] = ((0, 0, 0, 0), None, None)
                    all_in = False
                    continue
                slot = slots[assign[d.num]]
                p, psi, _, _ = e
                rc, pe, ye = control(p, psi, slot, others_of(d))
                cmds[d.num] = (rc, pe, ye)
                if pe > TOL_POS_M or abs(ye) > TOL_YAW_DEG:
                    all_in = False

            h10, src10 = TAG10_H.get()
            if h10 and getattr(main, "_h10_src", None) != src10:
                main._h10_src = src10
                print(f">>> tag 10 center is {h10 * 100:.0f} cm above the floor (from {src10})")

            # ---- who approaches: all localized, or waited long enough ----
            locked = [d for d in active if d.dstate == "LOCKED"]
            ready = all(d.dstate != "SEARCH" for d in active) or \
                (first_lock_t is not None and now - first_lock_t > WAIT_ALL_S)
            if locked and ready:
                for d in locked:
                    if d.aligned:                   # at tag-10 height and facing it
                        d.dstate = "APPROACH"
            new_members = {d.num for d in active if d.dstate == "APPROACH"}
            if new_members != members:
                pos = {}
                for d in active:
                    if d.num in new_members:
                        ee = d.est(now)
                        if ee is not None:
                            pos[d.num] = ee[0]
                assign.clear()
                assign.update(match_slots(pos, slots) if pos else {})
                members = new_members
                print(f">>> approaching: slots {assign}")

            if phase == "RUN":
                settle_t = (settle_t or now) if (all_in and members) else None
                if settle_t and now - settle_t > SETTLE_S:
                    phase, hold_t = "HOLD", now
                    print(f">>> all in slots, holding {args.hold:.0f}s")
            elif now - hold_t > args.hold:
                print(">>> hold done -> landing all")
                return

            # ---- send / show ----
            for d in active:
                entry = cmds.get(d.num, ((0, 0, 0, 0), None, None))
                rc, pe, ye = entry if entry is not None else ((0, 0, 0, 0), None, None)
                if fly and entry is not None and not d.busy:
                    d.send_noreply("rc %d %d %d %d" % rc)
                e = d.est(now)
                p = e[0] if e else (np.nan,) * 3
                psi = e[1] if e else np.nan
                src = e[3] if e else "-"
                h = d.height_cm()
                fi = d.fix_info
                log.writerow([f"{now:.3f}", d.num, d.dstate, src, fi.get("tags"), fi.get("mode"),
                              f"{fi.get('rms', 0):.2f}", *[f"{x:.3f}" for x in p], f"{psi:.1f}",
                              "" if pe is None else f"{pe:.3f}", "" if ye is None else f"{ye:.1f}",
                              *rc, "" if h is None else h, f"{d.search_turned:.0f}", d.battery])
                img = (d.overlay if d.overlay is not None else np.zeros((RES[1], RES[0], 3), np.uint8)).copy()
                lines = [f"{d.name} {d.dstate}" + (f"/{d.sub}" if d.dstate == "SEARCH" else "")
                         + (" HOLD" if phase == "HOLD" else "") + f"  [{src}]",
                         d.reason[:60],
                         f"rc a{rc[0]:+d} b{rc[1]:+d} c{rc[2]:+d} d{rc[3]:+d}" + ("" if fly else " (DRY)"),
                         f"bat {d.battery}%  h {h if h is not None else '?'}cm"
                         + (f"  tag10 at {TAG10_H.get()[0] * 100:.0f}cm" if TAG10_H.get()[0] else "")
                         + (f"  searched {d.search_turned:.0f}" if d.dstate == "SEARCH" else "")]
                if e:
                    lines.append(f"pos x{p[0]:+.2f} y{p[1]:+.2f} z{p[2]:+.2f}  yaw {psi:+.0f}  age {e[2]:.1f}s")
                if pe is not None:
                    lines.append(f"err {pe:.2f} m {ye:+.0f} deg  slot s{assign.get(d.num)}")
                if e:
                    draw_tag10_prediction(img, d, p, psi, sizes[origin])
                    dist, brg, up = tag10_relative(p, psi)
                    lines.append(f"tag10: {dist:.2f} m, {abs(brg):.0f} deg "
                                 f"{'right' if brg >= 0 else 'left'}, {abs(up) * 100:.0f} cm "
                                 f"{'above' if up >= 0 else 'below'}")
                for i, s_ in enumerate(lines):
                    cv2.putText(img, s_, (10, 30 + 28 * i), 0, 0.65, (0, 0, 0), 4)
                    cv2.putText(img, s_, (10, 30 + 28 * i), 0, 0.65, (0, 255, 255), 2)
                cv2.imshow(d.name, cv2.resize(img, (480, 360)))
            cv2.imshow("top view", draw_topdown(drones, slots, assign, tag_map, now))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("q -> landing all")
                return
            if k == ord(" ") and fly:
                print("!!! EMERGENCY: motors off all")
                for d in drones:
                    d.send_noreply("emergency")
                    d.landed = True
                return
            time.sleep(max(0.0, period - (time.time() - t_loop)))
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all")
    finally:
        if fly:
            for d in drones:                                   # let in-progress landings finish
                th = getattr(d, "land_thread", None)
                if th is not None:
                    th.join(20)
            parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        logf.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()