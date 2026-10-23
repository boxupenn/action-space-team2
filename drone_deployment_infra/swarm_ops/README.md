# swarm_ops: Tello swarm commands

All commands run from `drone_deployment_infra/swarm_ops/`. Config lives in `../config/`
(`dongles.json`, `map.yaml`, `tags.yaml`, `camera/tello_<n>.yaml`).

```
cd ~/Action_Space/drone_deployment_infra/swarm_ops
source <your venv>/bin/activate
```

---

## 1. One-time setup (per laptop)

```
pip install -r requirements.txt
echo "net.ipv4.conf.all.rp_filter=2" | sudo tee /etc/sysctl.d/99-tello.conf
sudo sysctl --system
```
- Plug the USB Wi-Fi adapters into a **powered hub** (adapter names in `dongles.json`
  belong to those exact dongles).
- Create every drone's Wi-Fi profile from `dongles.json` (drones can be OFF):
```
python connect.py --create --all
```
  Each profile is locked to its adapter + its drone's radio (BSSID), auto-connects,
  uses static IP `192.168.10.10<n>`, 2.4 GHz only.

**New drone?** Add its entry in `../config/dongles.json` (`iface`, `drone`, `ssid`, `bssid`),
then `python connect.py --create <n>`.
Find a drone's BSSID: `nmcli -f SSID,BSSID dev wifi list | grep TELLO`.

---

## 2. Connect drones (every session)

Turn the drones on, wait ~20-30 s (they join by themselves), then check:
```
python connect.py 4 5 6 7 9          # verify; connects any that aren't up yet
python connect.py --all              # every drone in dongles.json
python connect.py 4 5 6 7 9 --wait 60   # run FIRST, then power on: each joins as it appears
python connect.py 4 5 6 7 9 --down   # disconnect
```
Every drone must say `OK`. `NOT DETECTED` = adapter unplugged / hub unpowered.

Tip: chain it so a flight only starts when all drones are ready:
```
python connect.py 4 5 6 7 9 && python swarm_formation.py --only 4,5,6,7,9 --fly --fast --headless
```

---

## 3. Swarm formation (`swarm_formation.py`)

Drones take off, find tag 10 (or any mapped tag), fly to slots on an arc in front of
tag 10, hold, land. **Without `--fly` it is a dry-run** (no takeoff, no motion).

| Mode | Command |
|---|---|
| Dry-run (props off, check tags/video) | `python swarm_formation.py --only 4,5,6` |
| Normal (slow, careful) | `python swarm_formation.py --only 4,5,6 --fly` |
| Fast | `python swarm_formation.py --only 4,5,6 --fly --fast` |
| Headless (no windows, terminal only) | `python swarm_formation.py --only 4,5,6 --fly --fast --headless` |
| Flip wave (back flip one by one) | `python swarm_formation.py --only 4,5,6 --fly --fast --flip` |
| Recommended | `python swarm_formation.py --only 4,5,6 --fly --fast --headless --rest-radius 0.15 --hold 1` |

**Options**
| Flag | Meaning |
|---|---|
| `--only 4,5,6` | which drones |
| `--fly` | actually fly (without it: dry-run) |
| `--fast` | faster search/approach/turns (same final precision) |
| `--headless` | no windows; keys in terminal: `q`+Enter land all, `e`+Enter EMERGENCY |
| `--no-video` | no camera windows, only the top-view map |
| `--display-hz 5` | refresh windows less often (saves CPU) |
| `--rest-radius 0.15` | slot "close enough" zone in m (default 0.08); inside it the drone just holds |
| `--hold 1` | seconds to hold the formation before landing (default 5) |
| `--flip` | back flip wave once in formation (needs battery >= 55 %, ~1 m free behind each slot) |
| `--radius 1.3` | slot distance from tag 10 in m (don't go below ~1.2) |
| `--slot-offset 0` | slot height vs tag 10 center, cm |
| `--no-map` | use tag 10 only (ignore map.yaml) |
| `--ignore-tags 6,7` | skip specific mapped tags |

**Keys** (window mode): `q` = land all, `SPACE` = EMERGENCY motor stop. `Ctrl+C` = land all.

**Placement:** drones 2-3 m in front of tag 10 (within +-60 deg), 1 m+ apart,
**facing tag 10** (search becomes instant). More drones -> the arc moves further out
(5 drones ~1.5 m, 6 drones ~2 m from tag 10).

**Logs:** `logs/formation_<date>_<time>.csv` (state, position, commands, battery per drone).

---

## 4. Multi hover (`multi_hover.py`, no video)

All selected drones take off together, hover, land. Good for testing many drones.
```
python multi_hover.py --config ../config/dongles.json --only 1,2,3,4,5,8,9 --hover 10
```
All-or-nothing preflight (one failure -> nobody takes off). `Ctrl+C` lands all.
Drones ~1.5 m apart.

---

## 5. Map the tags (`map_tags.py`, no flying)

Needed whenever tags are added or moved. Hold the drone (props off) so it sees
tag 10 together with each other tag until all are green, then press `q` to save.
```
python connect.py 4
python map_tags.py --drone 4 --tags-list 10,6,7
```
Every tag needs its size in `../config/tags.yaml` (`sizes_mm`, black square in mm).
Check: dry-run shows no `MAP CHECK` warnings.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `NOT DETECTED` / `OSError: No such device` | adapter unplugged or hub unpowered: replug, `nmcli dev status \| grep wlx` |
| `drone not broadcasting` | drone off / asleep / flat battery: power it on |
| Drones switch off while connecting | idle auto-off / heat: power on just before flying, connect right away |
| Takeoff failed (height 0) | low battery, hot, tilted surface, or IMU calibration needed (Tello app) |
| Pixelated video with 5+ drones | 2.4 GHz crowding: power drones on ~10 s apart, spread adapters, EDU video lower (see `EDU_MODIF_README.md`) |
| `Map: tags [...]` doesn't match the wall | re-map with `map_tags.py` |
| Wi-Fi name `TELLO-TELLO-6` | in the Tello app type only `6` (it adds `TELLO-`) |
