#!/usr/bin/env python3
"""Run the defend policy with no drones: three kinematic defenders (they move exactly as commanded)
against the scripted attackers of config/defend.yaml. Prints obs, action and rc every tick.

    python defend_offline.py --seconds 12
"""
import argparse

from policies.defend import CFG_PATH, offline

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--seconds", type=float, default=12.0)
ap.add_argument("--config", default=CFG_PATH)
ap.add_argument("--policy-dir", default=None, help="defender_{0,1,2}.pt folder (default: policy_dir in the config)")
ap.add_argument("--attackers", choices=("scripted", "udp", "telemetry"), default=None,
                help="override `attackers:` in the config (telemetry = drone_gui's feed on telemetry_port)")
ap.add_argument("--stagger", type=float, default=None,
                help="seconds the 2nd scripted attacker launches late (arena case; config default 0)")
args = ap.parse_args()
offline(args.seconds, cfg_path=args.config, policy_dir=args.policy_dir, attackers=args.attackers, stagger=args.stagger)
