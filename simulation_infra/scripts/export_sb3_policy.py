"""Export an SB3 checkpoint (+ its VecNormalize stats) to a TorchScript file.

Standalone CLI around ``action_space_kit.rl.policy_export.export_policy`` (the same export the
recorders use). Loads that module by file path so no Kit app (``pxr``) is needed.

    python scripts/export_sb3_policy.py --checkpoint logs/sb3/AS-Tello-Approach-v0/<run>/model.zip \
        --frame_stack 8 --out outputs/exports/approach_policy.pt

``--checkpoint`` may also be a task name (e.g. ``AS-Tello-Approach-v0``): then the newest run's
checkpoint under ``logs/sb3/<task>`` is exported. ``--frame_stack`` defaults to the value in the
run's ``params/agent.yaml`` (8 for every task in this kit).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / "source" / "action_space_kit" / "action_space_kit"


def load_policy_export():
    spec = importlib.util.spec_from_file_location("policy_export", KIT / "rl" / "policy_export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame_stack_of(ckpt: Path, default: int = 8) -> int:
    agent_yaml = ckpt.parent / "params" / "agent.yaml"
    if agent_yaml.exists():
        m = re.search(r"^frame_stack:\s*(\d+)", agent_yaml.read_text(encoding="utf-8"), re.M)
        if m:
            return int(m.group(1))
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="SB3 model .zip, or a task name for its newest run.")
    parser.add_argument("--frame_stack", type=int, default=None, help="Frames stacked at training (default: params/agent.yaml).")
    parser.add_argument("--out", type=str, default=None, help="Output .pt (default: outputs/exports/<task>_<run>.pt).")
    args = parser.parse_args()

    pe = load_policy_export()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():  # a task name
        ckpt = pe.latest_checkpoint(HERE.parent / "logs" / "sb3" / args.checkpoint)
    ckpt = ckpt.resolve()
    n_stack = args.frame_stack if args.frame_stack is not None else frame_stack_of(ckpt)
    out = Path(args.out) if args.out else HERE.parent / "outputs" / "exports" / f"{ckpt.parent.parent.name}_{ckpt.parent.name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] checkpoint: {ckpt}\n[export] norm stats: {pe.norm_path_for(ckpt)}\n[export] frame_stack: {n_stack}", flush=True)
    pe.export_policy(ckpt, n_stack, out)

    import torch  # noqa: E402  (after the export subprocess, so a broken torch import shows up there first)

    m = torch.jit.load(str(out), map_location="cpu")
    in_dim = m.mean.shape[0]
    act = m(torch.zeros(1, in_dim))
    print(f"[export] wrote {out} ({out.stat().st_size} bytes): obs {in_dim} -> action {tuple(act.shape)}", flush=True)


if __name__ == "__main__":
    main()
