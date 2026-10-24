"""Export a skrl checkpoint (PPO / IPPO / MAPPO from scripts/skrl/train.py) to TorchScript.

The exported module maps a raw observation batch (N, obs_dim) to the deterministic action
(N, act_dim): skrl's RunningStandardScaler (observation_preprocessor) -> the shared MLP
(net_container + policy_layer) -> clamp to [-1, 1]. No Kit app or skrl import is needed:
the network is rebuilt from the tensor shapes in the checkpoint and the activation named in
the run's params/agent.yaml.

    python scripts/export_skrl_policy.py --checkpoint logs/skrl/tello_hover/<run>/checkpoints/agent_1000.pt
    python scripts/export_skrl_policy.py --checkpoint tello_hover            # newest run under logs/skrl/tello_hover
    python scripts/export_skrl_policy.py --checkpoint tello_defend --agent defender_0   # multi-agent: one agent's policy

Multi-agent (IPPO/MAPPO) checkpoints hold one policy per agent; ``--agent`` picks it
(default: the first). IPPO/MAPPO defenders share nothing, so export each one you deploy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ACTIVATIONS = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh, "leaky_relu": torch.nn.LeakyReLU,
               "sigmoid": torch.nn.Sigmoid, "softplus": torch.nn.Softplus, "selu": torch.nn.SELU, "gelu": torch.nn.GELU}


def latest_checkpoint(exp_dir: Path) -> Path:
    runs = sorted((p for p in exp_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    for run in reversed(runs):
        best = run / "checkpoints" / "best_agent.pt"
        agents = [(int(m.group(1)), p) for p in (run / "checkpoints").glob("agent_*.pt") if (m := re.match(r"agent_(\d+)$", p.stem))]
        if agents:
            return max(agents)[1]
        if best.exists():
            return best
    raise FileNotFoundError(f"no checkpoint under {exp_dir}")


def activation_of(ckpt: Path) -> str:
    agent_yaml = ckpt.parent.parent / "params" / "agent.yaml"
    if agent_yaml.exists():
        cfg = yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))
        try:
            net = cfg["models"]["policy"]["network"][0]
            act = net.get("activations", "elu")
            return act if isinstance(act, str) else act[0]
        except (KeyError, IndexError, TypeError):
            pass
    return "elu"


def output_squash_of(ckpt: Path) -> str | None:
    """The activation wrapped around ACTIONS in the policy's ``output`` (e.g. ``tanh(ACTIONS)``), or None."""
    agent_yaml = ckpt.parent.parent / "params" / "agent.yaml"
    if agent_yaml.exists():
        cfg = yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))
        try:
            out = str(cfg["models"]["policy"].get("output", "ACTIONS"))
        except (KeyError, TypeError):
            return None
        m = re.fullmatch(r"\s*([a-z_]+)\(\s*ACTIONS\s*\)\s*", out)
        if m and m.group(1) in ACTIVATIONS:
            return m.group(1)
    return None


class SkrlPolicy(torch.nn.Module):
    def __init__(self, policy_sd: dict, pre_sd: dict | None, activation: str, clip_threshold: float = 5.0, squash: str | None = None):
        super().__init__()
        # output squash (skrl `output: tanh(ACTIONS)`): parameter-free, so it is invisible in the state dict
        self.squash = ACTIVATIONS[squash]() if squash else torch.nn.Identity()
        layer_keys = sorted((k for k in policy_sd if re.match(r"net_container\.\d+\.weight$", k)), key=lambda k: int(k.split(".")[1]))
        # shared policy/value model (models.separate: False, e.g. PPO/IPPO cfgs): hidden layers in net_container,
        # the action head in policy_layer. Separate models (MAPPO cfg): the head is the last net_container layer.
        if "policy_layer.weight" in policy_sd:
            head_key = "policy_layer.weight"
        else:
            head_key = layer_keys.pop()
        act_cls = ACTIVATIONS[activation.lower()]
        layers: list[torch.nn.Module] = []
        for k in layer_keys:
            w, b = policy_sd[k], policy_sd[k.replace("weight", "bias")]
            lin = torch.nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(b)
            layers += [lin, act_cls()]
        self.net = torch.nn.Sequential(*layers)
        w, b = policy_sd[head_key], policy_sd[head_key.replace("weight", "bias")]
        self.head = torch.nn.Linear(w.shape[1], w.shape[0])
        self.head.weight.data.copy_(w)
        self.head.bias.data.copy_(b)
        obs_dim = policy_sd[layer_keys[0] if layer_keys else head_key].shape[1]
        if pre_sd is not None:
            mean, var = pre_sd["running_mean"].float(), pre_sd["running_variance"].float()
        else:
            mean, var = torch.zeros(obs_dim), torch.ones(obs_dim)
        self.register_buffer("mean", mean)
        self.register_buffer("std", torch.sqrt(var + 1e-8))
        self.clip_threshold = float(clip_threshold)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.mean) / self.std, -self.clip_threshold, self.clip_threshold)
        return torch.clamp(self.squash(self.head(self.net(x))), -1.0, 1.0)


def _load_chase():
    """chase.py by path: importing the package needs a running Kit app, the module itself only torch."""
    import importlib.util

    path = ROOT / "source" / "action_space_kit" / "action_space_kit" / "tasks" / "direct" / "tello_defend" / "chase.py"
    spec = importlib.util.spec_from_file_location("tello_defend_chase", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Obs23Policy(torch.nn.Module):
    """Defend policy on the kit's 23-dim observation: appends the latched target assignment
    (chase.assign_latched, one-hot target + goalkeeper flag) for defender ``index`` and runs the
    cloned 26-dim network. The latch lives in a buffer, so one loaded module = one drone, called
    once per control tick with a (1, 23) batch; a different batch size runs un-latched."""

    def __init__(self, policy: SkrlPolicy, index: int, num_attackers: int):
        super().__init__()
        self.policy = policy
        self.index = index
        self.n = num_attackers
        self.register_buffer("prev", torch.full((1,), -1, dtype=torch.long))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        E = obs.shape[0]
        prev = self.prev if E == 1 else torch.full((E,), -1, dtype=torch.long, device=obs.device)
        target, goalkeeper, new_prev = assign_latched(obs, self.index, self.n, prev)
        if E == 1:
            self.prev.copy_(new_prev)
        onehot = torch.nn.functional.one_hot(target, self.n).float() * (~goalkeeper).float().unsqueeze(-1)
        return self.policy(torch.cat([obs, onehot, goalkeeper.float().unsqueeze(-1)], dim=-1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="skrl agent_*.pt / best_agent.pt, or an experiment dir name under logs/skrl.")
    parser.add_argument("--agent", type=str, default=None, help="Multi-agent checkpoints: which agent's policy to export (default: first).")
    parser.add_argument("--out", type=str, default=None, help="Output .pt (default: outputs/exports/<experiment>_<run>[_<agent>].pt).")
    parser.add_argument("--obs23", action="store_true", default=False,
                        help="Defend: wrap a 26-dim (23 + latched assignment) policy so the file takes the kit's 23 numbers (needs --agent defender_i).")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():  # an experiment name (a same-named directory may exist in the repo, e.g. tello_defend/)
        ckpt = latest_checkpoint(ROOT / "logs" / "skrl" / args.checkpoint)
    ckpt = ckpt.resolve()
    data = torch.load(ckpt, map_location="cpu", weights_only=False)

    agent_tag = ""
    if "policy" not in data:  # multi-agent: {agent_name: {policy, value, ...}}
        names = [k for k, v in data.items() if isinstance(v, dict) and "policy" in v]
        if not names:
            raise SystemExit(f"unrecognised checkpoint layout, top-level keys: {list(data)}")
        name = args.agent or names[0]
        if name not in names:
            raise SystemExit(f"agent {name!r} not in checkpoint; available: {names}")
        print(f"[export] multi-agent checkpoint, agents: {names}; exporting {name}", flush=True)
        data = data[name]
        agent_tag = f"_{name}"

    policy_sd = data["policy"]
    pre_sd = data.get("observation_preprocessor") or data.get("state_preprocessor")
    activation = activation_of(ckpt)
    squash = output_squash_of(ckpt)
    model = SkrlPolicy(policy_sd, pre_sd, activation, squash=squash).eval()

    run = ckpt.parent.parent
    out = Path(args.out) if args.out else ROOT / "outputs" / "exports" / f"{run.parent.name}_{run.name}{agent_tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    obs_dim, act_dim = model.mean.shape[0], model.head.out_features
    with torch.no_grad():
        if args.obs23:
            m = re.fullmatch(r"defender_(\d+)", args.agent or "")
            if not m:
                raise SystemExit("--obs23 needs --agent defender_<i>")
            n = (obs_dim - 1 - 9) // 8  # 26 = 9 + 7n + n + 1
            if obs_dim != 9 + 8 * n + 1:
                raise SystemExit(f"--obs23 expects a 9 + 8n + 1 = {9 + 8 * n + 1}-dim policy, got {obs_dim}")
            globals()["assign_latched"] = _load_chase().assign_latched  # resolved by torch.jit.script from Obs23Policy.forward
            scripted = torch.jit.script(Obs23Policy(model, int(m.group(1)), n))
            obs_dim = 9 + 7 * n
        else:
            scripted = torch.jit.trace(model, torch.zeros(2, obs_dim))
    scripted.save(str(out))
    check = torch.jit.load(str(out))(torch.zeros(1, obs_dim))
    print(f"[export] checkpoint: {ckpt}\n[export] activation: {activation}, output squash: {squash or 'none'}, preprocessor: {'yes' if pre_sd is not None else 'none'}", flush=True)
    print(f"[export] wrote {out} ({out.stat().st_size} bytes): obs {obs_dim} -> action {tuple(check.shape)} (clamped to [-1, 1])", flush=True)


if __name__ == "__main__":
    main()
