"""Explicit PPO warm-start changes that preserve the deterministic actor."""

import math

import torch


def prepare_finetune(model, action_std):
    policy = model.policy
    if not math.isfinite(action_std) or action_std <= 0:
        raise ValueError("action_std must be finite and positive")
    with torch.no_grad():
        policy.log_std.fill_(math.log(action_std))
    policy.optimizer.state.pop(policy.log_std, None)
