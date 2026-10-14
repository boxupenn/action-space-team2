"""Potential shaping with the same discount as the PPO return."""


def potential_delta(previous, current, terminated, gamma, xp):
    # Timeouts bootstrap; only absorbing terminations have zero next potential.
    return gamma * xp.where(terminated, 0 * current, current) - previous
