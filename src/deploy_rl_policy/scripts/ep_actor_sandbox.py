"""Load an Extreme Parkour base policy from a raw checkpoint, for deploy.

Composes the training-side `Actor` + `Estimator` into the single callable that
`save_jit_base.py` traces, and loads `model_*.pt` via `state_dict` — bypassing
torch.jit entirely. The previous JIT-traced policy produced action norms ~5x
different from the live training-side `policy(obs, hist_encoding=True)` call even
on bit-exactly identical obs; this module sidesteps that whole path.

The network classes are **imported from the training repo**, not reimplemented
here, so there is exactly one definition of the architecture:
https://github.com/Roman-089/go2-flyball-parkour  (`rsl_rl.modules`)

Usage:
    actor_module = build_base_actor("/abs/path/to/model_6500.pt")
    action = actor_module(obs_tensor)   # (1, 753) -> (1, 12)  Isaac order
"""
from pathlib import Path

import torch
import torch.nn as nn

from distill_common import EP_ROOT_DEFAULT, add_extreme_parkour_to_path

# Obs / model dims — must match legged_robot_config.py exactly.
N_PROPRIO       = 53
NUM_SCAN        = 132
N_PRIV_EXPLICIT = 9
N_PRIV_LATENT   = 29
HISTORY_LEN     = 10
NUM_ACTIONS     = 12
NUM_OBS = N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT + N_PRIV_LATENT + HISTORY_LEN * N_PROPRIO  # 753

# Architecture dims (legged_robot_config.py:708-711; estimator verified against
# the released teacher checkpoint: 53 -> 128 -> 64 -> 9).
SCAN_ENCODER_DIMS   = [128, 64, 32]
ACTOR_HIDDEN_DIMS   = [512, 256, 128]
PRIV_ENCODER_DIMS   = [64, 20]
ESTIMATOR_HIDDEN    = [128, 64]


class BasePolicy(nn.Module):
    """Actor + Estimator wrapper — the inference contract save_jit_base.py traces."""

    PRIV_START = N_PROPRIO + NUM_SCAN
    PRIV_END   = N_PROPRIO + NUM_SCAN + N_PRIV_EXPLICIT

    def __init__(self, actor: nn.Module, estimator: nn.Module):
        super().__init__()
        self.actor = actor
        self.estimator = estimator

    def forward(self, obs):  # obs: (B, 753)
        obs = obs.clone()
        obs[:, self.PRIV_START:self.PRIV_END] = self.estimator(obs[:, :N_PROPRIO])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=None)


def build_base_actor(checkpoint_path: str, device: str = "cpu",
                     ep_root: Path = EP_ROOT_DEFAULT) -> BasePolicy:
    """Load model_*.pt and return a BasePolicy ready for inference.

    The checkpoint stores the full ActorCriticRMA state under `model_state_dict`
    (so actor params have an `actor.` prefix), and the estimator weights under
    `estimator_state_dict`.
    """
    add_extreme_parkour_to_path(Path(ep_root))
    from rsl_rl.modules.actor_critic import Actor, get_activation
    from rsl_rl.modules.estimator import Estimator

    # Actor expects an activation *module* (ActorCriticRMA resolves it before
    # constructing); Estimator resolves the string itself.
    actor = Actor(
        num_prop=N_PROPRIO,
        num_scan=NUM_SCAN,
        num_actions=NUM_ACTIONS,
        scan_encoder_dims=SCAN_ENCODER_DIMS,
        actor_hidden_dims=ACTOR_HIDDEN_DIMS,
        priv_encoder_dims=PRIV_ENCODER_DIMS,
        num_priv_latent=N_PRIV_LATENT,
        num_priv_explicit=N_PRIV_EXPLICIT,
        num_hist=HISTORY_LEN,
        activation=get_activation("elu"),
    )
    estimator = Estimator(
        input_dim=N_PROPRIO,
        output_dim=N_PRIV_EXPLICIT,
        hidden_dims=ESTIMATOR_HIDDEN,
        activation="elu",
    )

    ckpt = torch.load(checkpoint_path, map_location=device)

    full_sd = ckpt["model_state_dict"]
    actor_sd = {k[len("actor."):]: v for k, v in full_sd.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_sd, strict=True)
    estimator.load_state_dict(ckpt["estimator_state_dict"], strict=True)

    policy = BasePolicy(actor, estimator).to(device)
    policy.eval()
    return policy


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python ep_actor_sandbox.py /path/to/model_XXXX.pt")
        sys.exit(1)
    pol = build_base_actor(sys.argv[1])
    with torch.no_grad():
        obs = torch.zeros(1, NUM_OBS)
        a = pol(obs)
    print(f"obs.shape={tuple(obs.shape)}  action.shape={tuple(a.shape)}  norm={a.norm().item():+.4f}")
    print(f"action: {a.squeeze().tolist()}")
