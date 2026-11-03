"""Where the deploy stack's non-build artefacts live.

Two different roots matter here, and conflating them is what pointed the default
checkpoint at a path that cannot exist:

    install/<pkg>/share/<pkg>          <- get_package_share_directory()
      -4 -> the colcon workspace root
      -3 -> ``install`` itself

Every launch file in this directory used to resolve the workspace with three
``..`` and call the result "the repo root". Three up is ``install``, so the
default ``ckpt`` became ``install/models/...`` and ``distill_fused_policy`` died
on a missing checkpoint while every script that passes ``ckpt:=`` explicitly
kept working — which is why it survived so long.

The root that actually holds ``models/`` and ``resources/`` is the deploy root:
the workspace root on this repo (``install/`` sits next to ``models/``), but its
*parent* on the Jetson (``~/go2_deploy/ws`` vs ``~/go2_deploy``). The depth is
therefore probed rather than counted, and ``$GO2_DEPLOY_ROOT`` — exported by
``~/go2_deploy/env.sh`` — overrides the probe outright.

Use from a launch file::

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deploy_paths import default_ep_root, deploy_root
"""
import os

from ament_index_python.packages import get_package_share_directory

_PKG = "deploy_rl_policy"
_SHARE_DEPTH = 4          # install/<pkg>/share/<pkg> -> workspace root


def workspace_root() -> str:
    """The colcon workspace root that ``install/`` belongs to."""
    return os.path.normpath(
        os.path.join(get_package_share_directory(_PKG), *[".."] * _SHARE_DEPTH))


def deploy_root() -> str:
    """Directory holding ``models/`` (and, in the dev repo, ``resources/``).

    ``models/`` is the marker because it is the one thing every deployment has:
    the Jetson stage ships ``ws/ ep/ models/ env.sh`` and nothing else, so
    ``resources/`` or ``deploy_mac/`` would answer "absent" there and be wrong.
    """
    ws = workspace_root()
    for cand in (os.environ.get("GO2_DEPLOY_ROOT"), ws, os.path.dirname(ws)):
        if cand and os.path.isdir(os.path.join(cand, "models")):
            return os.path.normpath(cand)
    return ws


def default_ep_root(root: str) -> str:
    """The training repo supplying Actor / Estimator / RecurrentDepthBackbone.

    ``add_extreme_parkour_to_path()`` reaches for ``$EP_ROOT/rsl_rl``, so that
    is what gets probed. Layouts: ``<parent>/go2-flyball-parkour`` on a dev
    machine, ``<root>/ep`` on the Jetson stage.
    """
    env = os.environ.get("EP_ROOT")
    if env:
        return env
    siblings = (
        os.path.join(os.path.dirname(root), "go2-flyball-parkour"),
        os.path.join(root, "ep"),
    )
    for cand in siblings:
        if os.path.isdir(os.path.join(cand, "rsl_rl")):
            return cand
    return siblings[0]
