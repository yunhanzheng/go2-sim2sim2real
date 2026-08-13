#!/usr/bin/env python
"""Pre-launch checkpoint/flag contract checker for the Go2 deploy stack.

The deploy wrappers (run_real_remote.sh / run_dryrun_remote.sh) can silently
launch any checkpoint with wrong observation-contract flags (env_class_eq_17,
yaw_head, zero_vel_obs) — each mismatch silently breaks the policy. This
script rejects such launches at startup.

Usage (exact interface consumed by the wrappers — do not change):

    python deploy_mac/diag/check_deploy_contract.py \
        --ckpt PATH --env_class_eq_17 0|1 --yaw_head 0|1 \
        --zero_vel_obs 0|1 --binary_vel 0|1 --lin_vel_max FLOAT

Exit 0 = contract OK; exit 1 = mismatch (human-readable reasons printed).

Two independent checks:

1. Manifest check — deploy_manifest.yaml next to the checkpoint (flat
   ``key: value`` file; comments allowed). Hard-contract fields
   (yaw_head, env_class_eq_17, zero_vel_obs) must match the CLI flags or
   the check FAILS. Soft fields (binary_vel, lin_vel_max, policy_kp,
   policy_kd, contact_threshold_real, sim_validated_ckpts) only produce
   warnings. No manifest at all → warning, continue.

2. Checkpoint dim probe — torch.load the .pt and inspect
   depth_encoder_state_dict["output_mlp.0.weight"]:
       (32, 512) → no yaw head   (latent only)
       (34, 512) → yaw head      (32 latent + 2 yaw_pred)
   A probed head that contradicts --yaw_head FAILS the check. If torch is
   unavailable or the layout is unrecognized, a warning is printed and the
   probe is skipped (no failure).

Plain Python, no ROS deps. yaml is optional (a minimal flat-file parser is
used as fallback — keep manifests flat so this always works).
"""

import argparse
import os
import sys
from pathlib import Path

DEPTH_LATENT_DIM = 32
YAW_PRED_DIM = 2
MANIFEST_NAME = "deploy_manifest.yaml"

HARD_FIELDS = ("yaw_head", "env_class_eq_17", "zero_vel_obs")
# Soft fields we compare against CLI flags (warn only).
SOFT_COMPARED_FIELDS = ("binary_vel", "lin_vel_max")
# Soft fields we merely surface as recommendations (no CLI counterpart here).
SOFT_INFO_FIELDS = ("policy_kp", "policy_kd", "contact_threshold_real", "note")

_errors = []
_warnings = []


def err(msg):
    _errors.append(msg)
    print("ERROR:   " + msg)


def warn(msg):
    _warnings.append(msg)
    print("WARNING: " + msg)


def info(msg):
    print("info:    " + msg)


# --------------------------------------------------------------------------
# Manifest parsing
# --------------------------------------------------------------------------

def _strip_scalar(value):
    """Strip quotes / trailing comments from a flat-YAML scalar."""
    value = value.strip()
    if value and value[0] in "\"'":
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end]
        return value[1:]
    # unquoted: a '#' preceded by whitespace starts a comment
    for i, ch in enumerate(value):
        if ch == "#" and (i == 0 or value[i - 1] in " \t"):
            return value[:i].strip()
    return value


def parse_flat_manifest(text):
    """Minimal parser for a FLAT ``key: value`` YAML file (fallback when the
    yaml module is unavailable). Comments (#) and blank lines are skipped."""
    data = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = key.strip()
        if not key or " " in key:
            continue  # not a flat scalar key — ignore
        data[key] = _strip_scalar(rest)
    return data


def load_manifest(path):
    text = path.read_text()
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
        warn(f"manifest {path} did not parse to a mapping; "
             "falling back to flat parser")
    except ImportError:
        pass
    return parse_flat_manifest(text)


def as_int01(value, field):
    """Coerce a manifest/CLI value to 0/1, or None if not coercible."""
    if isinstance(value, bool):
        return int(value)
    try:
        iv = int(str(value).strip())
    except (ValueError, TypeError):
        warn(f"manifest field '{field}' has non-integer value {value!r}; "
             "skipping comparison")
        return None
    if iv not in (0, 1):
        warn(f"manifest field '{field}' = {iv} is not 0/1; "
             "skipping comparison")
        return None
    return iv


# --------------------------------------------------------------------------
# Checkpoint yaw-head dim probe
# --------------------------------------------------------------------------

def probe_yaw_head(ckpt_path):
    """Return (has_yaw_head: bool, evidence: str) or (None, reason) when the
    probe cannot determine the head layout."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401
    except Exception as exc:  # torch missing or broken install
        return None, f"torch not available ({exc}); skipping dim probe"
    try:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=True)
        except Exception:
            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
    except Exception as exc:
        return None, f"could not torch.load checkpoint: {exc}"

    if not isinstance(ckpt, dict):
        return None, f"unexpected checkpoint type {type(ckpt).__name__}"

    # Standard extreme-parkour distill ckpt: depth encoder lives in
    # ckpt["depth_encoder_state_dict"]. Fall back to treating the top level
    # as a raw state dict (e.g. an exported encoder-only file).
    sd = ckpt.get("depth_encoder_state_dict")
    if not isinstance(sd, dict):
        sd = ckpt

    weight_keys = [k for k in sd
                   if isinstance(k, str) and k.endswith("output_mlp.0.weight")]
    if not weight_keys:
        return None, ("no depth-encoder output_mlp.0.weight found "
                      "(teacher-only or unknown checkpoint layout?)")
    key = weight_keys[0]
    tensor = sd[key]
    if not hasattr(tensor, "shape") or len(tensor.shape) != 2:
        return None, f"{key} is not a 2-D tensor; cannot probe"
    out_dim = int(tensor.shape[0])
    evidence = f"{key} shape {tuple(tensor.shape)} -> encoder output {out_dim}-D"
    if out_dim == DEPTH_LATENT_DIM:
        return False, evidence + " (32 latent, NO yaw head)"
    if out_dim == DEPTH_LATENT_DIM + YAW_PRED_DIM:
        return True, evidence + " (32 latent + 2 yaw_pred, YAW HEAD)"
    return None, evidence + " — neither 32 nor 34, cannot classify"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_manifest(ckpt_path, cli):
    manifest_path = ckpt_path.parent / MANIFEST_NAME
    if not manifest_path.exists():
        warn(f"no {MANIFEST_NAME} in {ckpt_path.parent} — manifest check "
             "skipped (consider adding one; only the dim probe protects "
             "this launch)")
        return
    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        warn(f"could not read/parse {manifest_path}: {exc} — manifest "
             "check skipped")
        return
    info(f"manifest: {manifest_path}")

    # Hard contract fields — mismatch fails the launch.
    for field in HARD_FIELDS:
        if field not in manifest:
            warn(f"manifest missing hard field '{field}'; not checked")
            continue
        want = as_int01(manifest[field], field)
        if want is None:
            continue
        got = cli[field]
        if want != got:
            hint = ""
            if field == "zero_vel_obs" and want == 1:
                hint = (" (this checkpoint was trained with "
                        "zero_student_velocity_obs=True — pass "
                        "--zero_student_velocity_obs 1 / ZERO_VEL_OBS=1)")
            elif field == "yaw_head":
                hint = (" (34-D vs 32-D encoder mismatch breaks strict "
                        "load / silently corrupts the latent)")
            elif field == "env_class_eq_17":
                hint = (" (env-class one-hot bits prop[11]/prop[12] must "
                        "match training terrain)")
            err(f"hard contract mismatch: manifest {field}={want} but "
                f"launch flags say {field}={got}{hint}")
        else:
            info(f"hard field ok: {field}={got}")

    # Soft fields compared against CLI flags — warn only.
    for field in SOFT_COMPARED_FIELDS:
        if field not in manifest:
            continue
        rec_raw = manifest[field]
        got = cli[field]
        try:
            rec = float(rec_raw)
            mismatch = abs(rec - float(got)) > 1e-6
        except (ValueError, TypeError):
            rec = rec_raw
            mismatch = str(rec_raw).strip() != str(got)
        if mismatch:
            warn(f"soft field: manifest recommends {field}={rec_raw} but "
                 f"launch uses {field}={got} (not fatal, but off the "
                 "validated configuration)")
        else:
            info(f"soft field ok: {field}={got}")

    # Recommended settings without a CLI counterpart here — informational.
    for field in SOFT_INFO_FIELDS:
        if field in manifest:
            info(f"manifest recommends {field}: {manifest[field]}")

    # Validated checkpoint list — warn if this exact ckpt was never
    # sim-validated.
    validated_raw = manifest.get("sim_validated_ckpts")
    if validated_raw is not None:
        validated = [v.strip() for v in str(validated_raw).split(",")
                     if v.strip()]
        stem = ckpt_path.stem  # e.g. model_8000
        if validated and stem not in validated:
            warn(f"checkpoint {ckpt_path.name} is not in the "
                 f"sim-validated list ({', '.join(validated)}) — "
                 "run sim2sim before deploying it on hardware")
        elif stem in validated:
            info(f"checkpoint {ckpt_path.name} is sim-validated")


def check_dim_probe(ckpt_path, cli):
    has_yaw, evidence = probe_yaw_head(ckpt_path)
    if has_yaw is None:
        warn(f"yaw-head dim probe inconclusive: {evidence}")
        return
    info(f"dim probe: {evidence}")
    probed = int(has_yaw)
    if probed != cli["yaw_head"]:
        err(f"yaw-head mismatch: checkpoint encoder is "
            f"{'34-D (yaw head)' if has_yaw else '32-D (no yaw head)'} "
            f"but launch flags say yaw_head={cli['yaw_head']} — the "
            "depth-encoder node would fail strict load or feed a "
            "wrong-shaped latent")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def flag01(value):
    iv = int(value)
    if iv not in (0, 1):
        raise argparse.ArgumentTypeError(f"expected 0 or 1, got {value!r}")
    return iv


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reject deploy launches whose flags contradict the "
                    "checkpoint's observation contract.")
    ap.add_argument("--ckpt", required=True,
                    help="path to the .pt checkpoint being deployed")
    ap.add_argument("--env_class_eq_17", required=True, type=flag01)
    ap.add_argument("--yaw_head", required=True, type=flag01)
    ap.add_argument("--zero_vel_obs", required=True, type=flag01)
    ap.add_argument("--binary_vel", required=True, type=flag01)
    ap.add_argument("--lin_vel_max", required=True, type=float)
    args = ap.parse_args(argv)

    ckpt_path = Path(args.ckpt).expanduser()
    cli = {
        "yaw_head": args.yaw_head,
        "env_class_eq_17": args.env_class_eq_17,
        "zero_vel_obs": args.zero_vel_obs,
        "binary_vel": args.binary_vel,
        "lin_vel_max": args.lin_vel_max,
    }

    print(f"deploy-contract check: ckpt={ckpt_path}")
    print("launch flags: " + ", ".join(f"{k}={v}" for k, v in cli.items()))

    if not ckpt_path.is_file():
        err(f"checkpoint file not found: {ckpt_path}")
    else:
        check_manifest(ckpt_path, cli)
        check_dim_probe(ckpt_path, cli)

    if _errors:
        print(f"\nCONTRACT MISMATCH — {len(_errors)} error(s), "
              f"{len(_warnings)} warning(s). Refusing launch.")
        return 1
    print(f"\nCONTRACT OK ({len(_warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
