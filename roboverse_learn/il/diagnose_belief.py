"""Open-loop belief / decoder / flow error decomposition for B2A-family policies.

Loads a trained checkpoint and runs `policy.eval_belief_diagnostics` over the
train and held-out (validation) splits of the SAME dataset the policy was
trained on. Because it has ground-truth future actions, it can answer the
question closed-loop rollout cannot: is the policy's error coming from the
belief mean (overfitting / poor generalization) or from the action decoder
(reconstruction floor), and how much does the flow + source noise add?

The error budget reported (action-space L1) is additive:

    decoder floor       = decode(true target latent)  vs GT action
    + belief gap        = decode(belief mu)            - decoder floor
    + flow/noise gap     = decode(flow output)          - decode(belief mu)
    = full policy error  = decode(flow output)          vs GT action

Comparing the train column against the val column isolates which component
fails to generalize.

Usage:
    conda activate a2a
    python roboverse_learn/il/diagnose_belief.py \
        --ckpt il_outputs/b2a_bench/<run>/conditional_fm/<method>/nfe1/<task>/checkpoints/100.ckpt

    # several checkpoints at once:
    python roboverse_learn/il/diagnose_belief.py --ckpt A/100.ckpt B/100.ckpt
"""

import argparse
import pathlib

import dill
import numpy as np
import torch

# Ensure repo root on path when invoked directly.
import sys
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra  # noqa: E402
from roboverse_learn.il.runners.default_runner import create_dataloader  # noqa: E402


def _load_policy(payload, device):
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy_config)
    state_dicts = payload["state_dicts"]
    key = "ema_model" if "ema_model" in state_dicts else "model"
    missing, unexpected = policy.load_state_dict(state_dicts[key], strict=False)
    if missing:
        print(f"  [warn] missing keys when loading {key}: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    policy.to(device)
    policy.eval()
    return policy, cfg, key


def _run_split(policy, dataset, dataloader, device, max_batches):
    acc = {}
    total_n = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = dataset.postprocess(batch, device)
        out = policy.eval_belief_diagnostics(batch)
        w = out.pop("n")
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + v * w
        total_n += w
        n_batches += 1
        if max_batches and n_batches >= max_batches:
            break
    if total_n == 0:
        return None, 0, 0
    return {k: v / total_n for k, v in acc.items()}, int(total_n), n_batches


def _fmt(train, val, key):
    t = train.get(key, float("nan"))
    v = val.get(key, float("nan"))
    gap = v - t
    return f"{t:>12.6f}  {v:>12.6f}  {gap:>+12.6f}"


def diagnose(ckpt_path, device, max_batches, zarr_override):
    ckpt_path = pathlib.Path(ckpt_path)
    print("=" * 78)
    print(f"checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path.open("rb"), pickle_module=dill, map_location="cpu")
    policy, cfg, which = _load_policy(payload, device)
    print(f"  policy: {cfg.policy_config._target_.split('.')[-1]}  (weights: {which})")
    print(f"  deterministic_eval={getattr(policy, 'deterministic_eval', '?')}  "
          f"source_noise_std={getattr(policy, 'source_noise_std', 0.0)}  "
          f"num_sampling_steps={policy.num_sampling_steps}")

    if not hasattr(policy, "eval_belief_diagnostics"):
        print("  [skip] policy has no eval_belief_diagnostics (not a B2A-family policy)")
        return

    if zarr_override:
        cfg.dataset_config.zarr_path = zarr_override
    print(f"  dataset: {cfg.dataset_config.zarr_path}")

    dataset = hydra.utils.instantiate(cfg.dataset_config)
    val_dataset = dataset.get_validation_dataset()

    def loader(ds, base_cfg):
        kw = dict(base_cfg)
        kw["num_workers"] = 0
        kw["persistent_workers"] = False
        kw["shuffle"] = False
        return create_dataloader(ds, **kw)

    train_loader = loader(dataset, cfg.train_config.dataloader)
    val_loader = loader(val_dataset, cfg.train_config.val_dataloader)

    train, nt, bt = _run_split(policy, dataset, train_loader, device, max_batches)
    val, nv, bv = _run_split(policy, dataset, val_loader, device, max_batches)
    if train is None or val is None:
        print("  [skip] empty split")
        return

    print(f"\n  samples: train={nt} ({bt} batches)   val={nv} ({bv} batches)")
    print(f"\n  {'metric':<34}{'train':>12}  {'val(held-out)':>12}  {'gen.gap':>12}")
    print("  " + "-" * 72)

    print("  -- latent-space MSE to true target latent --")
    for k in ["belief_mu_target_mse", "source_target_mse", "flow_target_mse"]:
        print(f"  {k:<34}{_fmt(train, val, k)}")

    print("  -- action L1 (normalized units) --")
    for k in ["decode_target_l1", "decode_mu_l1", "decode_flow_l1"]:
        print(f"  {k:<34}{_fmt(train, val, k)}")

    print("  -- action L1 (raw action units) --")
    for k in ["decode_target_l1_raw", "decode_mu_l1_raw", "decode_flow_l1_raw"]:
        print(f"  {k:<34}{_fmt(train, val, k)}")

    print("  -- belief internals at eval --")
    for k in ["belief_avg_std", "belief_gate"]:
        print(f"  {k:<34}{_fmt(train, val, k)}")

    # Error budget (val split, normalized action L1).
    floor = val["decode_target_l1"]
    belief_gap = val["decode_mu_l1"] - floor
    flow_gap = val["decode_flow_l1"] - val["decode_mu_l1"]
    full = val["decode_flow_l1"]
    print(f"\n  error budget on held-out (normalized action L1):")
    print(f"    decoder floor      : {floor:.6f}  ({100*floor/full:5.1f}% of full)")
    print(f"    + belief gap       : {belief_gap:+.6f}  ({100*belief_gap/full:5.1f}%)")
    print(f"    + flow/noise gap    : {flow_gap:+.6f}  ({100*flow_gap/full:5.1f}%)")
    print(f"    = full policy error : {full:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoint paths")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all batches")
    ap.add_argument("--zarr", default=None, help="override dataset zarr_path")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = args.device if torch.cuda.is_available() else "cpu"
    for ckpt in args.ckpt:
        diagnose(ckpt, device, args.max_batches, args.zarr)


if __name__ == "__main__":
    main()
