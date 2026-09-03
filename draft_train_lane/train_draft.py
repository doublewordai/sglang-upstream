#!/usr/bin/env python3
"""Train the NextN draft. torchrun --nproc-per-node=3 train_draft.py --data synthetic|<capdir>

Loads extracted real draft weights (draft_weights/), FSDP full-shards the model over
the local GPUs, fine-tunes with CE + EAGLE feature loss on windows of
(tokens, teacher-forced target hiddens), reports val top-1/top-4 draft accuracy on
held-out sessions, and saves a rank-0 consolidated state dict for export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draft_data import RealData, SyntheticData, make_batch  # noqa: E402
from draft_model import HIDDEN, DraftNextN, chain_loss, draft_loss  # noqa: E402


def build_model(weights_dir: str, device) -> DraftNextN:
    from safetensors.torch import load_file

    small = load_file(os.path.join(weights_dir, "draft.safetensors"))
    experts = load_file(os.path.join(weights_dir, "experts.safetensors"))
    vocab, _ = small["embed"].shape
    m = DraftNextN(vocab, small["embed"], small["lm_head"])
    sd = m.state_dict()
    mapping = {
        "enorm": "enorm",
        "hnorm": "hnorm",
        "eh_proj.weight": "eh_proj",
        "input_ln": "input_ln",
        "post_attn_ln": "post_attn_ln",
        "shared_head_norm": "shared_head_norm",
        "attn.q_a.weight": "q_a",
        "attn.q_a_ln": "q_a_ln",
        "attn.q_b.weight": "q_b",
        "attn.kv_a.weight": "kv_a",
        "attn.kv_a_ln": "kv_a_ln",
        "attn.kv_b": "kv_b",
        "attn.o.weight": "o_proj",
        "moe.gate.w": "gate_w",
        "moe.gate.bias": "gate_bias",
        "moe.shared.gate": "s_gate",
        "moe.shared.up": "s_up",
        "moe.shared.down": "s_down",
        "moe.eg.w": "e_gate",
        "moe.eu.w": "e_up",
        "moe.ed.w": "e_down",
    }
    loaded = {}
    for model_key, ckpt_key in mapping.items():
        assert ckpt_key in small or ckpt_key in experts, f"missing {ckpt_key}"
        src = small.get(ckpt_key, experts.get(ckpt_key))
        loaded[model_key] = src
    missing, unexpected = m.load_state_dict(loaded, strict=False)
    real_missing = [k for k in missing if "embed" not in k and "lm_head" not in k]
    assert not real_missing, f"missing params: {real_missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    # trainable params fp32 (FSDP units need uniform dtype); expert buffers stay
    # bf16 (frozen, replicated; not flattened by FSDP)
    for p in m.parameters():
        p.data = p.data.float()
    m.moe.eg.w.data = m.moe.eg.w.data.to(torch.bfloat16)
    m.moe.eu.w.data = m.moe.eu.w.data.to(torch.bfloat16)
    m.moe.ed.w.data = m.moe.ed.w.data.to(torch.bfloat16)
    return m.to(device)


def all_reduce_scalar(v, device):
    t = torch.tensor([v], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic")
    ap.add_argument("--weights", default="/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/draft_weights")
    ap.add_argument("--out", default="runs/synthetic")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--feature-weight", type=float, default=1.0)
    ap.add_argument("--chain-weight", type=float, default=0.0,
                    help=">0 mixes EAGLE-3.1-style chain-rollout loss (depth stability)")
    ap.add_argument("--chain-len", type=int, default=8)
    ap.add_argument("--chains-per-window", type=int, default=2)
    ap.add_argument("--chain-detach", action="store_true",
                    help="detach the draft's own hidden feedback (no BPTT through the chain)")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--val-windows", type=int, default=16)
    ap.add_argument("--holdout", type=int, default=6,
                    help="sessions held out for val (0 = train on everything)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", default="fsdp", choices=["fsdp", "ddp", "none"])
    ap.add_argument("--val-only", action="store_true")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    torch.manual_seed(args.seed + rank)

    model = build_model(args.weights, device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"model loaded: {nparams/1e9:.2f}B trainable params")

    if args.strategy == "fsdp" and world > 1:
        import functools
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            device_id=device,
            sync_module_states=True,
            use_orig_params=True,
            auto_wrap_policy=functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000),
        )
    elif args.strategy == "ddp" and world > 1:
        model = DDP(model, device_ids=[rank])

    raw_model = model.module if hasattr(model, "module") else model

    # data
    if args.data == "synthetic":
        data = SyntheticData(seed=args.seed)
        get = data.get
        train_pick = lambda rng, n: data.windows(n, args.window, rng, "train")
        val_pick = lambda rng, n: data.windows(n, args.window, rng, "val")
    else:
        data = RealData(args.data, window=args.window, seed=args.seed,
                        holdout_sessions=args.holdout)
        get = data.get
        train_pick = lambda rng, n: [data.train_idx[i] for i in rng.choice(len(data.train_idx), n)]
        def val_pick(rng, n):
            if not data.val_idx:
                return []
            return [data.val_idx[i] for i in rng.choice(len(data.val_idx), min(n, len(data.val_idx)))]

    decay, no_decay = [], []
    for n, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "lr": args.lr, "weight_decay": 0.01},
            {"params": no_decay, "lr": args.lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed + 1000 * rank)
    log_path = os.path.join(args.out, f"log-{rank}.jsonl")
    logf = open(log_path, "a")

    def evaluate():
        model.eval()
        vrng = np.random.RandomState(1234)
        items = val_pick(vrng, args.val_windows)
        # split val across ranks
        items = items[rank::world]
        if not items:
            model.train()
            return {}
        tokens, prev, pos = make_batch(get, items, args.window, device)
        with torch.autocast("cuda", torch.bfloat16):
            loss, m = draft_loss(
                model, tokens, prev, pos, args.feature_weight, return_metrics=True
            )
        model.train()
        for k in m:
            m[k] = all_reduce_scalar(m[k], device)
        return m

    if args.val_only:
        m = evaluate()
        if rank == 0:
            print("VAL", json.dumps(m))
            logf.write(json.dumps({"step": -1, "val": m}) + "\n")
        dist.barrier()
        dist.destroy_process_group()
        return

    t0 = time.time()
    step_tokens = args.micro_bs * args.window
    for step in range(1, args.steps + 1):
        model.train()
        items = train_pick(rng, args.micro_bs)
        tokens, prev, pos = make_batch(get, items, args.window, device)
        with torch.autocast("cuda", torch.bfloat16):
            loss, m = draft_loss(
                model, tokens, prev, pos, args.feature_weight, return_metrics=True
            )
            if args.chain_weight > 0:
                closs, cm = chain_loss(
                    model,
                    tokens[:1],
                    prev[:1],
                    pos[:1],
                    args.feature_weight,
                    chain_len=args.chain_len,
                    n_chains=args.chains_per_window,
                    detach_feedback=args.chain_detach,
                    return_metrics=True,
                )
                loss = loss + args.chain_weight * closs
                m = dict(m)
                m["chain_ce"] = cm["ce"]
                m["chain_top1_d0123"] = cm["top1_by_depth"][:4]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0 and rank == 0:
            dt = time.time() - t0
            msg = {
                "step": step,
                "ce": m["ce"],
                "mse": m["mse"],
                "top1": m["top1"],
                "tok_s": step * 10 * step_tokens / max(dt, 1e-9),
            }
            if "chain_ce" in m:
                msg["chain_ce"] = m["chain_ce"]
                msg["chain_top1_by_depth"] = [
                    round(x, 4) for x in m["chain_top1_d0123"]
                ]
            print(json.dumps(msg), flush=True)
            logf.write(json.dumps(msg) + "\n")
            logf.flush()

        if step % args.val_every == 0 or step == args.steps:
            m = evaluate()
            if rank == 0:
                print("VAL", json.dumps(m), flush=True)
                logf.write(json.dumps({"step": step, "val": m}) + "\n")
                logf.flush()

        if step % args.save_every == 0 or step == args.steps:
            # FSDP full-state-dict consolidation (collective: ALL ranks call
            # state_dict under the context; rank0_only gets the full CPU dict)
            if args.strategy == "fsdp" and world > 1:
                from torch.distributed.fsdp import (
                    FullStateDictConfig,
                    StateDictType,
                )

                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                    sd = model.state_dict()
            else:
                sd = {
                    k: v.detach().to(torch.bfloat16).cpu()
                    for k, v in raw_model.state_dict().items()
                }
            if rank == 0:
                torch.save(sd, os.path.join(args.out, "draft_finetuned.pt"))
                print(f"saved step {step}", flush=True)
            dist.barrier()

    if rank == 0:
        print("DONE")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import numpy as np

    main()
