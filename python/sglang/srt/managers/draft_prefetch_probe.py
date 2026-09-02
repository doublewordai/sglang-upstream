
"""Draft-prefetch probe (lane/draft-prefetch, SGLANG_DPF_PROBE=1).

Per verify step under hisparse + speculation, stash on device (stream-ordered,
CUDA-graph safe):
  - the draft seed top-k (the draft layer's indexer selection at the last
    accepted position, published into EagleDraftInput.dsa_topk_indices by
    draft-extend and reused by the draft steps under
    index_share_for_mtp_iteration) -- the M2 prefetch input;
  - the target's per-layer, per-draft-position top-k (positions);
  - the per-layer resident-token table before the layer's first swap-in;
  - per-(layer, position) miss counts (kernel miss-plan recording) and
    CUDA-event timings around each swap-in launch.

After the step (host, post-sync) compute per layer: Jaccard / recall of the
draft seed vs the target selection, the baseline resident hit rate, the
predicted hit rate if the seed were prefetched (target ∩ (resident ∪ seed)),
and append one JSON line per step to the output file.

Dummy weights validate the plumbing only; real hit rates need real weights
(run the same env on a real system).
"""

import json
import logging
import os
import time
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DraftPrefetchProbe:
    def __init__(
        self,
        coordinator,
        out_path: str,
        probe_reqs: int = 2,
        raw_steps: int = 0,
    ):
        self.coord = coordinator
        assert not coordinator.is_dsv4_hisparse, "DPF probe: DSA hisparse only"
        self.out_path = out_path or "/tmp/dpf_probe.jsonl"
        self.probe_reqs = max(1, int(probe_reqs))
        self.raw_steps = int(raw_steps)
        self.layer_num = int(coordinator.mem_pool_device.layer_num)
        self.top_k = int(coordinator.top_k)
        self.dbs = int(coordinator.device_buffer_size)
        self.pad_buf = int(coordinator.padded_buffer_size)
        self.device = coordinator.device
        self.step = 0
        self.n = None  # speculative_num_draft_tokens, learned at first layer

        # per-step state
        self.seed = None  # [bs, K] int32 (draft seed positions) or None
        self.seq_lens = None  # [pbs] int (pre-verify S)
        self.req_pool_indices = None  # [pbs]

        # lazily allocated device stashes (fixed addresses afterwards)
        self._alloc = False
        self.stash_tk = None  # [L, pbs, n, K] int32
        self.stash_res = None  # [L, pbs, dbs] int32
        self.stash_miss = None  # [L, pbs, n] int32
        self.stash_seq = None  # [pbs] int64
        self.stash_req = None  # [pbs] int64
        self.plan_src = None  # [rows, K] int64  (shared kernel scratch)
        self.plan_dst = None  # [rows, K] int32
        self.plan_count = None  # [rows] int32
        self.events = None  # [L][p] -> (start, end)

    # -- device stash helpers ------------------------------------------------
    def _ensure_alloc(self, bs: int, n: int) -> None:
        if self._alloc:
            assert n == self.n, f"DPF probe: draft-token count changed {self.n}->{n}"
            return
        self.n = n
        pbs = min(self.probe_reqs, bs)
        assert pbs >= 1
        self.pbs = pbs
        dev = self.device
        self.stash_tk = torch.zeros(
            (self.layer_num, pbs, n, self.top_k), dtype=torch.int32, device=dev
        )
        self.stash_res = torch.zeros(
            (self.layer_num, pbs, self.dbs), dtype=torch.int32, device=dev
        )
        self.stash_miss = torch.zeros(
            (self.layer_num, pbs, n), dtype=torch.int32, device=dev
        )
        self.stash_seq = torch.zeros(pbs, dtype=torch.int64, device=dev)
        self.stash_req = torch.zeros(pbs, dtype=torch.int64, device=dev)
        rows = max(1024, bs)
        self.plan_src = torch.zeros((rows, self.top_k), dtype=torch.int64, device=dev)
        self.plan_dst = torch.zeros((rows, self.top_k), dtype=torch.int32, device=dev)
        self.plan_count = torch.zeros(rows, dtype=torch.int32, device=dev)
        ev_cls = torch.cuda.Event
        self.events = [
            [
                (ev_cls(enable_timing=True), ev_cls(enable_timing=True))
                for _ in range(n)
            ]
            for _ in range(self.layer_num)
        ]
        self._alloc = True
        logger.info(
            "DPF probe: alloc L=%d pbs=%d n=%d K=%d dbs=%d rows=%d",
            self.layer_num, pbs, n, self.top_k, self.dbs, rows,
        )

    # -- hooks (called from the verify forward; graph-capture safe) ----------
    def on_draft_seed(self, seed: Optional[torch.Tensor]) -> None:
        """EAGLEWorkerV2.forward_batch_generation, before draft()."""
        self.seed = seed  # keep the ref; content read post-sync at finalize

    def on_verify_layer(
        self,
        layer_id: int,
        tk: torch.Tensor,  # [bs, n, K] int (positions)
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        bs: int,
        n: int,
    ) -> None:
        """dsa_backend verify branch, before the per-position swap-in loop."""
        self._ensure_alloc(bs, n)
        assert bs <= self.plan_src.shape[0], (
            f"DPF probe: bs {bs} > plan rows {self.plan_src.shape[0]}"
        )
        pbs = self.pbs
        self.stash_tk[layer_id, :pbs].copy_(tk[:pbs])
        # resident tokens of the probed reqs, before this layer's swap-ins
        rpi = req_pool_indices[:pbs]
        res = self.coord.req_device_buffer_tokens[layer_id]  # [slots, buf]
        self.stash_res[layer_id, :pbs].copy_(res[rpi, : self.dbs])
        if layer_id == 0:
            self.stash_seq.copy_(seq_lens[:pbs])
            self.stash_req.copy_(rpi)

    def events_for(self, layer_id: int, p: int) -> Optional[Tuple]:
        if not self._alloc:
            return None
        return self.events[layer_id][p]

    def plan(self) -> Optional[Dict[str, torch.Tensor]]:
        """Miss-plan buffers for the swap-in kernel (verify path)."""
        if not self._alloc:
            return None
        return dict(
            miss_src=self.plan_src,
            miss_dst=self.plan_dst,
            miss_count=self.plan_count,
        )

    def after_swap_in(self, layer_id: int, p: int) -> None:
        """Stream-ordered (post-kernel) copy of this call's miss counts."""
        self.stash_miss[layer_id, : self.pbs, p].copy_(
            self.plan_count[: self.pbs]
        )

    # -- host-side finalize ---------------------------------------------------
    def finalize_step(self, seq_lens_cpu, accept_lens) -> None:
        if not self._alloc:
            return
        torch.cuda.synchronize()
        t0 = time.time()
        pbs, n, K, L = self.pbs, self.n, self.top_k, self.layer_num
        seed_cpu = self.seed.cpu() if self.seed is not None else None
        tk_cpu = self.stash_tk.cpu()
        res_cpu = self.stash_res.cpu()
        miss_cpu = self.stash_miss.cpu()
        seq = self.stash_seq.cpu().tolist()
        reqs = self.stash_req.cpu().tolist()
        accept = (
            accept_lens.cpu().tolist() if accept_lens is not None else None
        )

        layers = []
        raw = None
        if self.step < self.raw_steps:
            raw = {
                "seed": seed_cpu.tolist() if seed_cpu is not None else None,
                "target": tk_cpu.tolist(),
                "resident": res_cpu.tolist(),
            }
        for l in range(L):
            entry = {"l": l, "jac": [], "recall": [], "waste": [],
                     "res_size": [], "base_hit": [], "pref_hit": [],
                     "miss_ct": [], "win_ct": [], "t_us": []}
            for r in range(pbs):
                S = seq[r]
                seed_set = (
                    {int(x) for x in seed_cpu[r].tolist() if int(x) >= 0}
                    if seed_cpu is not None
                    else set()
                )
                T = [int(x) for x in tk_cpu[l, r, 0].tolist() if int(x) >= 0]
                T_set = set(T)
                win = sum(1 for x in T if x >= S)
                inter = len(seed_set & T_set)
                union = len(seed_set | T_set)
                R = {
                    int(x)
                    for x in res_cpu[l, r].tolist()
                    if int(x) >= 0
                }
                To = T_set - {x for x in T if x >= S}  # old positions only
                entry["jac"].append(round(inter / union, 4) if union else None)
                entry["recall"].append(round(inter / len(T_set), 4) if T_set else None)
                entry["waste"].append(
                    round(len(seed_set - T_set) / len(seed_set), 4) if seed_set else None
                )
                entry["res_size"].append(len(R))
                entry["base_hit"].append(
                    round(len(To & R) / len(To), 4) if To else None
                )
                entry["pref_hit"].append(
                    round(len(To & (R | seed_set)) / len(To), 4) if To else None
                )
                entry["win_ct"].append(win)
                entry["miss_ct"].append([int(miss_cpu[l, r, p]) for p in range(n)])
                entry["t_us"].append(
                    [
                        round(
                            self.events[l][p][0].elapsed_time(self.events[l][p][1])
                            * 1000.0,
                            1,
                        )
                        for p in range(n)
                    ]
                )
            layers.append(entry)

        row = {
            "step": self.step,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pbs": pbs,
            "n": n,
            "K": K,
            "dbs": self.dbs,
            "seq_lens": seq,
            "req_pool_indices": reqs,
            "accept_lens": accept,
            "seed_present": seed_cpu is not None,
            "layers": layers,
        }
        if raw is not None:
            row["raw"] = raw
        with open(self.out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.step += 1
        # drop the seed ref; the next draft() re-publishes
        self.seed = None
        logger.info(
            "DPF probe step %d written (%.1f ms analysis)", self.step,
            (time.time() - t0) * 1e3,
        )
