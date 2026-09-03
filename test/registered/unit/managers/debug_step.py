#!/usr/bin/env python3
"""Debug: one verify step, sync vs prefetch, dump the first table difference."""
import os
import random
import sys
from array import array
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hisparse_prefetch_spec as T
from test_hisparse_prefetch_spec import (
    FILL_LEN,
    LAYER_NUM,
    N_POS,
    PATTERN,
    _make_req,
)

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29601")
torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)

T.TestHiSparseVerifyPrefetchSpec.setUpClass()
self = T.TestHiSparseVerifyPrefetchSpec("test_verify_prefetch_matches_sync")

for stack in (self.sync, self.prefetch):
    stack.reset()
reqs = [_make_req(f"dbg-{i}", list(range(FILL_LEN))) for i in range(2)]
for stack in (self.sync, self.prefetch):
    for req in reqs:
        stack.req_to_token_pool.alloc([req])
        self._admit(stack, req, FILL_LEN)

seq_lens = [FILL_LEN, FILL_LEN]
rng = random.Random(11)
tks = self._make_selections(rng, seq_lens)
rpi = torch.tensor([r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda")
sls = torch.tensor(seq_lens, dtype=torch.int64, device="cuda")

for stack in (self.sync, self.prefetch):
    per_req = [
        self._alloc_step_tokens(stack, req, seq_lens[i], N_POS)
        for i, req in enumerate(reqs)
    ]
    out_cache_loc = torch.cat(per_req, dim=0).to("cuda")
    stack.coordinator.map_verify_locs_to_buffer(
        sls, out_cache_loc, rpi, sls.cpu(), rpi.cpu(), N_POS
    )

outs_sync, _ = self._verify_step(self.sync, rpi, sls, tks)
outs_pre, _ = self._verify_step(self.prefetch, rpi, sls, tks)
torch.cuda.synchronize()

for key in outs_sync:
    a, b = outs_sync[key], outs_pre[key]
    if not torch.equal(a, b):
        d = (a != b)
        print(f"DIFF at {key}: {int(d.sum())} of {a.numel()} entries")
        rows, cols = d.nonzero(as_tuple=True)
        for i in range(min(6, rows.numel())):
            r, c = int(rows[i]), int(cols[i])
            tok = int(tks[key[1]][r, c].item())
            print(
                f"  req{r} idx{c} token={tok}: sync={int(a[r,c].item())} "
                f"prefetch={int(b[r,c].item())}"
            )
        # check state before this layer/position on both stacks
        lid = key[0]
        print(
            "  lru equal before:",
            torch.equal(
                self.sync.coordinator.lru_slots[lid],
                self.prefetch.coordinator.lru_slots[lid],
            ),
        )
        print(
            "  dbt equal before:",
            torch.equal(
                self.sync.coordinator.req_device_buffer_tokens[lid],
                self.prefetch.coordinator.req_device_buffer_tokens[lid],
            ),
        )
        print(
            "  dblocs equal:",
            torch.equal(
                self.sync.coordinator.req_device_buffer_token_locs[lid],
                self.prefetch.coordinator.req_device_buffer_token_locs[lid],
            ),
        )
        break
else:
    print("ALL TABLES EQUAL")

# also dump miss counts per position (prefetch stack, layer 0 plans)
for p in range(N_POS):
    print(
        f"miss_count_v[{p}] =",
        self.prefetch.coordinator._miss_count_v[p, : len(reqs)].tolist(),
    )
