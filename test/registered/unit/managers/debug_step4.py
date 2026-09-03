#!/usr/bin/env python3
"""Debug4: full step-0, dump (1,0) skip-layer table diff + stash contents."""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hisparse_prefetch_spec as T
from test_hisparse_prefetch_spec import FILL_LEN, LAYER_NUM, N_POS, _make_req

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
print("req_pool_idx after both allocs:", [r.req_pool_idx for r in reqs])

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
    same = torch.equal(outs_sync[key], outs_pre[key])
    print(key, "equal" if same else "DIFF")
    if not same:
        a, b = outs_sync[key], outs_pre[key]
        d = a != b
        rows, cols = d.nonzero(as_tuple=True)
        print(f"  {int(d.sum())} diffs; first:")
        for i in range(min(6, rows.numel())):
            r, c = int(rows[i]), int(cols[i])
            tok = int(tks[key[1]][r, c].item())
            print(
                f"  req{r}(pool{int(rpi[r])}) idx{c} token={tok}: "
                f"sync={int(a[r,c].item())} pre={int(b[r,c].item())}"
            )
        # stash vs sync layer-0 same-position table
        p = key[1]
        stash = self.prefetch.coordinator._verify_slot_table[p][: rpi.shape[0]]
        print(
            f"  stash[{p}] == sync(0,{p}):",
            torch.equal(stash, outs_sync[(0, p)]),
        )
        print(
            f"  stash[{p}] == sync({key[0]},{p}):",
            torch.equal(stash, outs_sync[key]),
        )
        # what does the sync stack's layer-1 state look like vs layer 0?
        s = self.sync.coordinator
        print(
            "  sync l1.dbt==l0.dbt:",
            torch.equal(s.req_device_buffer_tokens[1], s.req_device_buffer_tokens[0]),
        )
        print(
            "  sync l1.lru==l0.lru:",
            torch.equal(s.lru_slots[1], s.lru_slots[0]),
        )
        pr = self.prefetch.coordinator
        print(
            "  pre  l1.dbt==init(-1/arange?):",
            pr.req_device_buffer_tokens[1, :2, :4].tolist(),
        )
        break
