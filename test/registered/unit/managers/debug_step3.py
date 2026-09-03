#!/usr/bin/env python3
"""Debug3: layer-0 positions 0..3 on both stacks; first-divergence bisection."""
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_hisparse_prefetch_spec as T
from test_hisparse_prefetch_spec import FILL_LEN, N_POS, _make_req

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


def snap(stack):
    c = stack.coordinator
    return (
        c.req_device_buffer_tokens[0].clone(),
        c.lru_slots[0].clone(),
        stack.device_pool.kv_buffer[0].clone(),
    )


prev = {n: snap(s) for n, s in (("sync", self.sync), ("pre", self.prefetch))}
for p in range(N_POS):
    outs = {}
    for name, stack in (("sync", self.sync), ("pre", self.prefetch)):
        c = stack.coordinator
        c.num_real_reqs[0] = 2
        pt = c.swap_in_selected_pages(
            rpi, sls + (p + 1), tks[p], 0, num_newest=p + 1
        )
        outs[name] = pt.clone()
    torch.cuda.synchronize()
    now = {n: snap(s) for n, s in (("sync", self.sync), ("pre", self.prefetch))}
    print(f"--- after (layer0, p{p}):")
    print("  pt equal:", torch.equal(outs["sync"], outs["pre"]))
    print("  dbt equal:", torch.equal(now["sync"][0], now["pre"][0]))
    print("  lru equal:", torch.equal(now["sync"][1], now["pre"][1]))
    print("  kv  equal:", torch.equal(now["sync"][2], now["pre"][2]))
    mc = self.prefetch.coordinator._miss_count_v[p, :2].tolist()
    print("  pre miss:", mc)
    if not torch.equal(now["sync"][0], now["pre"][0]):
        d = now["sync"][0] != now["pre"][0]
        rows, cols = d.nonzero(as_tuple=True)
        for i in range(min(8, rows.numel())):
            r, c_ = int(rows[i]), int(cols[i])
            print(
                f"    req{r} slot{c_}: sync={int(now['sync'][0][r,c_])} "
                f"pre={int(now['pre'][0][r,c_])}"
            )
        break
    prev = now
