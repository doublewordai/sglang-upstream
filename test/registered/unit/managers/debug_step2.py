#!/usr/bin/env python3
"""Debug2: single (layer0,pos0) kernel call, sync vs prefetch stack, full diff."""
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

results = {}
for name, stack in (("sync", self.sync), ("pre", self.prefetch)):
    c = stack.coordinator
    c.num_real_reqs[0] = 2
    pt = c.swap_in_selected_pages(
        rpi, sls + 1, tks[0], 0, num_newest=1
    )
    results[name] = dict(
        pt=pt.clone(),
        dbt=c.req_device_buffer_tokens[0].clone(),
        lru=c.lru_slots[0].clone(),
        kv=stack.device_pool.kv_buffer[0].clone(),
        miss0=(
            c._miss_count_v[0, :2].tolist()
            if c._miss_count_v is not None
            else None
        ),
    )
torch.cuda.synchronize()

a, b = results["sync"], results["pre"]
print("pt equal:", torch.equal(a["pt"], b["pt"]))
print("dbt equal:", torch.equal(a["dbt"], b["dbt"]))
print("lru equal:", torch.equal(a["lru"], b["lru"]))
print("kv equal:", torch.equal(a["kv"], b["kv"]))
print("pre miss0:", b["miss0"])
d = (a["dbt"] != b["dbt"])
print("dbt diffs:", int(d.sum()))
if int(d.sum()):
    rows, cols = d.nonzero(as_tuple=True)
    for i in range(min(10, rows.numel())):
        r, c_ = int(rows[i]), int(cols[i])
        print(
            f"  req{r} slot{c_}: sync={int(a['dbt'][r,c_])} pre={int(b['dbt'][r,c_])}"
        )
# what tokens did pos0 select that are >= dbs (should miss)?
sel = tks[0]
for r in range(2):
    hi = (sel[r] >= 1024).sum().item()
    print(f"req{r}: selected {int(hi)} tokens >= dbs(1024); drafts in set:",
          int((sel[r] >= FILL_LEN).sum().item()))
# lru diff detail
dl = (a["lru"] != b["lru"])
print("lru diffs:", int(dl.sum()))
if int(dl.sum()):
    rows, cols = dl.nonzero(as_tuple=True)
    for i in range(min(6, rows.numel())):
        r, c_ = int(rows[i]), int(cols[i])
        print(
            f"  req{r} lru[{c_}]: sync={int(a['lru'][r,c_])} pre={int(b['lru'][r,c_])}"
        )
