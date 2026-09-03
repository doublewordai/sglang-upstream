#!/usr/bin/env python3
"""Unit test of eval_draft's per-segment alignment+classification on REAL
corpus text (the dummy capture uses random tokens and always skips)."""
import sys

import numpy as np

sys.path.insert(0, "/scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train")
from segment_shares import FENCE, TOOL_JSON  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TK = "/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516/tokenizer.json"
tk = Tokenizer.from_file(TK)
text = open(
    "/scratch/s6p/fergus.s6p/grace-1m/lanes/workload/out/corpus_pi.txt",
    encoding="utf-8",
    errors="replace",
).read(400_000)

enc = tk.encode(text)
t = np.array(enc.ids, dtype=np.int64)
print(f"corpus chunk: {len(text)} chars -> {len(t)} tokens")

# --- the exact eval_draft code path ---
text2 = tk.decode(t.tolist(), skip_special_tokens=False)
enc2 = tk.encode(text2)
ids = enc2.ids
ia = ib = 0
agree = []
while ia < len(t) and ib < len(ids):
    if int(t[ia]) == ids[ib]:
        agree.append((ia, ib))
        ia += 1
        ib += 1
    elif ib + 1 < len(ids) and int(t[ia]) == ids[ib + 1]:
        ib += 1
    else:
        ia += 1
print(f"decode/re-encode agreement: {len(agree)}/{len(t)} = {len(agree)/len(t):.3f}")
assert len(agree) >= 0.95 * len(t), "alignment failed on real text"

ccls = bytearray(b"p") * len(text2)
spans = sorted(
    [(m.start(), m.end(), "j") for m in TOOL_JSON.finditer(text2)]
    + [(m.start(), m.end(), "c") for m in FENCE.finditer(text2)]
)
last = 0
for st, en, k in spans:
    if st < last:
        continue
    for ci in range(st, en):
        ccls[ci] = ord(k)
    last = en

tcls = []
for st, en in enc2.offsets:
    mid = (st + en) // 2 if en > st else st
    tcls.append(chr(ccls[mid]) if mid < len(text2) else "p")
from collections import Counter

cnt = Counter(tcls)
share = {k: v / len(tcls) for k, v in cnt.items()}
print("token class shares on real text:", {k: f"{v:.3f}" for k, v in share.items()})
# sanity: non-trivial amounts of each class
assert share.get("p", 0) > 0.5 and share.get("c", 0) > 0.05 and share.get("j", 0) > 0.01
print("ALIGNMENT-CLASSIFY-OK")
