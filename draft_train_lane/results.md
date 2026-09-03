# Lane draft-train — a better speculative draft for agent traffic

Question (brief): GLM-5.3's NextN draft accepts 2.9/4 on text; per-user speed at b=1 scales with
accept length (1.86× at 2.5). Can a draft trained on OUR traffic (5.8M real tokens) accept more?
Milestones: (1) data plan + capture hook, (2) training code, (3) export to NextN format,
(4) accept-length A/B on the 8-node system.

## Status
- M1 (capture hook + dummy rig): **DONE, PASS** (2026-09-02)
- M2 (training code): in progress
- Branch `lane/draft-train` (worktree `$SCRATCH/src/sglang-draft-train-0902`, from integration-0902 @53abdd1)

## M1 — capture hook + data plan (DONE 2026-09-02)

**Design (why):** capture the target's post-final-norm hidden states (the lm_head input = exactly
the tensor the NextN draft consumes via hnorm / `spec_info.hidden_states`), per request, keyed by
(rid-hash, absolute position), **extend-mode only**. The corpus replay (workload lane) re-feeds
every generated token as the next turn's prompt, so the unique token stream = prompt growth = what
prefill sees; decode-side capture would require eager decode (~20 tok/s aggregate → ~40 h) and is
skipped by design (missed: each session's final-turn output, ~31k of 5.8M tokens, 0.5%).

**Implementation:** `sglang/srt/draft_capture.py` + 3-line hook in
`DeepseekV2ForCausalLM.forward` (deepseek_v2.py, last PP rank, before logits_processor) — commit
`9cb6c67` on lane/draft-train. Env-gated (`SGLANG_DRAFT_CAPTURE_DIR`, `_MODES=extend` default,
`_TAG`). Append-only shards `shard-<tag>-<rank>-<boot>.bin`, records
`[magic|ver|rid_hash|start_pos|n_tok|hdim|fp16] [i32 tokens] [fp16 hidden 6144]`, background
writer thread with bounded queue (drops counted, never blocks the engine), per-shard stats.json.
Safety: only plain EXTEND/DECODE enums; DP-padding-fabricated batches (`_original_forward_mode`)
and stream-capture passes skipped; sum(extend_lens)==rows and rids-present checks.

**Dummy rig tests** (1×GH200 nid010151, dummy-glm53 6-layer, chunk 2048; validator
`draft_capture_reader.py`, rerunnable via `dummy_capture_test.sh`):

| run | config | records/tokens | skips | checks | verdict |
|---|---|---|---|---|---|
| dummy3 | eager, extend+decode | 38 / 5446 | 0 | A: 3 chunks 2048+2048+904 contiguous, tokens exact; B: 308 tok from pos 4992 only (prefix reuse ✓); decode tokens == server output ids (12/12, 4/4, 4/4); hidden finite, 0 zero rows | **PASS** |
| dummy5 | server-default graphs (breakable prefill graphs ON) | 2 / 2 | 6×no_rids | extend forwards lose rids under breakable prefill CUDA graphs | capture broken under prefill graphs — rig must run prefill eager (prod prefill arm already does); boot survives graph capture with the hook, no garbage records |
| dummy6 | decode graphs ON + `--disable-prefill-cuda-graph` (= real-rig shape), extend-only | 8 / 5416 | 0 | prompts exact, no decode records (as designed) | **PASS** |

Note: hidden RMS p50 ≈ 1.04e-4 on the dummy — dummy RMSNorm weights are random-small; finiteness
and mapping are what M1 proves. Semantic validation of hidden VALUES (checkpoint-draft top-1
accuracy on captured hiddens ≈ plausible) comes with real data in M2.

**Data plan (5.8M tok ≈ 71 GB fp16@6144):** 4 shard files (prefill arm last-PP-stage ranks, one
per DP rank), one boot, few large files. Storage: `$SCRATCH/grace-1m/lanes/draft-train/capture/`.
Dedupe: radix-cache recompute can re-prefill evicted text → duplicate (rid, position-range) pairs
across different rids; the training-data prep will dedupe by token-window hash (keep first).

**Exact 8-node capture command** (when the supervisor assigns the system): the launch script is
`$SCRATCH/grace-1m/lanes/draft-train/l3-launch-capture.sh` — byte-diff of prod `l3-launch-v17.sh`
limited to: SGLANG_TREE → sglang-draft-train-0902, SGLANG_DRAFT_CAPTURE_* env (extend-only),
lane port range 57000–57999, `--enable-metrics` dropped, NONDISAGG fallback branch fixed.

```bash
# boot (on the holder, 8 nodes):
HOLDER=<jobid> bash $SCRATCH/grace-1m/lanes/draft-train/l3-launch-capture.sh
# replay (from any node with the venv, inside an srun step):
python3 $SCRATCH/pd-serve/bench/replay_client_ss_v2.py \
  $SCRATCH/grace-1m/lanes/workload/out/sessions_pi_measured.jsonl \
  --base-url http://<decode-master>:57200 --model glm-5.3-fp8 \
  --concurrency 16 --gap-scale 1.0 --timeout 1800 \
  --tokenizer /projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516/tokenizer.json \
  --corpus $SCRATCH/grace-1m/lanes/workload/out/corpus_pi.txt \
  --out $SCRATCH/grace-1m/lanes/draft-train/capture/replay_requests.jsonl
# verify (rerunnable):
python3 $SCRATCH/grace-1m/lanes/draft-train/draft_capture_reader.py stats <capture-dir>
```

Port assignments (LANES.txt line 17 → 57000–57999): 57000 prefill arm, 57100 decode arm, 57200 LB
router, 57300 bootstrap, 57400/57500/57600 dist-init.

**Unexplained (benign):** the dummy-rig srun step's bash gets SIGKILLed (exit 137) during server
teardown — the server's shutdown path self-SIGKILLs via kill_process_tree; the step bash dies with
it. All capture data is flushed and validated BEFORE teardown in the test script; sleep-only steps
survive 300 s, so it is teardown-specific, not a step reaper. No data loss possible (writer flushes
on ≤1 s idle).

## M2 — training code (synthetic proof DONE 2026-09-02)

**Architecture decision (fork):** fine-tune the checkpoint's NextN layer-78 in its own
architecture (drop-in export), NOT a new dense EAGLE-3 head. The **routed experts (9.7B params)
are FROZEN** (replicated bf16 buffers, gradients still flow through them): 5.8M training tokens
give 0.6 tok/expert-param (overfit territory), the frozen experts are what the engine runs, and
frozen experts make 3-GPU FSDP trivially fit. Trainable surface: eh_proj, enorm/hnorm, all
attention projections, router gate, shared expert, norms (0.26B params). Full-expert training
stays possible (promote buffers to params + give the MoE its own FSDP unit) — documented in the
code. The supervisor's EAGLE-3.1 note maps onto this design as-is: hnorm (norm on every incoming
target hidden) and shared_head.norm (post-norm feedback into the chain) are NextN's built-in
equivalents, preserved. Depth-4-6 acceptance measurement added to the M4 plan. Parallel-head /
two-head (AngelSpec) variants: recorded; head is swappable in the loss module.

**Training form (verified against the engine):** teacher-forced parallel windows — input at
position t = eh_proj([enorm(Emb(x_t)); hnorm(h^target_{t-1})]), dense per-window causal
attention (== the engine's top-2048 DSA selection for window positions < 2048, indexer frozen),
loss = CE(lm_head·shared_norm(g_t), x_{t+1}) + 1.0·MSE(g_t, h^target_t) (EAGLE feature loss).
MoE routing per sglang's biased_grouped_topk + remap_topk_for_per_rank_shared_slots: top-8 by
sigmoid+bias, renormalized weights, ×2.5 post-MoE, shared expert at net weight 1. Rope verified
line-for-line against sglang's GPT-J interleave (utils.py apply_rotary_emb, is_neox_style=False),
scale 256^-0.5 (deepseek_v2.py:1770).

**Synthetic proof (3× GH200, FSDP):** 400 steps, ~55k tok/s, val feature-MSE 0.283→0.271
(decreasing), CE finite (near ln(154880) — the synthetic Markov task is deliberately
~unlearnable; mechanics are the point), val top-1/top-4 plumbing works, FSDP full-state-dict
checkpoint save works. Rerun: `torchrun --nproc-per-node=3 --master-port 57100 train_draft.py
--data synthetic` on the holder node.

## M3 — export to NextN checkpoint format (path DONE 2026-09-02)

`export_draft.py`: training state dict → checkpoint-layout safetensors (`model.layers.78.*`,
bf16 keys as-is, fp8-block requant where the original was fp8, indexer pass-through
byte-identical from the source checkpoint, embed/lm_head included, config/tokenizer copied).
`verify_roundtrip.py`: 17 bf16/passthrough keys bit-exact, fp8 keys maxrel 2.1e-7 (extract→export
roundtrip of untrained weights). Load test on the 1-GPU dummy78 rig with
`--speculative-draft-model-path <export>`: server READY, EAGLE 3-step/4-draft chain runs with
the exported 256-expert draft (accept 1.00 = expected mismatch dummy-target/real-draft).
Remaining for M3: repeat the load test with TRAINED weights (same path, zero new machinery) and
validate acceptance semantics on real weights (needs the 8-node system or captured data).

## State of the project (night of 2026-09-02 UTC)

- **Works:** capture hook (M1, dummy-rig validated both eager and real-rig shapes); exact 8-node
  capture launch script + replay command written; weight extraction + training loop (3-GPU FSDP,
  synthetic proof) + export roundtrip + draft-dir load test on dummy78.
- **Does not work / not yet done:** real-data training (blocked on the 8-node capture);
  semantic validation of captured hiddens (checkpoint-draft top-1 accuracy on real windows —
  the decisive test that capture+model are right); depth-4-6 A/B; the two-head/parallel-head
  variants (recorded, not built).
- **Next step:** when the supervisor assigns the 8-node system: boot `l3-launch-capture.sh`,
  run the corpus replay (command in M1 section), validate with `draft_capture_reader.py stats`,
  then train on real data (sessions held out) and export+load-test the trained draft.
- **Blockers:** none on my side; waiting for the 8-node system assignment (QUEUE.md).
- Holder 6256423 ends 2026-09-03T15:53 UTC (~17 h from now) — no risk to tonight's work.

## State of project — day 2 open (2026-09-03 00:45 UTC)

**M1 capture hook: DONE** (validated on dummy rig; 8-node launch script ready).
**M2 training: synthetic proof DONE** (3-GPU FSDP, loss decreases, checkpoint
consolidation works). **EAGLE-3.1 chain-rollout variant implemented** (mixed
objective `--chain-weight`, per-depth top-1 diagnostic; synthetic run proves
mechanics — on unlearnable synthetic data the chain loss only injects noise,
as expected; real-data evaluation with w=0.1–0.25 pending capture).
**M3 export+load: DONE, closed with trained weights** (real FSDP checkpoint →
13.84 GB export → engine loads it → EAGLE chain runs).
**M4 A/B: fully staged** — `run_m4_ab.sh` (old/new × depth 3–6, corpus replay,
accept-len steady-window parsing; banners/pids/flags verified against the boot
script), `m4_summary.py` comparison table.
**Real-data pipeline: fully staged** — `run_realdata_pipeline.sh` sequences
boot-capture → full replay → capture stats → pre-training baseline eval
(`eval_draft.py`, the decisive semantic check) → training (srun GPU step) →
export (srun GPU step) → M4 A/B. Single command: `HOLDER=<8-node job> nohup
bash run_realdata_pipeline.sh > logs/pipeline.out 2>&1 &`.
**Variants: gated** (`variants.md` — two-drafter and block-drafter designs
with decision gates; segment labeling needs real capture).

Blocked on: 8-node system assignment (QUEUE.md). Holder F ends 2026-09-03
15:53 UTC; H windows per GPU-CLAIMS.md if GPUs are needed before then.

## PILOT — full pipeline on existing dummy-rig capture (2026-09-03 02:00 UTC)

Supervisor-directed dry-run so only the large capture remains when the 8-node
system lands. Data: `capture/dummy-20260902-195434` (1 session, 5012 tok,
18 windows — dummy78 target, semantically meaningless, format-exact).

| run | CE | top-1 | feat-MSE | chain top-1 by depth 0..5 |
|---|---|---|---|---|
| baseline (original weights) | 15.865 | 0.000 | 1.573 | — |
| 300 steps parallel | 0.037 | 0.997 | 0.132 | 1.0, 0.00, 0.01, 0.00, 0.01, 0.00 |
| 300 steps + chain w=0.5 | 0.284 | 0.996 | 1.219 | 1.0, 0.09, 0.19, 0.22, 0.25, 0.19 |

- **Learning proven**: 0 → 0.997 top-1 on a consistent target (the loop
  RealData→model→loss→FSDP is correct on capture-format data).
- **EAGLE-3.1 mechanism validated**: teacher-forced-only drafts collapse at
  depth 1+ (even here); the chain objective repairs it (depths 1-5: ~0 →
  0.09-0.25) at ~zero parallel-accuracy cost. Real-data M4 gets a third arm:
  pick pure vs chain-trained OFFLINE via eval_draft chain-depth metrics
  (no extra boots), then A/B the winner vs old at depths 3-6.
- **Export hardened**: `--orig-ckpt` fp8 passthrough — 1538/1538 frozen
  expert+kv_b tensors byte-identical to the checkpoint (without it:
  0.4-1.6% requant drift); trained keys requant ≤0.9% (fp8 grid); engine
  load test SERVER-READY with the chain-trained export.
- Chain-loss bugs found+fixed by the pilot: missing window context in the
  chain prefix, off-by-one seed/feature indexing, chains-counter
  mis-normalization (depth metrics read 6× low).

When the system lands: `HOLDER=<job> nohup bash run_capture.sh` (or the full
`run_realdata_pipeline.sh`), then eval_draft baseline (semantic check),
train (consider --chain-weight 0.5 arm), export with --orig-ckpt, M4 A/B.

## Gate: draft-swap decision rule (v17) — written 2026-09-03, numbers to date

**Proposed swap**: chain-trained NextN draft (pilot: depth-1..5 top-1 0.09-0.25
vs ~0 for the current teacher-forced-only draft, at unchanged parallel top-1)
loaded via `--speculative-draft-model-path`, fp8 passthrough export.

**Primary metric**: accept length A at fixed depth D (steady ≥10-min window of
the corpus replay, concurrency 16, identical rig). Baseline A_old = 2.9 (old
draft, prod depth). Swap iff:

1. **ΔA ≥ +0.15 tokens/step at prod depth** (2.9 → ≥3.05 ≈ +5% decode
   tokens/step). Below +0.10 = no swap (measurement noise band from
   mtp-accept-style runs is ~±0.03-0.05).
2. **No regression at any depth 3-6** (a draft that wins only at depth 3 but
   loses at 6 is not a swap; it would forclose the depth increase below).
3. **Depth-scaling bonus (not required)**: if A(new, D=5 or 6) ≥ A(new, D=3),
   the swap should come with the depth bump — throughput at depth D scales
   ~A/(1+c·D); c comes from the M4 arms' measured step times.
4. **Quality**: EAGLE greedy verify emits exactly the target's greedy stream
   regardless of draft — no output-quality gate needed for greedy traffic
   (pi corpus replay is greedy). If sampled traffic matters, run the
   rejection-sampling caveat check first.

**Two-drafter (Gate A) rule, with measured shares** (prose 62.6% / code 33.0%
/ tool-JSON 4.4%): a second drafter pays only if
`share_structured × ΔE_structured ≥ +0.15` → ΔE_structured ≥ +0.40 accept
tokens on 37.4% of traffic — a very high bar. Current evidence (chain
training lifts ALL segments' depth stability with one draft) argues the
single chain-trained draft captures most of the available gain. Gate A stays
closed unless real-data per-segment top-1 shows a ≥5-pt gap that chain
training does NOT close.

**Measurement protocol** (all numbers from identical rigs): old vs new draft ×
depth 3/4/5/6, `run_m4_ab.sh`, `m4_summary.py` reports mean/p50 and new-old
deltas per depth.

## Objective-arms comparison (pilot, lr2e-4/w256/s400, 2026-09-03 05:20 UTC)

| objective | parallel top-1 | chain top-1 by depth 0-5 | est accept-len (d4) |
|---|---|---|---|
| pure parallel | 0.996 | 1.0, .31, .63, .63, .50, .50 | **1.63** |
| chain w=0.5 | 0.990 | 1.0, .03, 0, 0, 0, .03 | 1.03 |
| chain + residual 0.5 | 0.989 | 1.0, .13, 0, 0, 0, 0 | 1.13 |
| chain + residual 1.0 | 0.966 | 1.0, .03, .03, 0, .03, .03 | 1.03 |

Depth stability is achievable by either recipe but they don't stack on the
dummy pilot; the real-data run arbitrates all three arms (each is one
command). Scout-note architectures recorded in variants.md: DFlash-style
verifier-hidden conditioning (zero retraining — maps exactly onto our
teacher-forced training distribution), P-EAGLE, AngelSpec routing (Gate B),
ATLAS adaptive drafter (Gate B #2), EAGLE-3.1 depth pathologies (our norm
discipline already covers the post-norm fix).
