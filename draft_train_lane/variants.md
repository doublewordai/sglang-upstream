# Draft architecture variants — decision-gated notes

Baseline (this lane's main path): fine-tuned NextN chain drafter (EAGLE, topk-1),
drop-in via `--speculative-draft-model-path`. Everything below is a *second
candidate* to evaluate only if the gates fire.

## Gate measurement (do this first when real capture data lands)

From the single fine-tuned drafter, compute **top-1 / top-4 per segment type**
over held-out windows, classifying segments by content (tool-call JSON, code,
prose/thinking). Session-spec `kind` (user/tool) covers only prompt-growth
source, not output segment type — content classification needed. Crude corpus
marker density (per 2 MB): 1994 code fences, 3716 `def`, ~315 tool-JSON blocks.

- Gate A (two-drafter): max per-segment accuracy gap > ~5 points top-1 AND the
  low segment is a large token share (>15%). Measured shares (full corpus,
  7.98M tok): prose 62.6%, code 33.0%, tool-JSON 4.4% — tool-JSON alone fails
  the share threshold; the split to measure is structured {code+JSON} 37.4%
  vs prose 62.6% (segment_shares.json).
- Gate B (chain depth): per-depth top-1 (chain_rollout diagnostic) collapses
  faster than linear with depth (i.e. depth-3 marginal token < half of depth-1).
- Gate C (parallel/block): none measurable pre-hoc; only build if chain results
  are mediocre AND a per-step-draft-latency argument dominates (one forward per
  block vs k forwards per chain).

## Option B: two-drafter (AngelSpec-style segment split)

- Drafter P (prose/thinking) + drafter T (tool-JSON/code); same NextN
  architecture, different fine-tune data weighting (or two full fine-tunes).
- Router: cheap classifier on the last ~64 draft-input tokens (n-gram markers
  are likely sufficient: `{"name"`, fence, `def `). Must be host-side and
  cached per position to add ~0 latency.
- Engine integration: sglang loads one draft worker; needs a small fork change
  to hold a second draft set + select at `forward_draft` time. VRAM is cheap
  (13.84 GB per drafter on 16×96 GB decode nodes). Draft step latency
  unchanged (one drafter runs per step). Verify path untouched.
- Risk: switching drafters mid-chain at segment boundaries — simplest rule is
  per-chain selection (the drafter chosen at chain start runs the whole chain).

## Option C: parallel/block drafter

- One draft forward emits the whole candidate block (heads at offsets 1..k,
  Medusa-style, on the draft trunk or directly on target hidden).
- Kills the chain's compounding error (Gate B failure mode) at the cost of
  per-position accuracy (no g-feedback) and a different verify shape
  (sglang's ngram/lookahead spec machinery or a custom verify batch).
- Larger fork surface in the engine; only worth it if Gate B fires hard AND
  Gate A doesn't (uniform difficulty across segments but depth-fragile).

## Current evidence (synthetic + priors)

- Chain-rollout diagnostic on synthetic data: chain CE 13.3-15.0 vs parallel
  11.9 (unlearnable data — magnitudes meaningless, mechanics proven only).
- No real-data per-segment numbers yet (needs capture).

## Gate B — second drafter paradigm for code segments (AngelSpec-style routing)

Supervisor deep-read (scout-arxiv-sweep, spec-drafter-paradigm-routing-deep /
AngelSpec): no single drafter wins across workloads; complementary drafters
(MTP for high-entropy chat; block-diffusion + predecessor-conditioned AR head
for code/math) routed per request give **1.98-2.40x** on their segments, spec
stays exact.

**Routing signal** (this lane's tooling): `eval_draft.py --per-segment`
top-1/est-accept-len by {prose, code, tool-JSON} on real capture — plus the
per-DEPTH chain metrics per segment (extension planned: segment × depth).

**Gate B (open a second-drafter lane) fires iff, on real data with the
chain-trained MTP draft**: (1) code+JSON (37.4% of tokens) est-accept-len
lag prose by ≥0.5 tokens/step at the prod depth, AND (2) the lag persists
after objective/hparam tuning (the sweep shows depth stability is trainable —
lr2e-4/w256 lifted depths 0→0.31-0.63 without any new architecture), AND
(3) projected blended gain ≥ +0.15 accept tokens (i.e. the code drafter must
close ≥40% of the segment's gap). A predecessor-conditioned AR head on the
same draft trunk is the cheapest second paradigm (reuses capture + training
stack; no block-diffusion infra).

Current evidence: none yet (needs real capture per-segment numbers). The
pilot's uniform dummy target cannot measure segment differentials.

## P-EAGLE / DFlash fit (scout note: parallel-drafting-deep, 2026-09-03)

**DFlash (verifier hidden states projected into the speculator KV)** — the
best fit, and remarkably cheap for us: our DraftNextN is TRAINED on exactly
this distribution — its input at every position is (x_t, TARGET hidden
h_{t-1}) (teacher forcing; the capture provides target final-layer hiddens).
The depth collapse we measured happens only because inference feeds the
draft's OWN g back. The verify forward computes target hiddens for every
verified position — an EAGLE-worker change that feeds those verifier hiddens
into the draft's inputs (instead of / alongside the draft's g) puts the draft
back on its training distribution with ZERO new training and no architecture
change. Cheapest deep-K stability fix; candidate "Gate C-prime": try this
worker path before any new drafter paradigm.

**P-EAGLE (all draft positions from one verifier forward's hiddens)** —
parallel multi-position heads on the verifier's hidden block; needs new
output heads (x_{t+1..t+k} each from a shared trunk over verifier hiddens)
and a custom verify shape. Our capture format (token, target hidden) per
position trains it directly; our MLA trunk could host the heads. Medium fork
surface; only if DFlash-style worker conditioning + chain training
underdeliver on real data.

Both keep spec exact (verify unchanged). Recorded, not built — real-data
per-segment and depth numbers first.

## EAGLE-3.1 depth pathologies (scout note: eagle31-postnorm-draft-deep)

Two fixes to apply when training deeper drafts (depth-5 arm for the GLM-5.3
report):
1. **Attention drift toward own generated tokens at depth** — the chain
   objective trains exactly this regime (self-feedback KV); the DFlash-style
   worker conditioning (verifier hiddens into the draft inputs) removes it
   outright. If neither suffices at depth 5 on real data, the explicit 3.1
   attention constraint (rebalance toward verified-prefix positions) is an
   architecture change — keep as the escalation.
2. **Hidden-magnitude growth from the unnormalized residual (post-norm
   fix)** — our NextN draft already has the norm discipline: the fed-back
   chain hidden passes through `hnorm` (rms), the residual stream is
   rms-normed before `lm_head` (`shared_head_norm`), and KV inputs are
   `eh_proj([enorm(emb); hnorm(g)])` on normalized operands. The unbounded
   residual accumulation the 3.1 post-norm fix targets is a multi-layer
   drafter pathology; our single-layer NextN does not accumulate across
   layers, but depth-5 chain rollout DOES accumulate across steps — the
   depth-5 training arm should verify g-magnitude stability per depth
   (add ||g||-by-depth to eval_draft if the real-data depth-5 arm shows
   drift).

## ATLAS — online-adaptive drafter (scout note: online-adaptive-draft-deep)

Heavyweight static draft (our fine-tuned NextN) + lightweight ADAPTIVE draft
continuously updated from live traffic, with a confidence-aware controller
choosing between them per step. Our capture pipeline is exactly the data
source: the hook streams (token, target hidden) pairs — a small head (or
LoRA delta on the static draft) could refresh on the live stream, and the
controller picks the adaptive drafter only when its recent acceptance
outperforms. Caveats for prod: online updates change the draft mid-flight
(acceptance-based speed is safe — greedy verify keeps outputs exact — but
the controller must be robust to distribution shift; and the draft weight
path must support hot-reload, which sglang does not have today).
**Gate B architecture #2**: fires under the same conditions as the AngelSpec
gate (segment-differential accept on real data), but when the differential is
*drift-shaped* (traffic composition shifts over days) rather than
*segment-shaped* (code vs prose). Adaptive wins where the static fine-tune
stales; routing wins where segments are stable and distinct.
