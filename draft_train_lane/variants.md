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
