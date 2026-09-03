# Lane draft-train worklog

## Day 1 — 2026-09-02 (timestamps UTC via `date -u`)

- 18:50 UTC Read COMMON/IMPL/PROJECT + brief. Mission: better speculative draft for agent traffic.
  Premises verified against sources:
  - accept 2.9 / speedup model: ../mtp-debug/results.md, ../verify-cost/results.md ✓
  - corpus 5.8M tok: ../workload/results.md (corpus_pi.txt, sessions_pi_measured.jsonl) ✓
  - holder 6256423 nodes nid[010147-010152,010155,010157]; 5th = **nid010151** ✓
  - GPU state on nid010151: GPU0 has 17.8 GiB resident (metrics lane rig), GPU1-3 free.
    Slurm `--gres=gpu:3` hands out physical 0,1,2 (collides with metrics). Pattern for all
    my GPU steps: `--gres=gpu:4` + `CUDA_VISIBLE_DEVICES=1,2,3` inside (device-cgroup
    isolation off, CVD override works; torch sees 3 devices — verified).
  - `~/slime-venv` does NOT exist (brief mentioned it as an option) → plain PyTorch in the
    sglang venv (env-U.sh). No new packages needed.
- 18:55 UTC Trees: branched worktree `$SCRATCH/src/sglang-draft-train-0902` as `lane/draft-train`
  from **integration-0902** (53abdd1) — the brief says "your worktree from integration-0902";
  IMPL.md's `pd/mtp-hisparse` is an ancestor of it (verified with merge-base), so this is the
  superset choice. +8099 files on scratch (inode budget noted).
- 19:00 UTC Architecture study (all in sglang-integ-0902 = my worktree):
  - Target `GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM)` in models/glm4_moe.py; forward in
    deepseek_v2.py:3178 — `hidden_states = self.model(...)` (post `self.norm`, i.e. the
    lm_head input = exactly what the EAGLE worker stores as spec_info.hidden_states) →
    logits_processor. `DeepseekV3ForCausalLM` is `pass` — one hook point covers the target.
  - Draft `GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN)`; DeepseekModelNextN.forward:
    `eh_input = fused_eh_norm(emb(input_ids) [enorm], spec_info.hidden_states [hnorm])`;
    `hidden = eh_proj(eh_input)` → 1 DSA decoder layer → shared_head.norm → logits.
  - Draft chain at decode (eagle_worker_v2.draft_forward): step i input token = previous
    argmax proposal, spec_info.hidden_states = target hidden (step 0) then draft's own
    hidden. I.e. **draft pos t: [enorm(Emb(x_t)); hnorm(g_{t-1})] → logits for x_{t+1}**,
    g_0 = target hidden. Training must mirror this (EAGLE scheme; feature loss
    MSE(g_t, h_t^target)).
  - `set_eagle3_layers_to_capture` already exists (aux hidden capture for EAGLE-3) — not
    needed for the NextN fine-tune path.
  - ForwardBatch always carries `rids` (init_new line 801). `--speculative-draft-model-path`
    exists (defaults to model_path) → milestone-3 export dir is supported.
  - Real checkpoint NextN keys: `model.layers.78.*` incl. `enorm`,`hnorm`,`eh_proj`,
    `shared_head.norm` (from quant-config modules_to_not_convert); runtime remap in
    GlmMoeDsaForCausalLMNextN._map_mtp_ckpt_name (`model.layers.78.<spec>` → `model.<spec>`,
    rest → `model.decoder.<rest>`).
  - Dummy rig: `$SCRATCH/grace-1m/dummy-glm53` (6-layer GLM-5.3 arch, 16 experts, hidden 6144,
    NextN=layer 6) + `--load-format dummy`; boot pattern from lanes/mtp/l2-boot-mtp.sh.
- 19:20 UTC Capture design decision (the fork): **prefill/extend-side capture only**, decode off.
  Reasoning: the corpus replay (workload lane) builds prompts from corpus text — every output
  token re-enters the next turn's prompt, so the unique token stream = prompt growth = what
  prefill sees; decode outputs are discarded by the client. Decode-side capture would need
  CUDA graphs off (python forward doesn't run under graph replay) → eager decode ~20 tok/s
  aggregate → ~40 h for 2.9M decode tokens. Prefill capture works WITH graphs on the decode
  arm (decode forwards simply don't hit python) and the prefill arm runs eager extends today.
  Missed tokens: each session's final-turn output (~36×867 ≈ 31k of 5.8M, 0.5%).
- 19:27 UTC Supervisor port rule (2026-09-02): lanes must use ports 40000+1000×line-in-LANES.txt..+999.
  draft-train = line 17 → **57000–57999**. Dummy rig server: 57000. Check `ss -ltn` before binds.
- 19:50 UTC Supervisor note (scout idea "parallel-drafting"): consider a parallel head
  (whole block emitted in one forward, rejection sampling keeps target distribution) as a
  second candidate architecture next to the NextN fine-tune. Recorded as a design fork:
  primary = NextN fine-tune (drop-in export, engine-unchanged); the training code will keep
  the head swappable (chain rollout vs parallel block emission) so a parallel-head variant
  can be trained on the same captured data later. Parallel head needs draft-worker changes
  (1 forward, 4-token head) — engine work beyond this lane's scope unless the supervisor
  asks; verify path is unchanged (topk-1 linear tree mask either way).
- 19:55-20:25 UTC M1 executed on the dummy rig (nid010151 GPU1, port 57000):
  * dummy3 (eager, extend+decode): VALIDATE-OK. 38 rec/5446 tok/0 skips. A=5000 (3 chunks
    2048+2048+904, contiguous) + 12 decode records whose tokens == server output ids exactly;
    B=308 tok from pos 4992 (8-tok page re-prefill + 300 growth — prefix reuse captured once);
    C=100+4. Hidden finite, 0 zero rows, RMS p50 1.04e-4 (dummy norm weights are random-small).
  * Client bug fixed on the way: meta_info.output_token_logprobs entries are
    [logprob, token_id, text] — read x[1], not x[0] (my first read produced int(logprob)=-11).
  * dummy5 (server-default graphs): this build captures PREFILL graphs by default
    (backend=breakable, buckets 4..2048) — extend forwards carry NO rids there (6 no_rids
    skips) => capture broken under prefill graphs. Not a problem for the real rig: the prod
    prefill arm runs --disable-cuda-graph (l3-launch-v17.sh line 164). Boot itself survived
    graph capture with the hook present (no crash, no garbage records).
  * dummy6 (decode graphs ON + --disable-prefill-cuda-graph + MODES=extend, the real-rig
    shape): VALIDATE-OK. 8 rec/5416 tok/0 skips, prompts exact, no decode records (by design).
  * Teardown 137: the step bash is SIGKILLed when the server self-destructs at shutdown
    (kill_process_tree SIGKILLs itself; something in that path takes the step bash too —
    setsid did not help). Sleep-only steps survive 300s => teardown-specific, no reaper.
    Mitigation: validate BEFORE teardown (already in the script). Data loss impossible
    (writer flushes every <=1s idle / 500 records).
- 20:30 UTC l3-launch-capture.sh written (lane dir): byte-diff vs prod v17 = tree, capture env,
  ports 57000-57999, no --enable-metrics, NONDISAGG branch fixed. bash -n OK. The exact 8-node
  command + replay command recorded in results.md.
- 20:32 UTC M1 results written to results.jsonl/results.md. Committing.
- 20:35 UTC Prod chain outage 20:03-20:16Z (decode rank died, HiSparse host KV pool exhausted,
  fix lane host-pool-backpressure) noted — unrelated to this lane (my traffic goes to my own
  dummy rig on 127.0.0.1:57000; nothing in this lane touches prod).
- 20:50 UTC Supervisor note (scout, lossless): EAGLE 3.1 (FC norm after each target hidden +
  post-norm hidden-state feedback) fixes deep-draft attention drift; up to 2x acceptance at long
  context. Our NextN 2.9-3.25/4 vs Zhipu ~4.5/5. Mapping to this lane:
  (a) The NextN architecture ALREADY has the 3.1 ingredients in its own form: hnorm (RMSNorm) on
      every incoming target hidden before the eh_proj fusion, and shared_head.norm post-norm on
      the hidden that is fed back into the chain (spec_info.hidden_states is the post-norm
      draft output). The fine-tune preserves both.
  (b) Depth is an M4 MEASUREMENT AXIS: the A/B on the 8-node system will measure accept length
      at --speculative-num-steps 3 (prod), 4, 5, 6 for old vs new draft (topk 1, draft-tokens
      steps+1) on the corpus replay incl. long contexts.
  (c) Training variant kept possible: "chain rollout" fine-tune (feed the draft's OWN g back
      instead of teacher-forced target hidden for positions t>0 of the window, sequential
      forward) — directly targets deep-chain stability. Will implement as a flag after the
      base loop is proven, time permitting.
- 21:00 UTC Supervisor rule (COMMON.md updated): wrap EVERY ssh/scp in local `timeout 120`
  (300 for big scp); remote work > 2 min stays detached, poll logs with short calls; never
  `find /`. Adopted from here on.
- 21:35 UTC Supervisor note (scouts): AngelSpec (+10-12% from two complementary drafters routed
  per step), AgentSpec structure-isolated drafting, EAGLE 3.1. Plan mapping: primary remains ONE
  NextN fine-tune on the full corpus. Cheap variant to quantify segment headroom WITHOUT engine
  work: when real data arrives, report top-1/top-4 accuracy split by segment type (thinking vs
  tool-call JSON vs prose — the corpus text carries the structure markers in-band); if a
  segment-specialized head shows a large gap, train a split-head variant and propose the
  per-step routing as engine work. Two-head routing itself needs draft-worker changes (out of
  this lane's scope unless asked).
- 22:00-22:45 UTC M2+M3 executed:
  * M2 synthetic proof: 7 debug iterations to a clean run (ssh-CVD/GPU mapping, env-U needed
    for CUDA forward-compat, eh_proj key name, numpy randrange->randint, rope broadcast shapes
    [k_pe has no head dim], fp16-numpy std OVERFLOW in SyntheticData.noise (-> inf hiddens ->
    NaN loss; fixed with fp32 slice std), nested-FSDP param access (experts as frozen bf16
    buffers), m.moe.eg path, FSDP state_dict (stuck 7 min on raw_model.state_dict under FSDP;
    proper FullStateDictConfig collective). Final: 400 steps, 54.8k tok/s, val feature-MSE
    0.283->0.271, checkpoint save OK. Note: ssh sometimes hangs after a detached launch even
    with nohup+</dev/null — the launch itself always lands; poll logs instead of holding ssh.
  * M3: extract->roundtrip.pt->export->verify: 17 bf16 keys bit-exact, fp8 maxrel 2.1e-7.
    Export bug fixed on the way: scale keys are X.weight_scale_inv (my first suffix-append
    silently skipped dequant; now asserted by dtype check).
  * M3 load test: dummy78 + --speculative-draft-model-path export_roundtrip, EAGLE 3/4:
    boots, generates, spec chain runs (accept 1.00 as expected for dummy target).
  * Rope equivalence verified by line-for-line comparison with sglang
    rotary_embedding/utils.py apply_rotary_emb (GPT-J interleave identical); attention scale
    256^-0.5 confirmed at deepseek_v2.py:1770; MoE routing semantics confirmed at
    topk.py:1291-1360 + remap_topk_for_per_rank_shared_slots (shared slot weight 1/2.5 with
    post-MoE 2.5x => net shared weight 1.0).
  * Scripts vendored into the worktree (draft_train_lane/) + committed 707384feb (durability
    against scratch purge).
- 22:46 UTC Cleanup per supervisor inode rule: no __pycache__ in my tree (0 found); removed
  4 superseded dummy-capture dirs + runs/smoke; kept PASS evidence dirs. Quota now
  812k/1024k files. At DONE I will `git worktree remove` after committing (noted).
- 22:47 UTC Day 1 close: M1 DONE+validated, M2 synthetic DONE, M3 path DONE (trained-weights
  rerun pending), M4 planned (incl. depth 4-6 A/B per supervisor note). results.md has the
  nightly state-of-project section. Holder ends 2026-09-03T15:53 UTC.
- 22:50 UTC Day-1 close-out: branch pushed attempt failed (no GitHub creds on the persistent
  node) -> durability via git bundle pulled to gotenks (draft-train.bundle, 53abdd1..707384f)
  + all scripts/results/worklog live in BOTH the cluster lane dir and the local lane dir.
  No lingering srun steps on the holder (checked). Holder ends 2026-09-03T15:53 UTC.
  Remaining verifications that need the 8-node system or real capture: (a) PP last-stage
  rids/extend-lens availability for the hook (stats.json will show skip reasons if absent),
  (b) checkpoint-draft top-1 accuracy on real windows (decisive semantic check of capture +
  training model), (c) trained-draft load + accept A/B at steps 3/4/5/6.
- 23:20 UTC EAGLE-3.1 chain-rollout variant implemented + mechanically proven (chain_loss in
  draft_model.py; --chain-weight/--chain-len/--chains-per-window/--chain-detach in train_draft).
  Synthetic 60-step run: mechanics PASS, 23k tok/s. Findings: chain CE/MSE much higher than
  parallel (13.3/1.8 vs 11.9/0.28) = the depth-drift signal; on unlearnable synthetic data the
  chain loss (w=0.5) degrades the parallel objective - on real data evaluate with w 0.1-0.25
  and the top1_by_depth diagnostic. Two-drafter segment labeling: session spec has only
  user/tool turn kinds; output segments need content classification (markers measured;
  deferred to real capture). Logits-in-forward fix for FSDP along the way.
- 23:58 UTC M3 trained-weights rerun PASS: export from the real syn3gpu FSDP checkpoint
  (13.84 GB, 1571 tensors) loads on the dummy78 rig with EAGLE spec on and runs the draft
  chain. Full train->export->load->spec-decode loop now proven end-to-end with trained
  weights. Holder F outage 23:32-23:52Z (prod took F; nothing of mine was running;
  synchain had completed; my only casualty was an export launch with a cd-after-redirect
  bug). H claim (nid010468 gpu0) released when F returned. m4_summary.py comparison script
  + M4 driver parse-assumptions verified against the boot script banners (system-ready,
  pid=, log= lines, SPEC_FLAGS passthrough on both arms, served name glm-5.3-fp8).
- 00:45 UTC Day-2: pipeline + eval + A/B all staged and committed (through 46bd22074);
  results.md state-of-project updated. eval_draft smoke-passed after three interface
  fixes (build_model reuse, RealData.get interface, bf16 autocast). Now waiting on an
  8-node window; holder check: F ends 15:53 UTC today.
- 01:20 UTC Day-2 burst complete: segment classifier + shares (prose 62.6 / code 33.0 /
  tool-JSON 4.4 over 7.98M corpus tokens; Gate A split = structured 37.4 vs prose 62.6),
  per-segment eval tooling (alignment agreement 1.000 on real text), expected capture
  volume 5.88M tok / ~5.7k windows / ~478 steps-per-epoch (STEPS=400 default = 0.84 ep).
  All committed through c80d7a005; bundles pulled to gotenks. Waiting on 8-node window.
- 02:00 UTC Day-2 PILOT COMPLETE (supervisor-directed, on the dummy-rig capture):
  baseline eval CE 15.865/top-1 0.0 -> pure-parallel 300 steps -> top-1 0.9974 (learning
  proven on capture-format data); chain eval exposed TWO chain_loss bugs (missing window
  context in the chain prefix; off-by-one seed/feature indexing; plus a chains-counter
  mis-normalization that made depth metrics read 6x low) - all fixed; chain-trained retrain
  (w=0.5) then validated the 3.1 mechanism: depths 1-5 top-1 ~0 -> 0.09-0.25 at unchanged
  parallel accuracy. Export hardened: --orig-ckpt fp8 passthrough (1538/1538 bit-identical
  frozen tensors; found+fixed a kv_b-scale passthrough miss), verify_pilot metrics fixed
  (maxabs/maxref for fp8 keys). Engine load test PASS with export_pilot2. Endpoint rule
  applied: all heavy commands now via srun steps on nid010151 (a verify_pilot on the
  persistent node likely contributed to an endpoint OOM - killed + moved).
- 04:50 UTC Day-2 (cont.): supervisor-directed burst complete:
  (1) 8-node capture launcher DRY-RUN PASS on holder E/nid010159 (SMOKE mode:
  1-GPU dummy78 through the full run_capture.sh flow - boot/banner/LB-proxy/
  replay/teardown/stats + RealData loads the fresh capture: 36 rids/20 sessions/
  1498 windows). Seven issues found+fixed along the way (SMOKE-block placement,
  wait_for_health order, LB env, CAP-as-argv after env mangling over the flapping
  endpoint, single-instance flock, one CXI service-release race retried per the
  addendum, replay drain hang on the no-EOS dummy). Real launch = HOLDER=<job> only.
  (2) 27-config sweep done: parallel top-1 converges everywhere; depth stability is
  config-dependent WITHOUT the chain objective (best lr2e-4/w256/s400: depths
  [1.0,.31,.63,.63] estA4 1.63); chain objective at that config HURTS (1.03) though
  it helped at w512/lr2e-5 - objective choice is data/config-sensitive, real-data
  arbitration planned. (3) Gate rules written: v17 draft-swap gate (+0.15 accept at
  prod depth, no depth regression), Gate A (two-drafter: needs structured-segment
  delta +0.40), Gate B (AngelSpec paradigm routing - recorded in variants.md with
  the routing signal = our per-segment tooling). (4) Scout notes recorded: P-EAGLE/
  DFlash fit analysis - DFlash (verifier hiddens into draft KV) maps EXACTLY onto
  our teacher-forced training distribution: an EAGLE-worker change with zero
  retraining; cheapest deep-K fix, try before any second drafter. Residual-objective
  arm (logit-delta matching, zero new params) implemented, running.
- 05:25 UTC Day-2 burst complete. Objective arms done: at lr2e-4/w256/s400 pure-parallel
  estA4 1.63 beats chain (1.03) and chain+residual (1.13 at w=0.5, 1.03 at w=1.0 with
  parallel top-1 degraded); recipes don't stack on the dummy pilot; real-data run
  arbitrates all three arms (one command each). Scout notes recorded in variants.md:
  DFlash-style verifier-hidden conditioning = ZERO-retraining worker change matching our
  teacher-forced training distribution (cheapest deep-K fix, try before any new drafter);
  P-EAGLE (medium fork); AngelSpec routing + ATLAS adaptive = Gate B architectures;
  EAGLE-3.1 depth pathologies (our hnorm/shared_head_norm discipline covers post-norm;
  attention-drift escalation path noted). All committed 6cbf291ed4; bundles on gotenks.
- 06:00 UTC Day-2: recipe milestone complete on the smoke capture (199k tok, real val):
  val top-1 0.165 (dummy-target ceiling - recalibrates pilot absolutes), first per-segment
  numbers (p .43 / c .22 / j .52), export+verify+load PASS after the verify caught a REAL
  bug (trainable kv_b silently discarded by unconditional fp8 passthrough - now conditional
  on drift; experts stay bit-identical 1536/1536). GO/NO-GO row written: GO - 1-2 days of
  prod capture (57M tok/day at 4k req/h) or a few replay passes cover the 10-100M-token
  fine-tune regime; <6 GPU-h per arm; recipe adjustment for real data = LOW lr (1e-5/5e-5
  arms + cosine decay + early stop) since the head starts at ~0.8 top-1, not from scratch.
  VAT (verification-aware training) implemented + arm in flight at the chain-helpful config.
