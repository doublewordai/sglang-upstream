#!/usr/bin/env bash
# draft-train CAPTURE rig: l3-launch-v17.sh + (a) SGLANG_TREE=sglang-draft-train-0902 (capture
# hook), (b) SGLANG_DRAFT_CAPTURE_* env (extend-mode hidden-state capture on the prefill arm,
# which runs eager extends), (c) lane port range 57000-57999, (d) no --enable-metrics,
# (e) NONDISAGG branch flags fixed (prefix_affinity + deepep normal) as a 4-node fallback.
# Consolidated single launch script; see the parameters below (MODEL, SERVED_NAME,
# OVERRIDE_ARGS, DECODE_A2A, NONDISAGG, EAGER, HOLDER).
#
# Topology (proven on GLM-5.2-FP8, lane U — 534 tok/s/GPU decode, 0 failures):
#   Nodes 0-3: prefill — PP4×EP4, DP-attn, DeepEP normal, chunk 8192
#   Nodes 4-6: decode  — EP12, DP-attn, DeepEP auto, FULL CUDA graphs
#   LB router — on decode master, routes prefill→decode
#
# KV handoff via NIXL(UCCL)/CXI. All engines in ONE srun step for Slingshot
# VNI sharing (lane U lesson: separate steps → VNI_NOT_FOUND).
#
# Usage:
#   sbatch pd-boot-glm53.sh
# Or on a holder:
#   HOLDER=<jobid> bash pd-boot-glm53.sh
#
# Environment:
#   NONDISAGG=1  — 4-node unified server, no PD split (for testing)
#   EAGER=1      — disable CUDA graphs on decode
#
#SBATCH --job-name=grace1m-pd
#SBATCH --partition=workq
#SBATCH --account=brics.s6p
#SBATCH --nodes=8
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/s6p/fergus.s6p/logs/slurm/%x-%j.out
#SBATCH --error=/scratch/s6p/fergus.s6p/logs/slurm/%x-%j.err

set -uo pipefail

MODEL="${MODEL:-/projects/s6p/hf/hub/models--zai-org--GLM-5.3/snapshots/e0b07fd2751b42d5efa199cc02c2b271deadc516}"
SERVED_NAME="${SERVED_NAME:-glm-5.3-fp8}"
# Model-specific config override (GLM-5.3 needs qk_rope_head_dim=64; pass OVERRIDE_ARGS= to disable)
OVERRIDE_ARGS="${OVERRIDE_ARGS-{\"qk_rope_head_dim\": 64}}"
# Decode-arm MoE all-to-all backend: megakernel (default) or deepep
DECODE_A2A="${DECODE_A2A:-megakernel}"
# grace-1m: hisparse decode (host-resident latent KV) + radix retention + big context
CTX="${CTX:-1048576}"
HISPARSE_RATIO="${HISPARSE_RATIO:-10}"
HISPARSE_BUF="${HISPARSE_BUF:-4096}"
# v12: v11 tree + whole-page host pool release (transfer destinations index host rows by page)
# v14: + SPEC_FLAGS (MTP / EAGLE draft from the checkpoint's NextN layer) on both arms
# v15: + PREFILL_FLAGS on the prefill arm only (hierarchical cache: host tier for its radix cache)
# v16: + DECODE_FLAGS on the decode arm only (e.g. --dsa-topk-backend torch)
# v17: + DECODE_NNODES (default 4): decode arm on 4*DECODE_NNODES ranks (TP=DP=EP=world), holder needs 4+DECODE_NNODES nodes
DECODE_NNODES="${DECODE_NNODES:-4}"
TOTAL_NODES=$((4 + DECODE_NNODES))
DECODE_WORLD=$((4 * DECODE_NNODES))
SGLANG_TREE="${SGLANG_TREE:-$SCRATCH/src/sglang-draft-train-0902}"
SGLANG_MAP_HOST_POOL_PRIVATE="${SGLANG_MAP_HOST_POOL_PRIVATE:-1}"
U_ROOT=$SCRATCH/runs/glm-isambard/U-uccl-send-abort
# draft-train capture output (extend-only; ~12.3 KB/token, fp16 hidden 6144)
export SGLANG_DRAFT_CAPTURE_DIR="${SGLANG_DRAFT_CAPTURE_DIR:-$SCRATCH/grace-1m/lanes/draft-train/capture/real-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$SGLANG_DRAFT_CAPTURE_DIR"
# Runtime deps promoted under pd-serve (env-F-U.sh / env-nixl-U.sh honor these overrides):
export UCCL_SRC=$SCRATCH/pd-serve/runtime/uccl-coalesce2
export UCCL_LIB_DIR=$SCRATCH/pd-serve/runtime/uccl-fix-20260824/uccl-prefix/lib
export NIXL_PLUGIN_DIR=$SCRATCH/pd-serve/runtime/uccl-fix-20260824/plugins
ROUTER_PREFIX=$PROJECTDIR/fergus.s6p/pd-stack/router-prefix

# --- Ports --------------------------------------------------------------------
PREFILL_PORT=57000
DECODE_PORT=57100
LB_PORT=57200
BOOTSTRAP_PORT=57300

# --- Node allocation ----------------------------------------------------------
if [[ -n "${HOLDER:-}" ]]; then
    NODES=$(scontrol show hostnames "$(squeue -j "$HOLDER" -h -o %N)" | head -$TOTAL_NODES | paste -sd,)
else
    NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -$TOTAL_NODES | paste -sd,)
fi

LOG="${LOG:-$SCRATCH/logs/pd-boot-glm53-$(date -u +%Y%m%dT%H%M%SZ).log}"
mkdir -p "$(dirname "$LOG")"

echo "=== GLM-5.3 PD disagg boot $(date -Is) ==="
echo "nodes=$NODES"
echo "log=$LOG"
echo "NONDISAGG=${NONDISAGG:-0} EAGER=${EAGER:-0}"

wait_for_health() {
    local url=$1 name=$2 max=${3:-600}
    for i in $(seq 1 "$max"); do
        if curl -sf "$url/health" >/dev/null 2>&1; then
            echo "[$(date -Is)] $name ready at $url"
            return 0
        fi
        sleep 5
    done
    echo "[$(date -Is)] $name NOT ready after ${max}x5s at $url"
    return 1
}

# --- Non-disaggregated mode: 4-node unified EP16 server -----------------------
if [[ "${NONDISAGG:-0}" == "1" ]]; then
    UNIFIED_NODES=$(echo "$NODES" | cut -d, -f1-4)
    UNIFIED_MASTER=${UNIFIED_NODES%%,*}
    echo "=== NONDISAGG=1: unified EP16 server on $UNIFIED_NODES ==="
    srun ${HOLDER:+--overlap --jobid=$HOLDER} -N4 -n4 --ntasks-per-node=1 \
        --nodelist="$UNIFIED_NODES" --gres=gpu:4 --cpus-per-task=72 \
        --export=ALL,MASTER="$UNIFIED_MASTER",MODEL="$MODEL",SERVED_NAME="$SERVED_NAME",OVERRIDE_ARGS="$OVERRIDE_ARGS",PORT="$PREFILL_PORT",U_ROOT="$U_ROOT" \
        bash -c '
            set -e
            export T_WITH_EP=1
            source "$U_ROOT/scripts/env-U.sh" 2>/dev/null
            export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
            export SGLANG_DG_CACHE_DIR=/projects/s6p/$USER/jit-cache/dg/zai-org_GLM-5.3
            exec python -m sglang.launch_server \
                --model-path "$MODEL" --served-model-name "$SERVED_NAME" --trust-remote-code ${OVERRIDE_ARGS:+--json-model-override-args "$OVERRIDE_ARGS"} \
                --host 0.0.0.0 --port "$PORT" \
                --tp-size 16 --dp-size 16 --enable-dp-attention --enable-dp-lm-head --ep-size 16 \
                --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
                --nnodes 4 --node-rank "$SLURM_NODEID" --dist-init-addr "$MASTER:57600" --dist-timeout 3600 \
                --disable-cuda-graph \
                --chunked-prefill-size 4096 --context-length 16384 --max-running-requests 64 \
                --moe-dense-tp-size 1 --load-balance-method prefix_affinity --mem-fraction-static 0.88
        ' > "$LOG" 2>&1
    exit $?
fi

# --- SMOKE mode: 1-node/1-GPU dry-run of this launcher flow (dummy78) ---------
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_NODE=${SMOKE_NODE:-${NODES%%,*}}
    echo "=== SMOKE=1: 1-GPU dry-run on $SMOKE_NODE (model=$MODEL) ==="
    echo "launcher SGLANG_DRAFT_CAPTURE_DIR=$SGLANG_DRAFT_CAPTURE_DIR"
    srun ${HOLDER:+--overlap --jobid=$HOLDER} -N1 -n1 -w "$SMOKE_NODE" --gres=gpu:1 \
        --cpus-per-task=16 --input=none \
        --export=ALL,MODEL="$MODEL",PORT="$PREFILL_PORT",U_ROOT="$U_ROOT",SGLANG_DRAFT_CAPTURE_DIR="$SGLANG_DRAFT_CAPTURE_DIR",SGLANG_DRAFT_CAPTURE_MODES="${SGLANG_DRAFT_CAPTURE_MODES:-extend}",SGLANG_DRAFT_CAPTURE_TAG="${SGLANG_DRAFT_CAPTURE_TAG:-prefill}" \
        bash -c '
            set -e
            echo "ENGINE-CAPTURE-DIR=$SGLANG_DRAFT_CAPTURE_DIR TAG=$SGLANG_DRAFT_CAPTURE_TAG" >&2
            export T_WITH_EP=1
            source "$U_ROOT/scripts/env-U.sh" 2>/dev/null
            export PYTHONPATH=/scratch/s6p/fergus.s6p/src/sglang-draft-train-0902/python:${PYTHONPATH:-}
            export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
            export PYTHONDONTWRITEBYTECODE=1
            exec python -m sglang.launch_server --model-path "$MODEL" --load-format dummy \
                --trust-remote-code --json-model-override-args "{\"qk_rope_head_dim\": 64}" \
                --host 0.0.0.0 --port "$PORT" --tp-size 1 --random-seed 42 \
                --disable-cuda-graph \
                --chunked-prefill-size 4096 --context-length 8192 --max-running-requests 4 \
                --max-total-tokens 16384 --mem-fraction-static 0.80
        ' > "$LOG" 2>&1 &
    SRUN_PID=$!
    echo "srun launched (pid=$SRUN_PID). Waiting for servers to boot..."
    ENGINE_URL="http://$SMOKE_NODE:$PREFILL_PORT"
    wait_for_health "$ENGINE_URL" "engine" 120 || { echo "engine boot failed"; tail -50 "$LOG"; kill $SRUN_PID 2>/dev/null; exit 1; }

    LB_URL="http://$SMOKE_NODE:$LB_PORT"
    srun ${HOLDER:+--overlap --jobid=$HOLDER} -N1 -n1 -w "$SMOKE_NODE" --gres=gpu:0 --cpus-per-task=2 \
        --export=ALL,FROM_PORT="$LB_PORT",TO_PORT="$PREFILL_PORT",U_ROOT="$U_ROOT" \
        bash -c '
            export T_WITH_EP=1
            source "$U_ROOT/scripts/env-U.sh" 2>/dev/null
            exec python /scratch/s6p/fergus.s6p/grace-1m/lanes/draft-train/smoke_lb_proxy.py
        ' >> "$LOG" 2>&1 &
    LB_PID=$!
    echo "LB router launched (pid=$LB_PID)"
    wait_for_health "$LB_URL" "LB" 24 || { echo "LB boot failed"; tail -50 "$LOG"; kill $SRUN_PID $LB_PID 2>/dev/null; exit 1; }
    echo ""
    echo "============================================"
    echo "PD disagg system ready!"
    echo "  LB:       $LB_URL"
    echo "  (SMOKE: single dummy78 engine behind a TCP-proxy LB)"
    echo "============================================"
    wait $SRUN_PID
    exit $?
fi

# --- PD disagg: split nodes ---------------------------------------------------
IFS=',' read -ra NODE_ARRAY <<< "$NODES"
PREFILL_NODES="${NODE_ARRAY[0]},${NODE_ARRAY[1]},${NODE_ARRAY[2]},${NODE_ARRAY[3]}"
DECODE_NODES=$(IFS=,; echo "${NODE_ARRAY[*]:4:$DECODE_NNODES}")
PREFILL_MASTER="${NODE_ARRAY[0]}"
DECODE_MASTER="${NODE_ARRAY[4]}"

echo "prefill(4)=$PREFILL_NODES master=$PREFILL_MASTER"
echo "decode($DECODE_NNODES)=$DECODE_NODES master=$DECODE_MASTER"

# --- CUDA graph flags for decode ----------------------------------------------
GRAPH_FLAGS="--cuda-graph-backend-decode full --cuda-graph-max-bs 64 --disable-piecewise-cuda-graph"
if [[ "${EAGER:-0}" == "1" ]]; then
    GRAPH_FLAGS="--disable-cuda-graph"
fi

# --- Launch both arms in ONE srun step (shared Slingshot VNI) ------------------
echo "=== Launching PD disagg in one srun step ==="

srun ${HOLDER:+--overlap --jobid=$HOLDER} \
    -N$TOTAL_NODES -n$TOTAL_NODES --ntasks-per-node=1 --nodelist="$NODES" --gres=gpu:4 --cpus-per-task=288 \
    --export=ALL,PREFILL_MASTER="$PREFILL_MASTER",DECODE_MASTER="$DECODE_MASTER",\
PREFILL_PORT="$PREFILL_PORT",DECODE_PORT="$DECODE_PORT",BOOTSTRAP_PORT="$BOOTSTRAP_PORT",DECODE_NNODES="$DECODE_NNODES",DECODE_WORLD="$DECODE_WORLD",\
MODEL="$MODEL",SERVED_NAME="$SERVED_NAME",OVERRIDE_ARGS="$OVERRIDE_ARGS",DECODE_A2A="$DECODE_A2A",CTX="$CTX",MTT_DECODE="${MTT_DECODE:-150000}",MRR_DECODE="${MRR_DECODE:-512}",HISPARSE_RATIO="$HISPARSE_RATIO",HISPARSE_BUF="$HISPARSE_BUF",U_ROOT="$U_ROOT",GRAPH_FLAGS="$GRAPH_FLAGS",SGLANG_TREE="$SGLANG_TREE",SGLANG_MAP_HOST_POOL_PRIVATE="$SGLANG_MAP_HOST_POOL_PRIVATE",SGLANG_DRAFT_CAPTURE_DIR="$SGLANG_DRAFT_CAPTURE_DIR",SGLANG_DRAFT_CAPTURE_MODES="${SGLANG_DRAFT_CAPTURE_MODES:-extend}",SGLANG_DRAFT_CAPTURE_TAG="${SGLANG_DRAFT_CAPTURE_TAG:-prefill}" \
    bash -c '
        set -e
        export T_WITH_EP=1
        source "$U_ROOT/scripts/env-U.sh" 2>/dev/null
        export PYTHONPATH="$SGLANG_TREE/python:${PYTHONPATH:-}"
        export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
        export SGLANG_DISABLE_DSA_INDEXER_FUSION=1
        export SGLANG_JIT_DEEPGEMM_PRECOMPILE=false
        export SGLANG_DG_CACHE_DIR=/projects/s6p/$USER/jit-cache/dg/zai-org_GLM-5.3
        export FI_LOG_LEVEL=warn FI_LOG_PROV=cxi
        export FI_CXI_DISABLE_DMABUF_CUDA="${FI_CXI_DISABLE_DMABUF_CUDA:-1}"
        KV_DTYPE=(--kv-cache-dtype fp8_e4m3)

        NODEID=${SLURM_NODEID:-0}

        if [ "$NODEID" -lt 4 ]; then
            # --- PREFILL (nodes 0-3): PP4×EP4, DeepEP normal, chunk 8192 ---
            NODE_RANK=$NODEID
            echo "[$(date -Is)] PREFILL starting on $(hostname) node_rank=$NODE_RANK"
            export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
            exec python -m sglang.launch_server \
                --model-path "$MODEL" --served-model-name "$SERVED_NAME" --trust-remote-code ${OVERRIDE_ARGS:+--json-model-override-args "$OVERRIDE_ARGS"} \
                --host 0.0.0.0 --port "$PREFILL_PORT" \
                --tp-size 4 --pp-size 4 --dp-size 4 --enable-dp-attention --ep-size 4 \
                --moe-a2a-backend deepep --deepep-mode normal --moe-runner-backend deep_gemm \
                "${KV_DTYPE[@]}" \
                --nnodes 4 --node-rank "$NODE_RANK" \
                --dist-init-addr "$PREFILL_MASTER:57400" --dist-timeout 3600 \
                --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
                --disable-cuda-graph --chunked-prefill-size 8192 --max-prefill-tokens 20480 \
                --context-length "$CTX" --max-running-requests 64 ${SPEC_FLAGS:-} ${PREFILL_FLAGS:-} \
                --moe-dense-tp-size 1 --load-balance-method "${PREFILL_LB:-prefix_affinity}" --mem-fraction-static 0.85 \
               
        else
            # --- DECODE (nodes 4-7): EP16 MEGAKERNEL, FULL CUDA graphs ---
            # env-U supplies NIXL(UCCL)/CXI; the megakernel branch sglang tree +
            # kernel python go FIRST on PYTHONPATH (carries the shared-expert
            # TP1 fix, c68411e).
            NODE_RANK=$((NODEID - 4))
            echo "[$(date -Is)] DECODE(mk) starting on $(hostname) node_rank=$NODE_RANK"
            A2A_FLAGS="--moe-a2a-backend deepep --deepep-mode auto --moe-runner-backend deep_gemm"
            if [ "$DECODE_A2A" = megakernel ]; then
                export MEGAKERNEL_SRC=$SCRATCH/src/megakernel
                export MEGAKERNEL_BUILD_DIR=$SCRATCH/megakernel-build
                export PYTHONPATH="$MEGAKERNEL_SRC/python:$SGLANG_TREE/python:${PYTHONPATH:-}"
                export SGLANG_MEGAKERNEL_NUM_MAX_TOKENS_PER_RANK=64
                export TORCH_CUDA_ARCH_LIST=9.0a
                A2A_FLAGS="--moe-a2a-backend megakernel"
            fi
            exec python -m sglang.launch_server \
                --model-path "$MODEL" --served-model-name "$SERVED_NAME" --trust-remote-code ${OVERRIDE_ARGS:+--json-model-override-args "$OVERRIDE_ARGS"} \
                --host 0.0.0.0 --port "$DECODE_PORT" \
                --tp-size $DECODE_WORLD --dp-size $DECODE_WORLD --enable-dp-attention --enable-dp-lm-head --ep-size $DECODE_WORLD \
                --moe-dense-tp-size 1 --load-balance-method "${DECODE_LB:-prefix_affinity}" \
                ${A2A_FLAGS} \
                --nnodes $DECODE_NNODES --node-rank "$NODE_RANK" \
                --dist-init-addr "$DECODE_MASTER:57500" --dist-timeout 3600 \
                --disaggregation-mode decode --disaggregation-transfer-backend nixl \
                --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
                ${GRAPH_FLAGS} \
                "${KV_DTYPE[@]}" \
                --enable-hisparse --disaggregation-decode-enable-radix-cache \
                --hisparse-config "{\"top_k\": 2048, \"device_buffer_size\": $HISPARSE_BUF, \"host_to_device_ratio\": $HISPARSE_RATIO}" \
                --chunked-prefill-size 4096 --context-length "$CTX" \
                --max-total-tokens "${MTT_DECODE:-150000}" \
                --max-running-requests "${MRR_DECODE:-512}" --mem-fraction-static 0.85 ${SPEC_FLAGS:-} ${DECODE_FLAGS:-} \
               
        fi
    ' > "$LOG" 2>&1 &

SRUN_PID=$!
echo "srun launched (pid=$SRUN_PID). Waiting for servers to boot..."

# --- Health check helpers -----------------------------------------------------

PREFILL_URL="http://$PREFILL_MASTER:$PREFILL_PORT"
DECODE_URL="http://$DECODE_MASTER:$DECODE_PORT"
LB_URL="http://$DECODE_MASTER:$LB_PORT"

# Wait for prefill (up to 50 min for weight loading + JIT)
wait_for_health "$PREFILL_URL" "prefill" 600 || {
    echo "prefill boot failed"; tail -50 "$LOG"; kill $SRUN_PID 2>/dev/null; exit 1;
}

# Wait for decode
wait_for_health "$DECODE_URL" "decode" 600 || {
    echo "decode boot failed"; tail -50 "$LOG"; kill $SRUN_PID 2>/dev/null; exit 1;
}

# --- Start LB router on decode master -----------------------------------------
echo "[$(date -Is)] Starting LB router: prefill=$PREFILL_URL decode=$DECODE_URL lb=$LB_URL"

srun ${HOLDER:+--overlap --jobid=$HOLDER} -N1 -n1 -w "$DECODE_MASTER" --gres=gpu:0 --cpus-per-task=4 \
    --export=ALL,PREFILL_URL="$PREFILL_URL",DECODE_URL="$DECODE_URL",LB_URL="$LB_URL",\
LB_PORT="$LB_PORT",BOOTSTRAP_PORT="$BOOTSTRAP_PORT",ROUTER_PREFIX="$ROUTER_PREFIX",U_ROOT="$U_ROOT" \
    bash -c '
        export T_WITH_EP=1
        source "$U_ROOT/scripts/env-U.sh" 2>/dev/null
        export PYTHONPATH="$ROUTER_PREFIX:${PYTHONPATH:-}"
        exec python -m sglang_router.launch_router \
            --pd-disaggregation \
            --prefill "$PREFILL_URL" "$BOOTSTRAP_PORT" \
            --decode "$DECODE_URL" \
            --host 0.0.0.0 --port "$LB_PORT"
    ' >> "$LOG" 2>&1 &

LB_PID=$!
echo "LB router launched (pid=$LB_PID)"

# Wait for LB to be ready
wait_for_health "$LB_URL" "LB" 24 || { echo "LB boot failed"; tail -50 "$LOG"; kill $SRUN_PID $LB_PID 2>/dev/null; exit 1; }

echo ""
echo "============================================"
echo "PD disagg system ready!"
echo "  LB:       $LB_URL"
echo "  Prefill:  $PREFILL_URL"
echo "  Decode:   $DECODE_URL"
echo "  Log:      $LOG"
echo "============================================"
echo ""
echo "Test with:"
echo "  curl $LB_URL/v1/models"
echo "  curl -s $LB_URL/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"glm-5.3-fp8\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":32}'"

# Wait for the srun job
wait $SRUN_PID
