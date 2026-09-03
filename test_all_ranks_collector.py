"""Unit tests for AllRanksLoadSnapshotCollector (metrics-allranks lane).

Stubs the heavy sglang module deps so metrics_collector.py can be imported
standalone; exercises the scrape-time collector against a fake reader and a
fake multiprocess collector.

Run: python3 test_all_ranks_collector.py  (exit 0 = all pass)
"""
import sys
import types
from dataclasses import dataclass, field


# ---- stub the sglang modules metrics_collector imports at module level ----
def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


from enum import Enum


class _DisaggMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode):
        return {"prefill": "prefill", "decode": "decode", "null": "engine"}[mode]


_stub("sglang")
_stub("sglang.srt")
_stub(
    "sglang.srt.disaggregation",
    utils=types.SimpleNamespace(DisaggregationMode=_DisaggMode),
)
sys.modules["sglang.srt.disaggregation.utils"] = sys.modules[
    "sglang.srt.disaggregation"
].utils
_stub("sglang.srt.model_executor")
_stub("sglang.srt.model_executor.forward_batch_info", ForwardMode=None)
_obs = _stub("sglang.srt.observability", utils=types.SimpleNamespace(exponential_buckets=None, generate_buckets=None))
_obs.__path__ = ["python/sglang/srt/observability"]  # real dir: metrics_collector imports as a submodule
sys.modules["sglang.srt.observability.utils"] = _obs.utils
_stub("sglang.srt.server_args", ServerArgs=object)
_stub("sglang.srt.utils", get_bool_env_var=lambda *a, **k: False)
_gh = types.ModuleType("sglang.srt.utils.gauge_histogram")
_gh.GaugeHistogram = object
sys.modules["sglang.srt.utils.gauge_histogram"] = _gh

sys.path.insert(0, "python")
from sglang.srt.observability.metrics_collector import (  # noqa: E402
    AllRanksLoadSnapshotCollector,
    build_load_snapshot_metrics_collector,
)


# ---- fakes -----------------------------------------------------------------
@dataclass
class QueueMetrics:
    waiting: int = 0
    grammar: int = 0
    paused: int = 0
    retracted: int = 0
    prealloc_ready: int = 0


@dataclass
class DisaggMetrics:
    mode: str = "decode"
    decode_prealloc_queue_reqs: int = 0


@dataclass
class Snap:
    dp_rank: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    moe_ep_rank: int = 0
    node_rank: int = 0
    host_pool_free_tokens: int = 0
    host_pool_total_tokens: int = 0
    host_pool_pinned_tokens: int = 0
    host_pool_evictable_tokens: int = 0
    host_pool_wait_events: int = 0
    host_pool_wait_age_s: float = 0.0
    queues: QueueMetrics = None
    disaggregation: DisaggMetrics = None


class FakeReader:
    def __init__(self, snaps):
        self.snaps = snaps

    def read_all(self):
        return self.snaps


class FakeMetric:
    def __init__(self, name, samples):
        self.name = name
        self.samples = samples


class FakeMP:
    def __init__(self, names):
        self.names = names

    def collect(self):
        return [FakeMetric(n, [(n, i) for i in range(2)]) for n in self.names]


def collect_text(collector):
    out = []
    for fam in collector.collect():
        name = getattr(fam, "name", None)
        for s in fam.samples:
            if hasattr(s, "value") and hasattr(s, "labels"):  # prometheus Sample
                out.append((name, tuple(sorted(s.labels.items())), s.value))
            elif isinstance(s, tuple):  # passthrough FakeMetric sample
                out.append((name, (), s[1] if len(s) > 1 else s))
            else:
                out.append((name, (), s))
    return out


def families(collector):
    return {fam.name for fam in collector.collect()}


POOL_NAMES = {
    "sglang:hisparse_host_pool_free_tokens",
    "sglang:hisparse_host_pool_total_tokens",
    "sglang:hisparse_host_pool_pinned_tokens",
    "sglang:hisparse_host_pool_evictable_tokens",
    "sglang:hisparse_host_pool_wait_events",
    "sglang:hisparse_host_pool_wait_age_s",
}
PREALLOC_Q = "sglang:num_decode_prealloc_queue_reqs"
QUEUE = "sglang:num_queue_reqs"

passed = failed = 0


def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {extra}")


# ---- case 1: single-node (all node_rank 0) -> pure passthrough -------------
print("case 1: single-node passthrough")
c = AllRanksLoadSnapshotCollector(
    FakeReader([Snap(dp_rank=0), Snap(dp_rank=1, queues=QueueMetrics(3))]),
    model_name="m",
    engine_type="decode",
    local_node_rank=0,
    enable_priority_scheduling=False,
    decode_hisparse=True,
)
c.attach_multiprocess_collector(FakeMP(POOL_NAMES | {QUEUE, "sglang:other"}))
fams = families(c)
check("mmdb families all pass through", fams == POOL_NAMES | {QUEUE, "sglang:other"}, fams)

# ---- case 2: 2-node decode+hisparse: takeover, all ranks, one family -------
print("case 2: 2-node decode+hisparse takeover")
snaps = [
    Snap(dp_rank=0, tp_rank=0, host_pool_free_tokens=100, host_pool_total_tokens=1000,
         host_pool_pinned_tokens=900, host_pool_evictable_tokens=400,
         host_pool_wait_events=2, host_pool_wait_age_s=0.0,
         queues=QueueMetrics(waiting=5),
         disaggregation=DisaggMetrics(decode_prealloc_queue_reqs=0)),
    Snap(dp_rank=1, tp_rank=1, node_rank=1, host_pool_free_tokens=50,
         host_pool_total_tokens=1000, host_pool_pinned_tokens=950,
         host_pool_evictable_tokens=250, host_pool_wait_events=7,
         host_pool_wait_age_s=130.5,
         queues=QueueMetrics(waiting=8),
         disaggregation=DisaggMetrics(decode_prealloc_queue_reqs=3)),
]
c = AllRanksLoadSnapshotCollector(
    FakeReader(snaps),
    model_name="m",
    engine_type="decode",
    local_node_rank=0,
    enable_priority_scheduling=False,
    decode_hisparse=True,
)
c.attach_multiprocess_collector(FakeMP(POOL_NAMES | {QUEUE, PREALLOC_Q, "sglang:other"}))
txt = collect_text(c)
fams = families(c)
check("pool+queue families emitted once", fams == POOL_NAMES | {QUEUE, PREALLOC_Q, "sglang:other"}, fams)
wa = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_wait_age_s")
check("wait_age both ranks (0.0 / 130.5)", wa == [0.0, 130.5], wa)
pq = sorted(v for (n, l, v) in txt if n == PREALLOC_Q)
check("prealloc queue both ranks (0 / 3)", pq == [0, 3], pq)
pins = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_pinned_tokens")
check("pinned both ranks", pins == [900, 950], pins)
wait = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_wait_events")
check("wait_events both ranks", wait == [2, 7], wait)
q = sorted(v for (n, l, v) in txt if n == QUEUE)
check("queue both ranks", q == [5, 8], q)
lbls = sorted(
    tuple(sorted(dict(l).items()))
    for (n, l, v) in txt
    if n == "sglang:hisparse_host_pool_pinned_tokens"
)
want = sorted([
    (("dp_rank", "0"), ("engine_type", "decode"), ("model_name", "m"), ("moe_ep_rank", "0"),
     ("pp_rank", "0"), ("tp_rank", "0")),
    (("dp_rank", "1"), ("engine_type", "decode"), ("model_name", "m"), ("moe_ep_rank", "0"),
     ("pp_rank", "0"), ("tp_rank", "1")),
])
check("labels exact scheduler set", lbls == want, lbls)

# ---- case 3: decode arm WITHOUT hisparse -> -1s ----------------------------
print("case 3: decode without hisparse -> -1 convention")
c = AllRanksLoadSnapshotCollector(
    FakeReader([Snap(dp_rank=0), Snap(dp_rank=1, node_rank=1)]),
    model_name="m", engine_type="decode", local_node_rank=0,
    enable_priority_scheduling=False, decode_hisparse=False,
)
c.attach_multiprocess_collector(FakeMP(POOL_NAMES | {QUEUE}))
txt = collect_text(c)
pins = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_pinned_tokens")
check("pinned -1 both ranks", pins == [-1, -1], pins)
free = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_free_tokens")
check("free -1 both ranks", free == [-1, -1], free)
wa = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_wait_age_s")
check("wait_age -1 both ranks", wa == [-1.0, -1.0], wa)

# ---- case 4: prefill arm with hicache host pool: free/total real, others -1
print("case 4: prefill arm hicache")
c = AllRanksLoadSnapshotCollector(
    FakeReader([
        Snap(dp_rank=0, host_pool_free_tokens=10, host_pool_total_tokens=100),
        Snap(dp_rank=1, node_rank=1, host_pool_free_tokens=20, host_pool_total_tokens=100),
    ]),
    model_name="m", engine_type="prefill", local_node_rank=0,
    enable_priority_scheduling=False, decode_hisparse=False,
)
c.attach_multiprocess_collector(FakeMP(POOL_NAMES | {QUEUE}))
txt = collect_text(c)
tot = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_total_tokens")
pin = sorted(v for (n, l, v) in txt if n == "sglang:hisparse_host_pool_pinned_tokens")
check("total real both ranks", tot == [100, 100], tot)
check("pinned -1 (not reported on prefill)", pin == [-1, -1], pin)

# ---- case 5: priority scheduling on -> queue stays with mmdb ---------------
print("case 5: priority scheduling keeps mmdb num_queue_reqs")
c = AllRanksLoadSnapshotCollector(
    FakeReader([Snap(dp_rank=0, queues=QueueMetrics(1)),
                Snap(dp_rank=1, node_rank=1, queues=QueueMetrics(2))]),
    model_name="m", engine_type="decode", local_node_rank=0,
    enable_priority_scheduling=True, decode_hisparse=False,
)
c.attach_multiprocess_collector(FakeMP(POOL_NAMES | {QUEUE}))
txt = collect_text(c)
q_samples = [v for (n, l, v) in txt if n == QUEUE]
# FakeMP samples pass through as tuples, not values: presence check only
check("queue family passes through (mmdb wins)",
      any(n == QUEUE for (n, l, v) in txt), q_samples)
check("pool families still taken over",
      {n for (n, l, v) in txt if n in POOL_NAMES} == POOL_NAMES)
lbls = [dict(l) for (n, l, v) in txt if n == "sglang:hisparse_host_pool_free_tokens"]
check("priority label present", all(l.get("priority") == "" for l in lbls), lbls)

# ---- case 6: remote snapshot without queues -> queue passes through --------
print("case 6: no queues in snapshots -> mmdb queue passthrough")
c = AllRanksLoadSnapshotCollector(
    FakeReader([Snap(dp_rank=0), Snap(dp_rank=1, node_rank=1)]),
    model_name="m", engine_type="decode", local_node_rank=0,
    enable_priority_scheduling=False, decode_hisparse=True,
)
c.attach_multiprocess_collector(FakeMP({QUEUE} | POOL_NAMES))
txt = collect_text(c)
check("queue from mmdb (no snapshot queues)",
      any(n == QUEUE for (n, l, v) in txt))
check("pool families from snapshots",
      {n for (n, l, v) in txt if n in POOL_NAMES} == POOL_NAMES)

# ---- case 7: reader exception -> passthrough, no crash ---------------------
print("case 7: reader failure degrades to passthrough")


class BrokenReader:
    def read_all(self):
        raise RuntimeError("shm gone")


c = AllRanksLoadSnapshotCollector(
    BrokenReader(), model_name="m", engine_type="decode", local_node_rank=0,
    enable_priority_scheduling=False, decode_hisparse=True,
)
c.attach_multiprocess_collector(FakeMP({QUEUE} | POOL_NAMES))
fams = families(c)
check("all mmdb families survive", fams == POOL_NAMES | {QUEUE}, fams)

# ---- case 8: factory guards ------------------------------------------------
print("case 8: build_load_snapshot_metrics_collector guards")


class FakeTM:
    def __init__(self, **kw):
        self.__dict__.update(kw)


check("None tm -> None", build_load_snapshot_metrics_collector(None) is None)
check("no reader -> None",
      build_load_snapshot_metrics_collector(FakeTM(server_args=object())) is None)
sa = types.SimpleNamespace(
    enable_metrics=True, served_model_name=["glm"], disaggregation_mode="decode",
    enable_hisparse=True, enable_priority_scheduling=False, node_rank=0,
)
tm = FakeTM(load_snapshot_reader=FakeReader([]), server_args=sa)
built = build_load_snapshot_metrics_collector(tm)
check("full tm -> collector", built is not None and built._decode_hisparse is True)
sa.enable_metrics = False
check("metrics off -> None", build_load_snapshot_metrics_collector(tm) is None)
sa.enable_metrics = True
built2 = build_load_snapshot_metrics_collector(tm)
check("string disaggregation_mode decodes (regression)",
      built2 is not None and built2._decode_hisparse is True)
sa.disaggregation_mode = "prefill"
check("prefill mode -> decode_hisparse False",
      build_load_snapshot_metrics_collector(tm)._decode_hisparse is False)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
