"""Generation handover: warm-start a new engine generation from the old one.

The production chain hands the endpoint to fresh engines whose caches are
empty; every user's next turn re-prefills. This package exports the old
generation's host-tier radix state (page bytes + manifest), pushes it over
NIXL/UCCL (old-initiated WRITE, the shape the UCCL backend supports), and
bulk-imports it into the heir between boot and validate.

Submodules:
  manifest    - chain records, pool specs, serialization, checksums
  transfer    - NIXL agent wrapper + page-run descriptor building
  prefill_arm - HiRadixCache + HostPoolGroup (KV + DSA indexer) export/import
  admin       - engine-side orchestrators behind /handover/import|export

Also registers the ``hiradix_dsa`` radix-cache backend (HiRadixCache for DSA
models; the default registry routes DSA+HiCache to UnifiedRadixCache, which
this lane's export/import does not yet target). Select with
``--radix-cache-backend hiradix_dsa``.
"""

from __future__ import annotations


def _register_radix_backends() -> None:
    try:
        from sglang.srt.mem_cache.registry import register_radix_cache_backend
    except Exception:  # registry not importable in some test contexts
        return

    def _hiradix_dsa_factory(ctx):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        cache = HiRadixCache(params=ctx.params, server_args=ctx.server_args)
        try:
            ctx.tp_worker.register_hicache_layer_transfer_counter(
                cache.cache_controller.layer_done_counter
            )
        except Exception:
            pass
        return cache

    try:
        register_radix_cache_backend("hiradix_dsa", _hiradix_dsa_factory)
    except ValueError:
        pass  # already registered (e.g. module reloaded)


_register_radix_backends()
