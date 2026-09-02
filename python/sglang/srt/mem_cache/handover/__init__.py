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
  decode_arm  - HiSparseRadixCache host rows + device index keys export/import
"""
