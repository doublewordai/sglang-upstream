from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class HostPoolExhaustedError(RuntimeError):
    """The HiSparse host KV pool has no whole pages left for a request.

    Raised by :meth:`HiSparseHostPoolMixin.alloc_paged_token_slots`. Decode
    pre-allocation treats it as a "not yet" admission signal (the request
    stays queued and is retried after eviction frees host rows), never as a
    scheduler-fatal error. Subclasses ``RuntimeError`` so pre-existing
    handlers keep working.
    """


class HiSparseHostPoolMixin:
    def _round_up_to_page_size(self, size: int) -> int:
        return (size + self.page_size - 1) // self.page_size * self.page_size

    def host_pages_needed(
        self, start_pos: int, num_tokens: int, allocated_len: int
    ) -> int:
        """Whole pages ``alloc_paged_token_slots`` would newly allocate for
        [start_pos, start_pos+num_tokens) given an already-allocated length.
        Mirrors the accounting in alloc_paged_token_slots exactly, so an
        admission pre-check agrees with the allocation it guards."""
        if num_tokens <= 0:
            return 0
        page_end = self._round_up_to_page_size(start_pos + num_tokens)
        if page_end <= allocated_len:
            return 0
        return (page_end - allocated_len) // self.page_size

    def has_free_pages(self, num_pages: int) -> bool:
        """Whether ``alloc_page(num_pages)`` can succeed right now."""
        return num_pages * self.page_size <= self.available_size()

    def alloc_page(self, num_pages: int) -> Optional[torch.Tensor]:
        host_locs = self.alloc(num_pages * self.page_size)
        if host_locs is not None:
            self._assert_whole_pages(host_locs)
        return host_locs

    def _assert_whole_pages(self, host_locs: torch.Tensor) -> None:
        """Transfer destinations name host rows by page id (row // page_size),
        so every page-sized run of an allocation must be one aligned page."""
        runs = host_locs.view(-1, self.page_size)
        offsets = torch.arange(self.page_size, dtype=runs.dtype)
        aligned = bool((runs[:, 0] % self.page_size == 0).all())
        contiguous = bool((runs == runs[:, :1] + offsets[None, :]).all())
        assert aligned and contiguous, (
            f"HiSparse host alloc of {runs.shape[0]} pages returned runs that are "
            "not whole aligned pages; transfer destinations index host rows by page."
        )

    def alloc_paged_token_slots(
        self,
        req_to_host_pool: torch.Tensor,
        req_to_host_pool_allocated_len: torch.Tensor,
        req_pool_idx: int,
        start_pos: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """Allocate request host slots by page and return token-granular slots."""
        device = req_to_host_pool.device
        if num_tokens <= 0:
            return torch.empty((0,), dtype=torch.int64, device=device)

        allocated_len = int(req_to_host_pool_allocated_len[req_pool_idx])
        end_pos = start_pos + num_tokens
        page_end = self._round_up_to_page_size(end_pos)
        assert start_pos <= allocated_len

        if page_end > allocated_len:
            num_new_pages = (page_end - allocated_len) // self.page_size
            host_locs = self.alloc_page(num_new_pages)
            if host_locs is None:
                logger.error(
                    "HiSparse: host mem pool alloc failed for %d host pages "
                    "(req_pool_idx=%d, start_pos=%d, num_tokens=%d)",
                    num_new_pages,
                    req_pool_idx,
                    start_pos,
                    num_tokens,
                )
                raise HostPoolExhaustedError(
                    f"HiSparse host mem pool alloc failed for {num_new_pages} pages"
                )

            req_to_host_pool[req_pool_idx, allocated_len:page_end] = host_locs.to(
                device=device, non_blocking=True
            )
            req_to_host_pool_allocated_len[req_pool_idx] = page_end

        return req_to_host_pool[req_pool_idx, start_pos:end_pos]

    def allocated_host_indices(
        self,
        req_to_host_pool: torch.Tensor,
        req_pool_idx: int,
        allocated_len: int,
    ) -> torch.Tensor:
        allocated_len = int(allocated_len)
        host_len = min(
            self._round_up_to_page_size(allocated_len),
            req_to_host_pool.shape[1],
        )
        host_indices = req_to_host_pool[req_pool_idx, :host_len]
        return host_indices[host_indices >= 0]
