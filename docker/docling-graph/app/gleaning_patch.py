"""Service-side monkey-patch: parallelize the delta orchestrator's gleaning loop.

The docling-graph library's delta orchestrator runs gleaning ("recall pass —
look at each batch again with an 'already found' summary in the prompt and
extract anything missed") as a sequential ``for`` loop over ``batch_plan`` at
``docling_graph.core.extractors.contracts.delta.orchestrator.DeltaOrchestrator.extract``.

There is no functional reason for it to be sequential: ``already_found`` is
computed once before the loop and frozen, every iteration's call to
``_run_one_batch`` is fully independent (same ``semantic_guide``,
``catalog_block``, ``global_context``, ``already_found``), and the only
shared state is two ``list.append`` operations after each call.

For a 25-batch pass with empirical mean LLM-call latency ~80s, sequential
gleaning costs ~35min wall-clock per pass. Dispatching it through a
``ThreadPoolExecutor`` sized to ``self._config.parallel_workers`` (matches
what the primary batch fan-out already uses) cuts that to ~6–8min — a
~5x speedup with zero recall change.

**Implementation strategy:** runtime monkey-patch via source rewrite
(NOT a build-time .patch file).

  - Read the upstream ``extract`` method's source via ``inspect.getsource``
  - Substring-replace the specific gleaning ``for`` loop with a parallel
    ``ThreadPoolExecutor`` + ``as_completed`` block
  - Compile the modified source in the orchestrator module's namespace so
    all the existing imports (``ThreadPoolExecutor``, ``as_completed``,
    ``build_already_found_summary_delta``, …) resolve correctly
  - Install the new function as ``DeltaOrchestrator.extract``

This approach binds to the **string shape** of one ~14-line block, not to
line numbers or to the rest of ``extract``. Upstream changes anywhere
ELSE in ``extract`` are absorbed transparently. If upstream changes the
gleaning loop itself, the substring match fails, ``install()`` logs a
WARNING, and the service runs with the original (sequential) gleaning
intact — no crash, no silent regression.

**Idempotent** — ``install()`` is safe to call more than once; later calls
short-circuit. Mirrors the pattern in ``resolver_patch.py``.
"""
from __future__ import annotations

import inspect
import logging
import textwrap
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = False

# The upstream gleaning block we expect to find. Match must be exact
# (after textwrap.dedent + strip) — any deviation in variable names, kwarg
# order, or whitespace within the block triggers a graceful fall-through.
_UPSTREAM_GLEANING_BLOCK = textwrap.dedent('''\
    for i, batch in enumerate(batch_plan):
        _batch_idx, graph_dict, _errors, _elapsed = self._run_one_batch(
            batch_index=i,
            total_batches=len(batch_plan),
            batch=batch,
            semantic_guide=semantic_guide,
            catalog_block=catalog_block,
            global_context=global_context,
            already_found=already_found,
        )
        if graph_dict is not None:
            gleaning_results.append(graph_dict)
            gleaning_batch_plan.append(batch)
''').strip()

# Replacement: dispatch the same calls through a ThreadPoolExecutor sized
# to parallel_workers. as_completed preserves the result-on-completion
# semantics; the post-loop merge step doesn't depend on order so the
# subset of batches that returned non-None still merges correctly.
_PATCHED_GLEANING_BLOCK = textwrap.dedent('''\
    _glean_workers = max(1, int(self._config.parallel_workers or 1))
    if _glean_workers > 1 and len(batch_plan) > 1:
        with ThreadPoolExecutor(max_workers=_glean_workers) as _glean_pool:
            _glean_futures = {
                _glean_pool.submit(
                    self._run_one_batch,
                    batch_index=i,
                    total_batches=len(batch_plan),
                    batch=batch,
                    semantic_guide=semantic_guide,
                    catalog_block=catalog_block,
                    global_context=global_context,
                    already_found=already_found,
                ): batch
                for i, batch in enumerate(batch_plan)
            }
            for _glean_future in as_completed(_glean_futures):
                batch = _glean_futures[_glean_future]
                _batch_idx, graph_dict, _errors, _elapsed = _glean_future.result()
                if graph_dict is not None:
                    gleaning_results.append(graph_dict)
                    gleaning_batch_plan.append(batch)
    else:
        for i, batch in enumerate(batch_plan):
            _batch_idx, graph_dict, _errors, _elapsed = self._run_one_batch(
                batch_index=i,
                total_batches=len(batch_plan),
                batch=batch,
                semantic_guide=semantic_guide,
                catalog_block=catalog_block,
                global_context=global_context,
                already_found=already_found,
            )
            if graph_dict is not None:
                gleaning_results.append(graph_dict)
                gleaning_batch_plan.append(batch)
''').strip()


def install() -> None:
    """Replace ``DeltaOrchestrator.extract`` with a parallel-gleaning version.

    Idempotent + best-effort. If anything goes wrong (import failure,
    upstream block shape changed, exec error), logs a WARNING and leaves
    the original sequential gleaning intact.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        # Late import — orchestrator pulls in heavy graph-processing code,
        # don't pay that cost unless the patch is being installed.
        from docling_graph.core.extractors.contracts.delta import orchestrator as _orch_mod
        from docling_graph.core.extractors.contracts.delta.orchestrator import (
            DeltaOrchestrator,
        )
    except ImportError as exc:
        logger.warning(
            "gleaning_patch: docling_graph.core.extractors.contracts.delta."
            "orchestrator not importable (%s); skipping. Sequential gleaning "
            "will run as upstream defines it.",
            exc,
        )
        return

    try:
        original_src = inspect.getsource(DeltaOrchestrator.extract)
    except (OSError, TypeError) as exc:
        logger.warning(
            "gleaning_patch: cannot read source of DeltaOrchestrator.extract "
            "(%s); skipping.", exc,
        )
        return

    # The gleaning loop sits inside `extract()` at 12-space indentation
    # (4 for class scope + 4 for method body + 4 for the
    # `if gleaning_enabled:` block). Re-indent both blocks to that depth
    # for a verbatim find-and-replace against the original method source.
    indent = " " * 12
    old_indented = textwrap.indent(_UPSTREAM_GLEANING_BLOCK, indent)
    new_indented = textwrap.indent(_PATCHED_GLEANING_BLOCK, indent)

    if old_indented not in original_src:
        logger.warning(
            "gleaning_patch: upstream gleaning loop signature changed — "
            "expected 12-space-indented block not found in "
            "DeltaOrchestrator.extract. Falling through; sequential "
            "gleaning will continue. Update _UPSTREAM_GLEANING_BLOCK in "
            "app/gleaning_patch.py to re-enable.",
        )
        return

    patched_src = original_src.replace(old_indented, new_indented, 1)

    # Compile and exec in the orchestrator module's globals so that
    # ThreadPoolExecutor, as_completed, build_already_found_summary_delta,
    # merge_delta_graphs, normalize_delta_ir_batch_results,
    # sanitize_batch_echo_from_graph, etc. all resolve via the
    # already-loaded module namespace.
    module_globals = _orch_mod.__dict__
    local_ns: dict[str, Any] = {}

    try:
        # Strip the leading whitespace of the def itself so exec sees a
        # top-level function definition. The body's relative indentation
        # is preserved.
        compile_src = textwrap.dedent(patched_src)
        exec(compile_src, module_globals, local_ns)  # noqa: S102 — controlled source
    except SyntaxError as exc:
        logger.warning(
            "gleaning_patch: rewrite produced invalid Python (%s); skipping. "
            "Original extract() preserved.", exc,
        )
        return

    new_extract = local_ns.get("extract")
    if new_extract is None or not callable(new_extract):
        logger.warning(
            "gleaning_patch: rewritten module did not define `extract`; "
            "skipping. Original preserved.",
        )
        return

    DeltaOrchestrator.extract = new_extract  # type: ignore[method-assign]
    _INSTALLED = True
    # WARNING level mirrors resolver_patch's convention so the startup
    # surface log shows the patch landed (the uvicorn process configures
    # only WARNING+ for the third-party namespace).
    logger.warning(
        "gleaning_patch: installed parallel gleaning on DeltaOrchestrator.extract "
        "— uses ThreadPoolExecutor sized to config.parallel_workers; "
        "upstream sequential code retained as fallback for parallel_workers <= 1.",
    )
