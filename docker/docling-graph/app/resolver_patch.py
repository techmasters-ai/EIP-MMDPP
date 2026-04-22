"""Service-side monkey-patch of the delta resolver's semantic-similarity fn.

The docling-graph library's within-pass entity resolver at
``docling_graph.core.extractors.contracts.delta.resolvers`` computes
pairwise similarity between candidate-duplicate entities in three tiers
(identity / fuzzy / semantic). The semantic tier hardcodes
``spacy.blank('en')`` — a pipeline with NO word vectors — so
``doc.vector_norm`` is always zero, ``doc.similarity(...)`` raises
``ValueError('spaCy model has no vectors')``, and the code silently
falls through to a lowercase-split Jaccard approximation
(``semantic_fallback:ValueError`` / ``:ModuleNotFoundError``).

The fallback is effectively noise for near-duplicate entity names:
"Fan Song" vs "Fan-Song" tokenize to ``{'fan', 'song'}`` vs ``{'fan-song'}``
with Jaccard = 0. So the semantic tier never actually fires.

We install a ``spacy.load('en_core_web_md')``-backed replacement at
service startup. ``en_core_web_md`` carries real word vectors
(≈40MB download, ``spacy download en_core_web_md`` in the Dockerfile).

**Idempotent** — ``install()`` is safe to call more than once; later
calls short-circuit. If the model fails to load (wrong image build,
pip cache issue), we log a WARNING and leave the library's fallback
intact so the service still runs.

See ontology_bundles/_shared/prompt_rules.py for the same pattern
applied to the LLM system prompt.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = False

_MODEL_NAME = "en_core_web_md"


def install() -> None:
    """Replace the delta resolver's _semantic_similarity with a
    vectors-backed version. Idempotent + best-effort.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        import spacy
    except ImportError as exc:
        logger.warning(
            "resolver_patch: spacy not installed (%s); "
            "within-pass semantic resolver stays on the Jaccard fallback.",
            exc,
        )
        return

    try:
        nlp = spacy.load(_MODEL_NAME)
    except Exception as exc:
        logger.warning(
            "resolver_patch: failed to load spaCy model %r (%s); "
            "within-pass semantic resolver stays on the Jaccard fallback. "
            "Confirm Dockerfile runs 'python -m spacy download %s'.",
            _MODEL_NAME, exc, _MODEL_NAME,
        )
        return

    # Sanity-check the model actually carries vectors.
    probe = nlp("probe")
    if not probe.vector_norm:
        logger.warning(
            "resolver_patch: loaded spaCy model %r has vector_norm=0; "
            "expected a model with word vectors (en_core_web_md / _lg). "
            "Leaving library fallback in place.", _MODEL_NAME,
        )
        return

    try:
        from docling_graph.core.extractors.contracts.delta import resolvers
    except ImportError as exc:
        logger.warning(
            "resolver_patch: couldn't import docling-graph resolvers module "
            "(%s). Library may have been refactored — patch skipped.", exc,
        )
        return

    original = resolvers._semantic_similarity

    def patched_semantic_similarity(a: str, b: str) -> tuple[float, str | None]:
        """Drop-in replacement: uses the pre-loaded vectors model.

        Returns (score, None) on success — no fallback reason. On
        unexpected exception, defers to the library's original impl so
        we don't silently corrupt resolver behavior.
        """
        if not a or not b:
            return 0.0, None
        try:
            doc_a = nlp(a)
            doc_b = nlp(b)
            if not doc_a.vector_norm or not doc_b.vector_norm:
                # OOV both sides — fall back to exact-string via the
                # library's original (which hits Jaccard).
                return original(a, b)
            return float(doc_a.similarity(doc_b)), None
        except Exception as exc:
            logger.debug(
                "resolver_patch: spaCy similarity failed for (%r, %r): %s — "
                "falling back to library default.", a, b, exc,
            )
            return original(a, b)

    patched_semantic_similarity.__wrapped__ = original  # type: ignore[attr-defined]
    resolvers._semantic_similarity = patched_semantic_similarity

    _INSTALLED = True
    logger.warning(
        "resolver_patch: installed vectors-backed _semantic_similarity "
        "(model=%r, vector_size=%d).", _MODEL_NAME, nlp.vocab.vectors_length,
    )
