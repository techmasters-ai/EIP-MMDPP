"""Embedding-side renderer. Will be filled in Task 8.

Re-exports _render_column_as_text from render_graph so callers
on the embedding side import from the right module without coupling."""
from app.services.table_normalization.render_graph import _render_column_as_text

__all__ = ["_render_column_as_text"]
