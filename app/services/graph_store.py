"""Backend-agnostic GraphStore Protocol.

All graph + vector operations in the application flow through this Protocol,
allowing graph backends (ArcadeDB, etc.) to be swapped without
touching the rest of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceMetadata:
    """Provenance information linking a graph element back to a source document."""

    document_id: str
    page_numbers: list[int] = field(default_factory=list)
    upload_datetime: str | None = None
    document_datetime: str | None = None


@dataclass
class NodeRecord:
    """A graph node to be created or updated."""

    entity_type: str
    identity_fields: dict[str, Any]
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float = 1.0


@dataclass
class RelationshipRecord:
    """A directed relationship between two graph nodes."""

    from_type: str
    from_identity: dict[str, Any]
    to_type: str
    to_identity: dict[str, Any]
    rel_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float = 1.0


@dataclass
class GraphEntityResult:
    """A graph entity returned from a query."""

    node_id: str
    name: str
    entity_type: str
    canonical_name: str | None = None
    extraction_confidence: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaSyncReport:
    """Summary of a schema-sync operation."""

    types_created: int = 0
    properties_added: int = 0
    indexes_created: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class TextChunkRecord:
    """A TextChunk vertex to be created with an optional embedding."""

    chunk_id: str
    text: str
    document_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class ImageChunkRecord:
    """An ImageChunk vertex to be created with an optional embedding."""

    chunk_id: str
    document_id: str
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class EntityChunkEdge:
    """An EXTRACTED_FROM edge from an entity (by name+type) to a chunk RID."""

    entity_name: str
    entity_type: str
    chunk_rid: str


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphStore(Protocol):
    """Abstract interface for graph + vector storage backends.

    Implementations must provide both async methods (for FastAPI request
    handlers) and sync variants (for Celery workers and other sync contexts).
    """

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def upsert_node(
        self,
        record: NodeRecord,
        provenance: ProvenanceMetadata | None = None,
    ) -> str:
        """Upsert a single node and return its backend ID."""
        ...

    async def upsert_nodes_batch(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert multiple nodes and return their backend IDs."""
        ...

    async def upsert_document_node(
        self,
        document_id: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Upsert the root Document node for a document and return its ID."""
        ...

    async def create_text_chunk_vertex(
        self,
        chunk_id: str,
        text: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Create a TextChunk vertex (optionally with embedding) and return its ID."""
        ...

    async def create_image_chunk_vertex(
        self,
        chunk_id: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Create an ImageChunk vertex (optionally with embedding) and return its ID."""
        ...

    # ------------------------------------------------------------------
    # Edge / relationship operations
    # ------------------------------------------------------------------

    async def upsert_relationship(
        self,
        record: RelationshipRecord,
        provenance: ProvenanceMetadata | None = None,
    ) -> str:
        """Upsert a single relationship and return its backend ID."""
        ...

    async def upsert_relationships_batch(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Upsert multiple relationships and return their backend IDs."""
        ...

    async def create_structural_edge(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Create a structural edge (e.g. EXTRACTED_FROM) and return its ID."""
        ...

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    async def search_nodes(
        self,
        entity_type: str,
        filters: dict[str, Any] | None = None,
        limit: int = 20,
    ) -> list[GraphEntityResult]:
        """Search for nodes matching optional filters."""
        ...

    async def resolve_root_entity(
        self,
        name: str,
        entity_type: str | None = None,
    ) -> GraphEntityResult | None:
        """Resolve a name to its canonical root entity."""
        ...

    async def fulltext_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> list[GraphEntityResult]:
        """Full-text search across node properties."""
        ...

    async def get_neighborhood(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> list[GraphEntityResult]:
        """Return nodes in the k-hop neighbourhood of *node_id*."""
        ...

    async def get_neighborhood_graph(
        self,
        node_id: str,
        depth: int = 1,
        rel_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return neighbourhood as a {nodes, edges} dict."""
        ...

    async def get_ontology_linked_chunks(
        self,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Return text/image chunks that mention *node_id* via ontology edges."""
        ...

    async def get_graph_stats(self) -> dict[str, Any]:
        """Return backend-level graph statistics."""
        ...

    async def get_relationship_count(
        self,
        rel_type: str | None = None,
    ) -> int:
        """Return the number of relationships, optionally filtered by type."""
        ...

    async def get_co_extracted_entities(
        self,
        node_id: str,
        limit: int = 10,
    ) -> list[GraphEntityResult]:
        """Return entities that co-occur with *node_id* in the same source chunk."""
        ...

    # ------------------------------------------------------------------
    # Alias operations
    # ------------------------------------------------------------------

    async def create_alias(
        self,
        node_id: str,
        alias: str,
    ) -> None:
        """Attach an alias string to an existing node."""
        ...

    async def search_by_alias(
        self,
        alias: str,
        entity_type: str | None = None,
    ) -> list[GraphEntityResult]:
        """Find nodes by alias."""
        ...

    async def set_canonical_name(
        self,
        node_id: str,
        canonical_name: str,
    ) -> None:
        """Set the canonical name on a node."""
        ...

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    async def vector_search(
        self,
        vertex_type: str,
        embedding_property: str,
        query_vector: list[float],
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[GraphEntityResult]:
        """ANN search over node embeddings for a given vertex type and property."""
        ...

    async def set_vertex_embedding(
        self,
        vertex_type: str,
        vertex_id: str,
        embedding_property: str,
        embedding: list[float],
    ) -> None:
        """Attach a vector embedding to a node by RID."""
        ...

    async def set_vertex_embedding_by_chunk_id(
        self,
        chunk_id: str,
        embedding: list[float],
        model_name: str | None = None,
    ) -> None:
        """Update embedding on a vertex looked up by chunk_id."""
        ...

    async def graph_vector_search(
        self,
        root_id: str,
        query_vector: list[float],
        embedding_property: str = "text_embedding",
        traversal: str = "out('EXTRACTED_FROM')",
        similarity_threshold: float = 0.7,
        limit: int = 10,
    ) -> list[dict]:
        """Graph traversal filtered by vector similarity (ArcadeDB §4.13.6).

        Traverses from *root_id* via *traversal*, then filters results by
        ``vectorCosineSimilarity`` on the target's embedding property.
        Returns only records above *similarity_threshold*, ranked by similarity.
        """
        ...

    async def cross_model_search(
        self,
        text_embedding: list[float],
        image_embedding: list[float],
        limit: int = 10,
    ) -> list[GraphEntityResult]:
        """Search using both text and image embeddings simultaneously."""
        ...

    # ------------------------------------------------------------------
    # Community operations
    # ------------------------------------------------------------------

    async def run_community_algorithm(
        self,
        algorithm: str,
        params: dict,
    ) -> list[dict]:
        """Run a community detection algorithm and return per-node results."""
        ...

    async def get_community_reports(self) -> list[dict]:
        """Return all existing CommunityReport rows."""
        ...

    async def upsert_community_report(
        self,
        community_id: int,
        title: str,
        summary: str,
        member_count: int,
        membership_hash: str,
        model_name: str,
    ) -> str:
        """Upsert a CommunityReport vertex and return its RID."""
        ...

    async def list_community_reports(
        self,
        limit: int = 100,
    ) -> list[dict]:
        """Return community reports (summary view) up to *limit*."""
        ...

    async def get_community_report(
        self,
        community_id: int,
    ) -> dict | None:
        """Return a single community report by community_id, or None."""
        ...

    async def search_community_reports_by_vector(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """ANN search over CommunityReport.report_embedding.

        Returns raw report rows including community_id, title, summary,
        member_count, and a similarity score.
        """
        ...

    async def get_relationships_between_entities(
        self,
        entity_names: list[str],
    ) -> list[dict]:
        """Return ontology edges where both endpoints are in *entity_names*.

        Structural edges (EXTRACTED_FROM, CONTAINS_TEXT, etc.) are excluded —
        only domain relationships between entities are returned.
        """
        ...

    # ------------------------------------------------------------------
    # Lifecycle operations
    # ------------------------------------------------------------------

    async def delete_document_graph(
        self,
        document_id: str,
    ) -> int:
        """Delete all graph elements associated with *document_id*.

        Returns the number of elements deleted.
        """
        ...

    async def sync_schema(
        self,
        ontology: dict | None = None,
    ) -> SchemaSyncReport:
        """Ensure the backend schema matches the current ontology."""
        ...

    async def ensure_indexes(self) -> None:
        """Create any missing indexes required for efficient queries."""
        ...

    async def ensure_ready(self) -> None:
        """Block until the backend is ready to accept operations."""
        ...

    # ------------------------------------------------------------------
    # Sync variants (for Celery workers and other sync contexts)
    # ------------------------------------------------------------------

    def upsert_node_sync(
        self,
        record: NodeRecord,
        provenance: ProvenanceMetadata | None = None,
    ) -> str:
        """Synchronous variant of :meth:`upsert_node`."""
        ...

    def upsert_nodes_batch_sync(
        self,
        records: list[NodeRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Synchronous variant of :meth:`upsert_nodes_batch`."""
        ...

    def upsert_relationships_batch_sync(
        self,
        records: list[RelationshipRecord],
        provenance: ProvenanceMetadata | None = None,
    ) -> list[str]:
        """Synchronous variant of :meth:`upsert_relationships_batch`."""
        ...

    def create_structural_edge_sync(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Synchronous variant of :meth:`create_structural_edge`."""
        ...

    def set_vertex_embedding_sync(
        self,
        vertex_type: str,
        vertex_id: str,
        embedding_property: str,
        embedding: list[float],
    ) -> None:
        """Synchronous variant of :meth:`set_vertex_embedding`."""
        ...

    def create_text_chunk_vertex_sync(
        self,
        chunk_id: str,
        text: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Synchronous variant of :meth:`create_text_chunk_vertex`."""
        ...

    def create_image_chunk_vertex_sync(
        self,
        chunk_id: str,
        document_id: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        """Synchronous variant of :meth:`create_image_chunk_vertex`."""
        ...

    def create_text_chunks_batch_sync(
        self,
        records: list[TextChunkRecord],
    ) -> list[str]:
        """Create multiple TextChunk vertices (with embeddings) in one batched call."""
        ...

    def create_image_chunks_batch_sync(
        self,
        records: list[ImageChunkRecord],
    ) -> list[str]:
        """Create multiple ImageChunk vertices (with embeddings) in one batched call."""
        ...

    def batch_create_entity_chunk_edges_sync(
        self,
        edges: list[EntityChunkEdge],
    ) -> int:
        """Create EXTRACTED_FROM edges from entities (by name+type) to chunk RIDs in a single batched call.

        Returns the number of edges attempted (not necessarily created — entities
        that cannot be resolved silently result in no-op sub-statements).
        """
        ...

    def get_chunk_rid_sync(
        self,
        chunk_id: str,
        chunk_type: str = "TextChunk",
    ) -> str | None:
        """Return the ArcadeDB RID for an existing chunk vertex, or None if not found."""
        ...

    def create_entity_chunk_edge_sync(
        self,
        entity_name: str,
        entity_type: str,
        chunk_rid: str,
    ) -> bool:
        """Create an EXTRACTED_FROM edge from an entity (looked up by name+type) to a chunk RID.

        Returns True if the edge was created, False if the entity was not found.
        """
        ...

    def delete_document_graph_sync(
        self,
        document_id: str,
    ) -> int:
        """Synchronous variant of :meth:`delete_document_graph`."""
        ...

    def ensure_ready_sync(self) -> None:
        """Synchronous variant of :meth:`ensure_ready`."""
        ...

    def fuzzy_match_sync(
        self,
        name: str,
        entity_type: str,
        threshold: float = 0.8,
    ) -> list[dict]:
        """Find entities with names similar to *name* using server-side
        Levenshtein distance. Returns candidates where normalized similarity
        >= *threshold*, sorted by distance (best first).
        """
        ...

    def get_document_entities_sync(
        self,
        document_id: str,
    ) -> list[dict]:
        """Return all domain entities linked to a document via graph traversal.

        Traverses Document →(CONTAINS_TEXT)→ TextChunk ←(EXTRACTED_FROM)← Entity.
        Returns raw dicts with at least ``name``, ``entity_type``, and ``@rid``.
        """
        ...

    def close_sync(self) -> None:
        """Release any held resources (connections, thread pools, etc.)."""
        ...
