// ===========================================================================
// EIP-MMDPP — Neo4j schema initialization
// Constraints, indexes, and fulltext search for the knowledge graph.
// Mounted into /docker-entrypoint-initdb.d/ for auto-execution on first start.
// ===========================================================================

// ---------------------------------------------------------------------------
// Uniqueness constraints
// ---------------------------------------------------------------------------
CREATE CONSTRAINT entity_unique IF NOT EXISTS
  FOR (n:Entity) REQUIRE (n.name, n.entity_type) IS UNIQUE;

CREATE CONSTRAINT document_unique IF NOT EXISTS
  FOR (n:Document) REQUIRE n.document_id IS UNIQUE;

CREATE CONSTRAINT chunk_unique IF NOT EXISTS
  FOR (n:ChunkRef) REQUIRE n.chunk_id IS UNIQUE;

CREATE CONSTRAINT alias_unique IF NOT EXISTS
  FOR (n:Alias) REQUIRE n.alias_name IS UNIQUE;

// ---------------------------------------------------------------------------
// Standard indexes for common lookups
// ---------------------------------------------------------------------------
CREATE INDEX entity_type_idx IF NOT EXISTS
  FOR (n:Entity) ON (n.entity_type);

CREATE INDEX entity_name_idx IF NOT EXISTS
  FOR (n:Entity) ON (n.name);

CREATE INDEX entity_canonical_idx IF NOT EXISTS
  FOR (n:Entity) ON (n.canonical_name);

CREATE INDEX chunk_document_idx IF NOT EXISTS
  FOR (n:ChunkRef) ON (n.document_id);

// ---------------------------------------------------------------------------
// Fulltext index for fuzzy entity search (used by canonicalization)
// ---------------------------------------------------------------------------
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
  FOR (n:Entity) ON EACH [n.name, n.canonical_name];
