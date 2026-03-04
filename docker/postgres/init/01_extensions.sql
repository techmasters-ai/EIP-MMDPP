-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";
CREATE EXTENSION IF NOT EXISTS "vector";        -- pgvector
CREATE EXTENSION IF NOT EXISTS "age";           -- Apache AGE graph
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Load AGE into the search path for Cypher queries
LOAD 'age';
ALTER DATABASE eip SET search_path = ag_catalog, "$user", public;

-- Set search_path for THIS session (ALTER DATABASE only affects future connections)
SET search_path = ag_catalog, "$user", public;

-- Create the military equipment knowledge graph
SELECT create_graph('eip_kg');
