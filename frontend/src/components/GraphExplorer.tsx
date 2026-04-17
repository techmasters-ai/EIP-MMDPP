import React, { useState, useEffect, useCallback } from "react";
import {
  getActiveQueryProfiles,
  ingestGraphEntity,
  ingestGraphRelationship,
  queryGraph,
  getGraphNeighborhood,
  getIndexingSettings,
  triggerIndexing,
  getCommunityReports,
  type IndexingSettings,
  type CommunityReport,
  type QueryProfileDefinition,
  type QueryProfileRegistry,
  type QueryResultItem,
} from "../api/client";
import { uniqueSorted, normalizeNamedList } from "../utils/ontologyHelpers";
import type cytoscape from "cytoscape";
import { GraphView, toGraphElements } from "./GraphView";
import { QueryProfileRegistryPage } from "./QueryProfileRegistryPage";

const DEFAULT_ENTITY_TYPES = [
  "EQUIPMENT_SYSTEM",
  "RADAR_SYSTEM",
  "MISSILE_SYSTEM",
  "PLATFORM",
  "SUBSYSTEM",
  "COMPONENT",
  "SPECIFICATION",
  "CAPABILITY",
  "ORGANIZATION",
  "DOCUMENT",
];

const DEFAULT_REL_TYPES = [
  "PART_OF",
  "HAS_SUBSYSTEM",
  "HAS_COMPONENT",
  "HAS_STAGE",
  "OPERATED_BY",
  "MANUFACTURED_BY",
  "INSTALLED_ON",
  "DEPLOYED_ON",
  "ASSOCIATED_WITH",
  "OPERATES_IN_BAND",
  "USES_WAVEFORM",
  "USES_MODULATION",
  "EMITS",
  "RADIATES",
  "RECEIVES",
  "HAS_SIGNATURE",
  "HAS_SCAN",
  "HAS_RECEIVER",
  "HAS_TRANSMITTER",
  "HAS_ANTENNA",
  "HAS_PROCESSING_CHAIN",
  "HAS_PERFORMANCE",
  "SPECIFIED_BY",
  "CUES",
  "GUIDES",
  "TRACKS",
  "ENGAGES",
  "DETECTS",
  "DESIGNATES",
  "SUPPORTS_ENGAGEMENT_OF",
  "HAS_GUIDANCE",
  "HAS_PROPULSION",
  "HAS_SEEKER",
];

interface ValidationRule {
  source: string;
  relationship: string;
  target: string;
}

interface OntologyOptions {
  registryName: string | null;
  ontologyName: string | null;
  entityTypes: string[];
  relationshipTypes: string[];
  validationMatrix: ValidationRule[];
}


function deriveEntityTypes(
  registry: QueryProfileRegistry | null,
  profiles: QueryProfileDefinition[],
): string[] {
  const ontologyTypes = normalizeNamedList(registry?.ontology_definition?.entity_types);
  const profileTypes = profiles.flatMap((profile) => [
    ...profile.root_entity_types,
    ...profile.target_entity_types,
  ]);
  return uniqueSorted([...ontologyTypes, ...profileTypes, ...DEFAULT_ENTITY_TYPES]);
}

function deriveRelationshipTypes(
  registry: QueryProfileRegistry | null,
  profiles: QueryProfileDefinition[],
): string[] {
  const ontologyTypes = normalizeNamedList(registry?.ontology_definition?.relationship_types);
  const profileTypes = profiles.flatMap((profile) =>
    profile.traversals.flatMap((traversal) =>
      traversal.steps.flatMap((step) => step.rel_types),
    ),
  );
  return uniqueSorted([...ontologyTypes, ...profileTypes, ...DEFAULT_REL_TYPES]);
}

function deriveValidationMatrix(registry: QueryProfileRegistry | null): ValidationRule[] {
  const raw = registry?.ontology_definition?.validation_matrix;
  if (!Array.isArray(raw)) return [];

  const normalized = raw.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    const source = record.source ?? record.source_type;
    const relationship = record.relationship ?? record.rel_type;
    const target = record.target ?? record.target_type;
    if (
      typeof source !== "string" ||
      typeof relationship !== "string" ||
      typeof target !== "string"
    ) {
      return [];
    }
    return [{
      source: source.trim(),
      relationship: relationship.trim(),
      target: target.trim(),
    }];
  });

  return normalized.sort((a, b) => {
    const left = `${a.source}:${a.relationship}:${a.target}`;
    const right = `${b.source}:${b.relationship}:${b.target}`;
    return left.localeCompare(right);
  });
}

function buildOntologyOptions(
  registry: QueryProfileRegistry | null,
  profiles: QueryProfileDefinition[],
): OntologyOptions {
  return {
    registryName: registry?.name ?? null,
    ontologyName: (registry?.ontology_name as string | null | undefined) ?? null,
    entityTypes: deriveEntityTypes(registry, profiles),
    relationshipTypes: deriveRelationshipTypes(registry, profiles),
    validationMatrix: deriveValidationMatrix(registry),
  };
}

type Tab = "search" | "entity" | "relationship" | "indexing" | "profiles";

const TAB_LABELS: Record<Tab, string> = {
  search: "Search",
  entity: "Add Entity",
  relationship: "Add Relationship",
  indexing: "Indexing",
  profiles: "Ontology and Query Profiles",
};

export function GraphExplorer() {
  const [tab, setTab] = useState<Tab>("search");
  const [ontologyOptions, setOntologyOptions] = useState<OntologyOptions>(() =>
    buildOntologyOptions(null, []),
  );

  const refreshOntologyOptions = useCallback(() => {
    getActiveQueryProfiles()
      .then((payload) => {
        setOntologyOptions(buildOntologyOptions(payload.registry, payload.exposed_profiles));
      })
      .catch(() => {
        setOntologyOptions(buildOntologyOptions(null, []));
      });
  }, []);

  useEffect(() => {
    refreshOntologyOptions();
  }, [refreshOntologyOptions]);

  return (
    <div>
      <div className="tabs" style={{ marginBottom: "1rem" }}>
        {(Object.keys(TAB_LABELS) as Tab[]).map((t) => (
          <button
            key={t}
            className={`tab-btn${tab === t ? " active" : ""}`}
            onClick={() => setTab(t)}
          >
            {TAB_LABELS[t]}
          </button>
        ))}
      </div>

      {tab === "search" && <GraphSearch />}
      {tab === "indexing" && <GlobalSearchIndexingPanel />}
      {tab === "entity" && <EntityForm entityTypes={ontologyOptions.entityTypes} />}
      {tab === "relationship" && (
        <RelationshipForm
          entityTypes={ontologyOptions.entityTypes}
          relationshipTypes={ontologyOptions.relationshipTypes}
          validationMatrix={ontologyOptions.validationMatrix}
        />
      )}
      {tab === "profiles" && <QueryProfileRegistryPage onOntologyChanged={refreshOntologyOptions} />}
    </div>
  );
}

function GraphSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [graphViewIndex, setGraphViewIndex] = useState<number | null>(null);
  const [graphElements, setGraphElements] = useState<cytoscape.ElementDefinition[] | null>(null);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await queryGraph({ query: query.trim(), top_k: 20 });
      setResults(res.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  const handleToggleGraph = async (index: number, entityName: string) => {
    if (graphViewIndex === index) {
      setGraphViewIndex(null);
      setGraphElements(null);
      setGraphError(null);
      return;
    }
    setGraphViewIndex(index);
    setGraphLoading(true);
    setGraphError(null);
    try {
      const resp = await getGraphNeighborhood({ entity_name: entityName });
      const elements = toGraphElements(resp, entityName);
      if (elements.length === 0) {
        setGraphElements(null);
        setGraphError(`No graph data found for "${entityName}".`);
      } else {
        setGraphElements(elements);
      }
    } catch (err) {
      console.error("Graph neighborhood fetch failed:", err);
      setGraphElements(null);
      setGraphError(err instanceof Error ? err.message : "Failed to load graph data");
    } finally {
      setGraphLoading(false);
    }
  };

  const handleNodeClick = async (entityName: string) => {
    setGraphViewIndex(null);
    setGraphElements(null);
    setQuery(entityName);
    setLoading(true);
    setError(null);
    try {
      const res = await queryGraph({ query: entityName, top_k: 20 });
      setResults(res.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card card-body">
        <form onSubmit={(e) => void handleSearch(e)}>
          <div className="field-row" style={{ alignItems: "flex-end" }}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="graph-query">Search entities</label>
              <input
                id="graph-query"
                type="search"
                placeholder="e.g. Patriot PAC-3"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? "Searching..." : "Search"}
            </button>
          </div>
        </form>
      </div>

      {error && <div className="alert alert-error mt-md">{error}</div>}

      {results !== null && (
        <div className="results mt-md">
          {results.length === 0 ? (
            <p className="text-muted">No entities found.</p>
          ) : (
            results.map((item, i) => {
              // Use entity_name from context (not content_text which may be chunk evidence)
              const ctx = item.context as Record<string, unknown> | undefined;
              const entityName = String(ctx?.entity_name || item.content_text || "");
              const entityType = String(ctx?.entity_type || "");
              const neighbors = ctx?.neighbors as Array<Record<string, unknown>> | undefined;
              const sources = item.sources as Array<Record<string, unknown>> | undefined;

              return (
              <div key={i} className="result-card">
                <div className="result-card-header">
                  <span className="text-xs text-muted">#{i + 1}</span>
                  <span className="badge badge-info">{entityType || item.modality}</span>
                  {entityName && (
                    <button
                      className={`btn btn-ghost btn-sm graph-toggle-btn${graphViewIndex === i ? " active" : ""}`}
                      onClick={() => void handleToggleGraph(i, entityName)}
                      title="Toggle graph view"
                    >
                      ◉
                    </button>
                  )}
                </div>
                {graphViewIndex === i ? (
                  graphLoading ? (
                    <div className="graph-view-container" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
                      <span className="spinner" />
                    </div>
                  ) : graphElements && graphElements.length > 0 ? (
                    <GraphView
                      elements={graphElements}
                      onNodeClick={(name) => void handleNodeClick(name)}
                      onClose={() => { setGraphViewIndex(null); setGraphElements(null); setGraphError(null); }}
                    />
                  ) : (
                    <div className="alert alert-error mt-sm">
                      {graphError || "Could not load graph data."}
                    </div>
                  )
                ) : (
                  <>
                    {/* Entity name and type */}
                    <div style={{ fontWeight: 600, fontSize: "1.05rem" }}>{entityName}</div>
                    {item.classification && item.classification !== "UNCLASSIFIED" && (
                      <span className="badge" style={{ marginLeft: "0.5rem" }}>{item.classification}</span>
                    )}

                    {/* Neighbors */}
                    {neighbors && neighbors.length > 0 && (
                      <div style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
                        <strong>Connections:</strong>{" "}
                        {neighbors.map((n, j) => (
                          <span key={j}>
                            <a href="#" onClick={(e) => { e.preventDefault(); void handleNodeClick(String(n.name)); }}
                               style={{ color: "var(--color-primary)" }}>
                              {String(n.name)}
                            </a>
                            {` (${String(n.entity_type)})`}
                            {j < neighbors.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Source documents */}
                    {sources && sources.length > 0 && (
                      <details style={{ marginTop: "0.5rem", fontSize: "0.85rem" }}>
                        <summary style={{ color: "var(--color-text-muted)", cursor: "pointer" }}>
                          Sources ({sources.length} document{sources.length !== 1 ? "s" : ""})
                        </summary>
                        <div style={{ marginTop: "0.25rem" }}>
                          {sources.map((src, j) => (
                            <div key={j} style={{ color: "var(--color-text-muted)", marginBottom: "0.15rem" }}>
                              {String(src.document_id).slice(0, 8)}...
                              {src.page_number != null && <> p.{String(src.page_number)}</>}
                              {!!src.chunk_text_preview && (
                                <span style={{ opacity: 0.7, marginLeft: "0.5rem" }}>
                                  {String(src.chunk_text_preview).slice(0, 100)}...
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </>
                )}
              </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
}

function EntityForm({ entityTypes }: { entityTypes: string[] }) {
  const [entityType, setEntityType] = useState(entityTypes[0] ?? DEFAULT_ENTITY_TYPES[0]);
  const [name, setName] = useState("");
  const [propsJson, setPropsJson] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!entityTypes.includes(entityType)) {
      setEntityType(entityTypes[0] ?? DEFAULT_ENTITY_TYPES[0]);
    }
  }, [entityType, entityTypes]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    setLoading(true);
    setError(null);
    setSuccess(null);

    let properties: Record<string, unknown> | undefined;
    if (propsJson.trim()) {
      try {
        properties = JSON.parse(propsJson);
      } catch {
        setError("Invalid JSON in properties field");
        setLoading(false);
        return;
      }
    }

    try {
      await ingestGraphEntity({ entity_type: entityType, name: name.trim(), properties });
      setSuccess(`Entity "${name}" (${entityType}) created.`);
      setName("");
      setPropsJson("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create entity");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card card-body">
      <form onSubmit={(e) => void handleSubmit(e)}>
        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field">
            <label htmlFor="entity-type">Entity type</label>
            <select id="entity-type" value={entityType} onChange={(e) => setEntityType(e.target.value)}>
              {entityTypes.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="entity-name">Name</label>
            <input
              id="entity-name"
              type="text"
              placeholder="e.g. Patriot PAC-3"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
        </div>
        <div className="field">
          <label htmlFor="entity-props">Properties (JSON, optional)</label>
          <textarea
            id="entity-props"
            rows={3}
            placeholder='{"designation": "MIM-104F"}'
            value={propsJson}
            onChange={(e) => setPropsJson(e.target.value)}
            style={{ fontFamily: "monospace", width: "100%" }}
          />
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading || !name.trim()}>
          {loading ? "Creating..." : "Create Entity"}
        </button>
      </form>
      {error && <div className="alert alert-error mt-sm">{error}</div>}
      {success && <div className="alert alert-success mt-sm">{success}</div>}
    </div>
  );
}

function RelationshipForm({
  entityTypes,
  relationshipTypes,
  validationMatrix,
}: {
  entityTypes: string[];
  relationshipTypes: string[];
  validationMatrix: ValidationRule[];
}) {
  const [fromEntity, setFromEntity] = useState("");
  const [fromType, setFromType] = useState(entityTypes[0] ?? DEFAULT_ENTITY_TYPES[0]);
  const [toEntity, setToEntity] = useState("");
  const [toType, setToType] = useState(entityTypes[1] ?? entityTypes[0] ?? DEFAULT_ENTITY_TYPES[0]);
  const [relType, setRelType] = useState(relationshipTypes[0] ?? DEFAULT_REL_TYPES[0]);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!entityTypes.includes(fromType)) {
      setFromType(entityTypes[0] ?? DEFAULT_ENTITY_TYPES[0]);
    }
    if (!entityTypes.includes(toType)) {
      setToType(entityTypes[1] ?? entityTypes[0] ?? DEFAULT_ENTITY_TYPES[0]);
    }
  }, [entityTypes, fromType, toType]);

  const filteredRelationshipTypes = React.useMemo(() => {
    if (validationMatrix.length === 0) {
      return relationshipTypes;
    }
    const matches = validationMatrix
      .filter((rule) => rule.source === fromType && rule.target === toType)
      .map((rule) => rule.relationship);
    return matches.length > 0 ? uniqueSorted(matches) : relationshipTypes;
  }, [fromType, relationshipTypes, toType, validationMatrix]);

  useEffect(() => {
    if (!filteredRelationshipTypes.includes(relType)) {
      setRelType(filteredRelationshipTypes[0] ?? relationshipTypes[0] ?? DEFAULT_REL_TYPES[0]);
    }
  }, [filteredRelationshipTypes, relType, relationshipTypes]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!fromEntity.trim() || !toEntity.trim()) return;
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      await ingestGraphRelationship({
        from_entity: fromEntity.trim(),
        from_type: fromType,
        to_entity: toEntity.trim(),
        to_type: toType,
        relationship_type: relType,
      });
      setSuccess(`Relationship ${fromEntity} -[${relType}]-> ${toEntity} created.`);
      setFromEntity("");
      setToEntity("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create relationship");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card card-body">
      <form onSubmit={(e) => void handleSubmit(e)}>
        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field" style={{ flex: 1 }}>
            <label>From entity</label>
            <input
              type="text"
              placeholder="Entity name"
              value={fromEntity}
              onChange={(e) => setFromEntity(e.target.value)}
            />
          </div>
          <div className="field">
            <label>From type</label>
            <select value={fromType} onChange={(e) => setFromType(e.target.value)}>
              {entityTypes.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <div className="field">
          <label>Relationship type</label>
          <select value={relType} onChange={(e) => setRelType(e.target.value)}>
            {filteredRelationshipTypes.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          {validationMatrix.length > 0 && (
            <div className="text-xs text-muted" style={{ marginTop: "0.25rem" }}>
              {filteredRelationshipTypes.length === relationshipTypes.length
                ? "Showing all ontology relationship types"
                : `Filtered by validation matrix for ${fromType} -> ${toType}`}
            </div>
          )}
        </div>

        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field" style={{ flex: 1 }}>
            <label>To entity</label>
            <input
              type="text"
              placeholder="Entity name"
              value={toEntity}
              onChange={(e) => setToEntity(e.target.value)}
            />
          </div>
          <div className="field">
            <label>To type</label>
            <select value={toType} onChange={(e) => setToType(e.target.value)}>
              {entityTypes.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="btn btn-primary mt-sm"
          disabled={loading || !fromEntity.trim() || !toEntity.trim()}
        >
          {loading ? "Creating..." : "Create Relationship"}
        </button>
      </form>
      {error && <div className="alert alert-error mt-sm">{error}</div>}
      {success && <div className="alert alert-success mt-sm">{success}</div>}
    </div>
  );
}


// ---------------------------------------------------------------------------
// Global Search Indexing Panel
// ---------------------------------------------------------------------------

function GlobalSearchIndexingPanel() {
  const [settings, setSettings] = useState<IndexingSettings | null>(null);
  const [triggering, setTriggering] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minutesRemaining, setMinutesRemaining] = useState<number | null>(null);
  const [reports, setReports] = useState<CommunityReport[]>([]);
  const [showReports, setShowReports] = useState(false);

  const computeRemaining = useCallback((s: IndexingSettings) => {
    if (!s.indexing_enabled || !s.last_indexing_at) {
      setMinutesRemaining(null);
      return;
    }
    const elapsed = (Date.now() - new Date(s.last_indexing_at).getTime()) / 60000;
    setMinutesRemaining(Math.max(0, Math.round(s.indexing_interval_minutes - elapsed)));
  }, []);

  useEffect(() => {
    getIndexingSettings().then((s) => {
      setSettings(s);
      computeRemaining(s);
    }).catch(() => {});
  }, [computeRemaining]);

  useEffect(() => {
    if (!settings) return;
    const id = setInterval(() => computeRemaining(settings), 60000);
    return () => clearInterval(id);
  }, [settings, computeRemaining]);

  const handleUpdate = async () => {
    setTriggering(true);
    setStatus(null);
    setError(null);
    try {
      const res = await triggerIndexing("incremental");
      setStatus(`Incremental update started (run ${res.run_id.slice(0, 8)}...)`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger update");
    } finally {
      setTriggering(false);
    }
  };

  const handleFullReindex = async () => {
    const confirmed = window.confirm(
      "This will regenerate all community reports from scratch. " +
      "Existing reports will be replaced. This can take several minutes " +
      "for large document collections.\n\nProceed?"
    );
    if (!confirmed) return;
    setTriggering(true);
    setStatus(null);
    setError(null);
    try {
      const res = await triggerIndexing("full");
      setStatus(`Full reindex started (run ${res.run_id.slice(0, 8)}...)`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to trigger full reindex");
    } finally {
      setTriggering(false);
    }
  };

  const handleShowReports = async () => {
    if (showReports) { setShowReports(false); return; }
    try {
      const data = await getCommunityReports();
      setReports(data.reports || []);
    } catch { setReports([]); }
    setShowReports(true);
  };

  if (!settings) {
    return <div className="empty-state"><span className="spinner" /> Loading indexing settings...</div>;
  }

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>Global Search Indexing</h3>
      <p style={{ color: "var(--color-text-muted)", fontSize: "0.85rem", marginBottom: "1rem" }}>
        Global Search requires indexed community reports. This panel lets you schedule and trigger
        the indexing workflow: community detection, report generation, and embedding.
      </p>

      <div style={{
        border: "1px solid var(--color-border)", borderRadius: "var(--radius)",
        padding: "1rem", marginBottom: "1rem", background: "var(--color-surface-2)",
        fontSize: "0.9rem", lineHeight: 1.8,
      }}>
        <div>
          Automatic indexing is{" "}
          <strong>{settings.indexing_enabled ? "enabled" : "disabled"}</strong>
          {settings.indexing_enabled && (
            <>, running every <strong>{settings.indexing_interval_minutes} minutes</strong></>
          )}
        </div>
        {settings.post_ingest_enabled && (
          <div style={{ color: "var(--color-text-muted)" }}>
            Auto-triggers after every <strong>{settings.post_ingest_threshold}</strong> documents ingested
          </div>
        )}
        {settings.indexing_enabled && (
          <div style={{ color: "var(--color-text-muted)" }}>
            {minutesRemaining !== null
              ? <>Next auto indexing in <strong>{minutesRemaining} minute{minutesRemaining !== 1 ? "s" : ""}</strong>
                  {settings.last_indexing_at && (
                    <> (at {new Date(
                      new Date(settings.last_indexing_at).getTime() + settings.indexing_interval_minutes * 60000
                    ).toLocaleTimeString()})</>
                  )}
                </>
              : "Pending first run"}
          </div>
        )}
        {settings.last_run && (
          <div style={{ color: "var(--color-text-muted)", fontSize: "0.8rem", marginTop: "0.25rem" }}>
            Last run: <strong>{settings.last_run.status}</strong>
            {settings.last_run.completed_at && (
              <> at {new Date(settings.last_run.completed_at).toLocaleString()}</>
            )}
            {settings.last_run.total_communities > 0 && (
              <> — {settings.last_run.total_communities} communities, {settings.last_run.reports_generated} generated, {settings.last_run.reports_reused} reused</>
            )}
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
        <button className="btn btn-primary" onClick={handleUpdate} disabled={triggering}>
          {triggering ? "Starting..." : "Update Index"}
        </button>
        <button className="btn btn-danger" onClick={handleFullReindex} disabled={triggering}>
          {triggering ? "Starting..." : "Force Full Reindex"}
        </button>
        <button className="btn" onClick={handleShowReports}>
          {showReports ? "Hide Reports" : "View Reports"}
        </button>
      </div>

      {status && <div className="alert alert-success mt-sm">{status}</div>}
      {error && <div className="alert alert-error mt-sm">{error}</div>}

      {showReports && (
        <div style={{ marginTop: "1rem" }}>
          <h4>Community Reports ({reports.length})</h4>
          {reports.length === 0 ? (
            <div className="empty-state">No reports generated yet. Run indexing first.</div>
          ) : (
            <div style={{ maxHeight: "400px", overflowY: "auto" }}>
              {reports.map((r) => (
                <div key={r.community_id} style={{
                  border: "1px solid var(--color-border)", borderRadius: "var(--radius)",
                  padding: "0.75rem", marginBottom: "0.5rem", background: "var(--color-surface)",
                }}>
                  <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>
                    Community {r.community_id}: {r.title}
                  </div>
                  <div style={{ fontSize: "0.85rem", color: "var(--color-text-muted)" }}>
                    {r.member_count} members
                    {r.generated_at && <> — generated {new Date(r.generated_at).toLocaleString()}</>}
                  </div>
                  <div style={{ fontSize: "0.85rem", marginTop: "0.25rem", whiteSpace: "pre-line" }}>
                    {r.summary.length > 300 ? r.summary.slice(0, 300) + "..." : r.summary}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
