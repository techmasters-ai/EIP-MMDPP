import React, { useState } from "react";
import { ingestGraphEntity, ingestGraphRelationship, queryGraph, type QueryResultItem } from "../api/client";

const ENTITY_TYPES = [
  "EQUIPMENT_SYSTEM", "SUBSYSTEM", "COMPONENT", "ASSEMBLY",
  "SPECIFICATION", "CAPABILITY", "STANDARD", "DOCUMENT",
  "ORGANIZATION", "PROCEDURE", "FAILURE_MODE", "TEST_EVENT",
];

type Tab = "search" | "entity" | "relationship";

export function GraphExplorer() {
  const [tab, setTab] = useState<Tab>("search");

  return (
    <div>
      <div className="tabs" style={{ marginBottom: "1rem" }}>
        {(["search", "entity", "relationship"] as Tab[]).map((t) => (
          <button
            key={t}
            className={`tab-btn${tab === t ? " active" : ""}`}
            onClick={() => setTab(t)}
          >
            {t === "search" ? "Search" : t === "entity" ? "Add Entity" : "Add Relationship"}
          </button>
        ))}
      </div>

      {tab === "search" && <GraphSearch />}
      {tab === "entity" && <EntityForm />}
      {tab === "relationship" && <RelationshipForm />}
    </div>
  );
}

function GraphSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<QueryResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
            results.map((item, i) => (
              <div key={i} className="result-card">
                <div className="result-card-header">
                  <span className="text-xs text-muted">#{i + 1}</span>
                  <span className="badge badge-info">{item.modality}</span>
                </div>
                {item.content_text && <p>{item.content_text}</p>}
                {item.context && (
                  <pre className="text-xs" style={{ whiteSpace: "pre-wrap", margin: "0.5rem 0" }}>
                    {JSON.stringify(item.context, null, 2)}
                  </pre>
                )}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

function EntityForm() {
  const [entityType, setEntityType] = useState("EQUIPMENT_SYSTEM");
  const [name, setName] = useState("");
  const [propsJson, setPropsJson] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
              {ENTITY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
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

function RelationshipForm() {
  const [fromEntity, setFromEntity] = useState("");
  const [fromType, setFromType] = useState("EQUIPMENT_SYSTEM");
  const [toEntity, setToEntity] = useState("");
  const [toType, setToType] = useState("COMPONENT");
  const [relType, setRelType] = useState("CONTAINS");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const REL_TYPES = [
    "IS_SUBSYSTEM_OF", "CONTAINS", "IMPLEMENTS", "MEETS_STANDARD",
    "SPECIFIED_BY", "DESCRIBED_IN", "PERFORMED_BY", "AFFECTS",
    "SUPERSEDES", "TESTED_IN",
  ];

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
              {ENTITY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        <div className="field">
          <label>Relationship type</label>
          <select value={relType} onChange={(e) => setRelType(e.target.value)}>
            {REL_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
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
              {ENTITY_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
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
