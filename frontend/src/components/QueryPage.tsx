import React, { useState } from "react";
import { query, type QueryMode, type QueryResultItem } from "../api/client";

const MODES: { value: QueryMode; label: string; description: string }[] = [
  {
    value: "semantic",
    label: "Semantic",
    description: "Vector similarity search (BGE-large embeddings)",
  },
  {
    value: "graph",
    label: "Graph",
    description: "Entity + relationship traversal via Apache AGE",
  },
  {
    value: "hybrid",
    label: "Hybrid",
    description: "Semantic results boosted by graph entity co-occurrence",
  },
  {
    value: "cross_modal",
    label: "Cross-modal",
    description: "CLIP text→image retrieval",
  },
  {
    value: "cognee_graph",
    label: "Cognee Graph",
    description: "LLM-enhanced knowledge graph reasoning via Cognee",
  },
];

function scoreColor(score: number): string {
  if (score >= 0.85) return "var(--color-success)";
  if (score >= 0.65) return "var(--color-primary)";
  return "var(--color-text-muted)";
}

function ResultCard({ item, index }: { item: QueryResultItem; index: number }) {
  const [expanded, setExpanded] = useState(false);

  // For graph_node items, derive display text from context
  let displayText = item.content_text;
  let entityName = "";
  let relInfo = "";

  if (!displayText && item.context) {
    const entity = item.context["entity"] as Record<string, unknown> | undefined;
    if (entity) {
      const props = entity["properties"] as Record<string, unknown> | undefined;
      entityName = String(props?.["name"] ?? entity["name"] ?? "");
      const relType = item.context["rel_type"] as string | undefined;
      const neighbor = item.context["neighbor"] as Record<string, unknown> | undefined;
      if (neighbor) {
        const nProps = neighbor["properties"] as Record<string, unknown> | undefined;
        const nName = String(nProps?.["name"] ?? neighbor["name"] ?? "");
        if (relType && nName) relInfo = `${relType} → ${nName}`;
      }
    }
  }

  const truncated = displayText && displayText.length > 400 && !expanded;

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="text-xs text-muted">#{index + 1}</span>
        <span
          className="result-score"
          style={{ color: scoreColor(item.score) }}
        >
          {(item.score * 100).toFixed(0)}%
        </span>
        <span className="badge badge-info">{item.modality}</span>
        {item.classification && item.classification !== "UNCLASSIFIED" && (
          <span className="badge badge-error">{item.classification}</span>
        )}
        {item.page_number != null && (
          <span className="text-xs text-muted">p.{item.page_number}</span>
        )}
      </div>

      {entityName && (
        <div className="text-bold" style={{ marginBottom: "0.25rem" }}>
          {entityName}
        </div>
      )}
      {relInfo && (
        <div className="text-sm text-muted" style={{ marginBottom: "0.25rem" }}>
          {relInfo}
        </div>
      )}

      {displayText && (
        <>
          <p className="result-text">
            {truncated ? displayText.slice(0, 400) + "…" : displayText}
          </p>
          {displayText.length > 400 && (
            <button
              className="btn btn-ghost btn-sm mt-sm"
              onClick={() => setExpanded((v) => !v)}
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          )}
        </>
      )}

      <div className="result-meta">
        {item.chunk_id && <span>Chunk: {String(item.chunk_id).slice(0, 8)}…</span>}
        {item.artifact_id && <span>Artifact: {String(item.artifact_id).slice(0, 8)}…</span>}
      </div>
    </div>
  );
}

export function QueryPage() {
  const [queryText, setQueryText] = useState("");
  const [mode, setMode] = useState<QueryMode>("hybrid");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<QueryResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = queryText.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setResults(null);
    setElapsed(null);
    const t0 = performance.now();

    try {
      const res = await query({ query: q, mode, top_k: topK, include_context: true });
      setResults(res.results);
      setElapsed(Math.round(performance.now() - t0));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card card-body">
        <form onSubmit={(e) => void handleQuery(e)}>
          <div className="field">
            <label>Query mode</label>
            <div className="mode-selector" style={{ marginBottom: "1rem" }}>
              {MODES.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  className={`mode-btn${mode === m.value ? " active" : ""}`}
                  title={m.description}
                  onClick={() => setMode(m.value)}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <div className="field-row" style={{ alignItems: "flex-end" }}>
            <div className="field" style={{ flex: 1 }}>
              <label htmlFor="query-input">Search query</label>
              <input
                id="query-input"
                type="search"
                placeholder="e.g. Patriot PAC-3 guidance computer specifications"
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                autoFocus
              />
            </div>
            <div className="field" style={{ width: "80px", flexShrink: 0 }}>
              <label htmlFor="top-k">Top K</label>
              <input
                id="top-k"
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value, 10) || 10)}
              />
            </div>
            <div style={{ paddingBottom: "0" }}>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={loading || !queryText.trim()}
              >
                {loading ? <><span className="spinner" /> Searching…</> : "Search"}
              </button>
            </div>
          </div>
        </form>
      </div>

      {error && <div className="alert alert-error" style={{ marginTop: "1rem" }}>{error}</div>}

      {results !== null && (
        <div className="results">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
            <span className="text-sm text-muted">
              {results.length === 0
                ? "No results found."
                : `${results.length} result${results.length !== 1 ? "s" : ""}`}
              {elapsed != null && ` — ${elapsed}ms`}
            </span>
          </div>

          {results.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🔍</div>
              <div className="empty-state-title">No matching documents</div>
              <div className="text-muted text-sm">
                Try a different query or upload documents first.
              </div>
            </div>
          ) : (
            results.map((item, i) => (
              <ResultCard key={i} item={item} index={i} />
            ))
          )}
        </div>
      )}

      <div className="api-info mt-md">
        <strong>LangGraph tool endpoint:</strong>{" "}
        <code>GET /v1/agent/context?query=…&amp;mode=hybrid&amp;top_k=10</code>
        <br />
        Returns a pre-formatted markdown context string ready for LLM prompt injection.
      </div>
    </div>
  );
}
