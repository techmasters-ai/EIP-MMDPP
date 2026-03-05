import React, { useState } from "react";
import { unifiedQuery, type QueryMode, type QueryResultItem, type SectionResults } from "../api/client";

const MODES: { value: QueryMode; label: string; description: string }[] = [
  {
    value: "text_semantic",
    label: "Text",
    description: "BGE vector similarity search on text chunks",
  },
  {
    value: "image_semantic",
    label: "Images",
    description: "CLIP vector search on image chunks (text-to-image or image-to-image)",
  },
  {
    value: "graph",
    label: "Graph",
    description: "Entity + relationship traversal via Apache AGE",
  },
  {
    value: "cross_modal",
    label: "Cross-Modal",
    description: "Text-to-image or image-to-text via graph bridging",
  },
  {
    value: "memory",
    label: "Memory",
    description: "Search Cognee approved memory",
  },
];

function scoreColor(score: number): string {
  if (score >= 0.85) return "var(--color-success)";
  if (score >= 0.65) return "var(--color-primary)";
  return "var(--color-text-muted)";
}

function ResultCard({ item, index }: { item: QueryResultItem; index: number }) {
  const [expanded, setExpanded] = useState(false);

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
        if (relType && nName) relInfo = `${relType} -> ${nName}`;
      }
    }
  }

  const truncated = displayText && displayText.length > 400 && !expanded;

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="text-xs text-muted">#{index + 1}</span>
        <span className="result-score" style={{ color: scoreColor(item.score) }}>
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
            {truncated ? displayText.slice(0, 400) + "\u2026" : displayText}
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
        {item.chunk_id && <span>Chunk: {String(item.chunk_id).slice(0, 8)}\u2026</span>}
        {item.artifact_id && <span>Artifact: {String(item.artifact_id).slice(0, 8)}\u2026</span>}
      </div>
    </div>
  );
}

export function QueryPage() {
  const [queryText, setQueryText] = useState("");
  const [selectedModes, setSelectedModes] = useState<Set<QueryMode>>(new Set(["text_semantic"]));
  const [topK, setTopK] = useState(10);
  const [sections, setSections] = useState<Record<string, SectionResults> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);

  const toggleMode = (mode: QueryMode) => {
    setSelectedModes((prev) => {
      const next = new Set(prev);
      if (next.has(mode)) {
        if (next.size > 1) next.delete(mode); // keep at least one
      } else {
        next.add(mode);
      }
      return next;
    });
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = queryText.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setSections(null);
    setElapsed(null);
    const t0 = performance.now();

    try {
      const res = await unifiedQuery({
        query_text: q,
        modes: Array.from(selectedModes),
        top_k: topK,
        include_context: true,
      });
      setSections(res.sections);
      setElapsed(Math.round(performance.now() - t0));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(false);
    }
  };

  const totalResults = sections
    ? Object.values(sections).reduce((sum, s) => sum + s.total, 0)
    : 0;

  return (
    <div>
      <div className="card card-body">
        <form onSubmit={(e) => void handleQuery(e)}>
          <div className="field">
            <label>Query modes (select one or more)</label>
            <div className="mode-selector" style={{ marginBottom: "1rem" }}>
              {MODES.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  className={`mode-btn${selectedModes.has(m.value) ? " active" : ""}`}
                  title={m.description}
                  onClick={() => toggleMode(m.value)}
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
                max={100}
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
                {loading ? <><span className="spinner" /> Searching&hellip;</> : "Search"}
              </button>
            </div>
          </div>
        </form>
      </div>

      {error && <div className="alert alert-error" style={{ marginTop: "1rem" }}>{error}</div>}

      {sections !== null && (
        <div className="results">
          <div className="flex-center gap-sm" style={{ marginBottom: "0.5rem" }}>
            <span className="text-sm text-muted">
              {totalResults === 0
                ? "No results found."
                : `${totalResults} result${totalResults !== 1 ? "s" : ""} across ${Object.keys(sections).length} mode${Object.keys(sections).length !== 1 ? "s" : ""}`}
              {elapsed != null && ` \u2014 ${elapsed}ms`}
            </span>
          </div>

          {totalResults === 0 ? (
            <div className="empty-state">
              <div className="empty-state-title">No matching documents</div>
              <div className="text-muted text-sm">
                Try a different query or upload documents first.
              </div>
            </div>
          ) : (
            Object.entries(sections).map(([mode, section]) => (
              <div key={mode} style={{ marginBottom: "1.5rem" }}>
                <h3 style={{ marginBottom: "0.5rem", textTransform: "capitalize" }}>
                  {mode.replace(/_/g, " ")}
                  <span className="text-sm text-muted" style={{ marginLeft: "0.5rem" }}>
                    ({section.total} result{section.total !== 1 ? "s" : ""})
                  </span>
                </h3>
                {section.results.length === 0 ? (
                  <p className="text-sm text-muted">No results for this mode.</p>
                ) : (
                  section.results.map((item, i) => (
                    <ResultCard key={`${mode}-${i}`} item={item} index={i} />
                  ))
                )}
              </div>
            ))
          )}
        </div>
      )}

      <div className="api-info mt-md">
        <strong>LangGraph tool endpoint:</strong>{" "}
        <code>GET /v1/agent/context?query=&hellip;&amp;mode=text_semantic&amp;top_k=10</code>
        <br />
        Returns a pre-formatted markdown context string ready for LLM prompt injection.
      </div>
    </div>
  );
}
