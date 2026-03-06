import React, { useEffect, useState } from "react";
import {
  proposeMemory,
  listMemoryProposals,
  approveMemory,
  rejectMemory,
  type MemoryProposal,
  unifiedQuery,
  type QueryResultItem,
} from "../api/client";

type Tab = "propose" | "proposals" | "search";
type StatusFilter = "all" | "proposed" | "approved" | "rejected";

export function MemoryPanel() {
  const [tab, setTab] = useState<Tab>("proposals");

  return (
    <div>
      <div className="tabs" style={{ marginBottom: "1rem" }}>
        {([
          ["proposals", "Proposals"],
          ["propose", "Propose Knowledge"],
          ["search", "Search Trusted Data"],
        ] as [Tab, string][]).map(([t, label]) => (
          <button
            key={t}
            className={`tab-btn${tab === t ? " active" : ""}`}
            onClick={() => setTab(t)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "proposals" && <ProposalsList />}
      {tab === "propose" && <ProposeForm />}
      {tab === "search" && <MemorySearch />}
    </div>
  );
}

function ProposeForm() {
  const [content, setContent] = useState("");
  const [confidence, setConfidence] = useState(0.8);
  const [contextJson, setContextJson] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;
    setLoading(true);
    setError(null);
    setSuccess(null);

    let source_context: Record<string, unknown> | undefined;
    if (contextJson.trim()) {
      try {
        source_context = JSON.parse(contextJson);
      } catch {
        setError("Invalid JSON in source context field");
        setLoading(false);
        return;
      }
    }

    try {
      const proposal = await proposeMemory({
        content: content.trim(),
        source_context,
        confidence,
      });
      setSuccess(`Proposal created (${proposal.id.slice(0, 8)}\u2026) — status: ${proposal.status}`);
      setContent("");
      setContextJson("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create proposal");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card card-body">
      <form onSubmit={(e) => void handleSubmit(e)}>
        <div className="field">
          <label htmlFor="memory-content">Knowledge content</label>
          <textarea
            id="memory-content"
            rows={4}
            placeholder="e.g. The SA-2 Guideline uses a Fan Song radar for target tracking and missile guidance."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field" style={{ width: "120px" }}>
            <label htmlFor="memory-confidence">Confidence</label>
            <input
              id="memory-confidence"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value) || 0.8)}
            />
          </div>
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="memory-context">Source context (JSON, optional)</label>
            <input
              id="memory-context"
              type="text"
              placeholder='{"document": "SA-2 Technical Manual", "page": 42}'
              value={contextJson}
              onChange={(e) => setContextJson(e.target.value)}
              style={{ fontFamily: "monospace" }}
            />
          </div>
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || !content.trim()}
        >
          {loading ? "Submitting..." : "Propose Knowledge"}
        </button>
      </form>
      {error && <div className="alert alert-error mt-sm">{error}</div>}
      {success && <div className="alert alert-success mt-sm">{success}</div>}
    </div>
  );
}

function ProposalsList() {
  const [proposals, setProposals] = useState<MemoryProposal[]>([]);
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const fetchProposals = async () => {
    setLoading(true);
    setError(null);
    try {
      const status = filter === "all" ? undefined : filter;
      const data = await listMemoryProposals(status);
      setProposals(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load proposals");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchProposals();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  const handleApprove = async (id: string) => {
    setActionLoading(id);
    try {
      await approveMemory(id);
      await fetchProposals();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to approve");
    } finally {
      setActionLoading(null);
    }
  };

  const handleReject = async (id: string) => {
    setActionLoading(id);
    try {
      await rejectMemory(id);
      await fetchProposals();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reject");
    } finally {
      setActionLoading(null);
    }
  };

  const STATUS_FILTERS: { value: StatusFilter; label: string }[] = [
    { value: "all", label: "All" },
    { value: "proposed", label: "Pending" },
    { value: "approved", label: "Approved" },
    { value: "rejected", label: "Rejected" },
  ];

  return (
    <div>
      <div className="card card-body" style={{ marginBottom: "1rem" }}>
        <div className="flex-center gap-sm">
          <span className="text-sm text-muted">Filter:</span>
          {STATUS_FILTERS.map((f) => (
            <button
              key={f.value}
              type="button"
              className={`mode-btn${filter === f.value ? " active" : ""}`}
              onClick={() => setFilter(f.value)}
            >
              {f.label}
            </button>
          ))}
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={() => void fetchProposals()}
            disabled={loading}
            style={{ marginLeft: "auto" }}
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error mt-sm">{error}</div>}

      {proposals.length === 0 && !loading ? (
        <div className="empty-state">
          <div className="empty-state-title">No proposals</div>
          <div className="text-muted text-sm">
            {filter === "all"
              ? "No trusted data proposals yet. Use the \"Propose Knowledge\" tab to add one."
              : `No ${filter} proposals found.`}
          </div>
        </div>
      ) : (
        proposals.map((p) => (
          <div key={p.id} className="result-card">
            <div className="result-card-header">
              <span className="text-xs text-muted">{p.id.slice(0, 8)}&hellip;</span>
              <span
                className={`badge ${
                  p.status === "approved"
                    ? "badge-success"
                    : p.status === "rejected"
                    ? "badge-error"
                    : "badge-info"
                }`}
              >
                {p.status}
              </span>
              <span className="text-xs text-muted">
                confidence: {(p.confidence * 100).toFixed(0)}%
              </span>
              <span className="text-xs text-muted">
                {new Date(p.created_at).toLocaleDateString()}
              </span>
            </div>
            <p style={{ margin: "0.5rem 0" }}>{p.content}</p>
            {p.source_context && (
              <pre
                className="text-xs"
                style={{ whiteSpace: "pre-wrap", margin: "0.25rem 0" }}
              >
                {JSON.stringify(p.source_context, null, 2)}
              </pre>
            )}
            {p.review_notes && (
              <div className="text-sm text-muted" style={{ marginTop: "0.25rem" }}>
                Review notes: {p.review_notes}
              </div>
            )}
            {p.status === "proposed" && (
              <div className="flex-center gap-sm" style={{ marginTop: "0.5rem" }}>
                <button
                  className="btn btn-sm btn-primary"
                  disabled={actionLoading === p.id}
                  onClick={() => void handleApprove(p.id)}
                >
                  {actionLoading === p.id ? "..." : "Approve"}
                </button>
                <button
                  className="btn btn-sm btn-ghost"
                  disabled={actionLoading === p.id}
                  onClick={() => void handleReject(p.id)}
                >
                  {actionLoading === p.id ? "..." : "Reject"}
                </button>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  );
}

function MemorySearch() {
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
      const res = await unifiedQuery({
        query_text: query.trim(),
        mode: "memory",
        top_k: 20,
      });
      setResults(res.results ?? []);
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
              <label htmlFor="memory-query">Search approved trusted data</label>
              <input
                id="memory-query"
                type="search"
                placeholder="e.g. SA-2 guidance radar"
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
            <p className="text-muted">No trusted data results found.</p>
          ) : (
            results.map((item, i) => (
              <div key={i} className="result-card">
                <div className="result-card-header">
                  <span className="text-xs text-muted">#{i + 1}</span>
                  <span className="badge badge-info">{item.modality}</span>
                </div>
                {item.content_text && <p>{item.content_text}</p>}
                {item.context && (
                  <pre
                    className="text-xs"
                    style={{ whiteSpace: "pre-wrap", margin: "0.5rem 0" }}
                  >
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
