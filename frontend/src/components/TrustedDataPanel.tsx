import React, { useEffect, useState } from "react";
import {
  proposeTrustedData,
  listTrustedDataSubmissions,
  approveTrustedData,
  rejectTrustedData,
  reindexTrustedData,
  queryTrustedData,
  type TrustedDataSubmission,
  type TrustedDataQueryResult,
} from "../api/client";

type Tab = "propose" | "proposals" | "search";
type StatusFilter = "all" | "proposed" | "approved_pending_index" | "approved_indexed" | "index_failed" | "rejected";

export function TrustedDataPanel() {
  const [tab, setTab] = useState<Tab>("proposals");

  return (
    <div>
      <div className="tabs" style={{ marginBottom: "1rem" }}>
        {([
          ["proposals", "Submissions"],
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
      {tab === "search" && <TrustedDataSearch />}
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
      const submission = await proposeTrustedData({
        content: content.trim(),
        source_context,
        confidence,
      });
      setSuccess(`Submission created (${submission.id.slice(0, 8)}\u2026) \u2014 status: ${submission.status}`);
      setContent("");
      setContextJson("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create submission");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card card-body">
      <form onSubmit={(e) => void handleSubmit(e)}>
        <div className="field">
          <label htmlFor="td-content">Knowledge content</label>
          <textarea
            id="td-content"
            rows={4}
            placeholder="e.g. The SA-2 Guideline uses a Fan Song radar for target tracking and missile guidance."
            value={content}
            onChange={(e) => setContent(e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
        <div className="field-row" style={{ gap: "1rem" }}>
          <div className="field" style={{ width: "120px" }}>
            <label htmlFor="td-confidence">Confidence</label>
            <input
              id="td-confidence"
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value) || 0.8)}
            />
          </div>
          <div className="field" style={{ flex: 1 }}>
            <label htmlFor="td-context">Source context (JSON, optional)</label>
            <input
              id="td-context"
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

function statusBadgeClass(status: string): string {
  if (status === "APPROVED_INDEXED") return "badge-success";
  if (status === "REJECTED") return "badge-error";
  if (status === "INDEX_FAILED") return "badge-error";
  if (status === "APPROVED_PENDING_INDEX") return "badge-warning";
  return "badge-info";
}

function ProposalsList() {
  const [submissions, setSubmissions] = useState<TrustedDataSubmission[]>([]);
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  const fetchSubmissions = async () => {
    setLoading(true);
    setError(null);
    try {
      const status = filter === "all" ? undefined : filter;
      const data = await listTrustedDataSubmissions(status);
      setSubmissions(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load submissions");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchSubmissions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  const handleApprove = async (id: string) => {
    setActionLoading(id);
    try {
      await approveTrustedData(id);
      await fetchSubmissions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to approve");
    } finally {
      setActionLoading(null);
    }
  };

  const handleReject = async (id: string) => {
    setActionLoading(id);
    try {
      await rejectTrustedData(id);
      await fetchSubmissions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reject");
    } finally {
      setActionLoading(null);
    }
  };

  const handleReindex = async (id: string) => {
    setActionLoading(id);
    try {
      await reindexTrustedData(id);
      await fetchSubmissions();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reindex");
    } finally {
      setActionLoading(null);
    }
  };

  const STATUS_FILTERS: { value: StatusFilter; label: string }[] = [
    { value: "all", label: "All" },
    { value: "proposed", label: "Pending" },
    { value: "approved_pending_index", label: "Indexing" },
    { value: "approved_indexed", label: "Indexed" },
    { value: "index_failed", label: "Failed" },
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
            onClick={() => void fetchSubmissions()}
            disabled={loading}
            style={{ marginLeft: "auto" }}
          >
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
      </div>

      {error && <div className="alert alert-error mt-sm">{error}</div>}

      {submissions.length === 0 && !loading ? (
        <div className="empty-state">
          <div className="empty-state-title">No submissions</div>
          <div className="text-muted text-sm">
            {filter === "all"
              ? "No trusted data submissions yet. Use the \"Propose Knowledge\" tab to add one."
              : `No ${filter.replace(/_/g, " ")} submissions found.`}
          </div>
        </div>
      ) : (
        submissions.map((s) => (
          <div key={s.id} className="result-card">
            <div className="result-card-header">
              <span className="text-xs text-muted">{s.id.slice(0, 8)}&hellip;</span>
              <span className={`badge ${statusBadgeClass(s.status)}`}>
                {s.status}
              </span>
              {s.index_status && s.index_status !== "COMPLETE" && (
                <span className="badge badge-warning">{s.index_status}</span>
              )}
              <span className="text-xs text-muted">
                confidence: {(s.confidence * 100).toFixed(0)}%
              </span>
              <span className="text-xs text-muted">
                {new Date(s.created_at).toLocaleDateString()}
              </span>
            </div>
            <p style={{ margin: "0.5rem 0" }}>{s.content}</p>
            {s.source_context && (
              <pre
                className="text-xs"
                style={{ whiteSpace: "pre-wrap", margin: "0.25rem 0" }}
              >
                {JSON.stringify(s.source_context, null, 2)}
              </pre>
            )}
            {s.review_notes && (
              <div className="text-sm text-muted" style={{ marginTop: "0.25rem" }}>
                Review notes: {s.review_notes}
              </div>
            )}
            {s.index_error && (
              <div className="text-sm text-muted" style={{ marginTop: "0.25rem", color: "var(--error)" }}>
                Index error: {s.index_error}
              </div>
            )}
            <div className="flex-center gap-sm" style={{ marginTop: "0.5rem" }}>
              {s.status === "PROPOSED" && (
                <>
                  <button
                    className="btn btn-sm btn-primary"
                    disabled={actionLoading === s.id}
                    onClick={() => void handleApprove(s.id)}
                  >
                    {actionLoading === s.id ? "..." : "Approve"}
                  </button>
                  <button
                    className="btn btn-sm btn-ghost"
                    disabled={actionLoading === s.id}
                    onClick={() => void handleReject(s.id)}
                  >
                    {actionLoading === s.id ? "..." : "Reject"}
                  </button>
                </>
              )}
              {(s.status === "INDEX_FAILED" || s.status === "APPROVED_PENDING_INDEX") && (
                <button
                  className="btn btn-sm btn-primary"
                  disabled={actionLoading === s.id}
                  onClick={() => void handleReindex(s.id)}
                >
                  {actionLoading === s.id ? "..." : "Reindex"}
                </button>
              )}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function TrustedDataSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TrustedDataQueryResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await queryTrustedData({
        query: query.trim(),
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
              <label htmlFor="td-query">Search approved trusted data</label>
              <input
                id="td-query"
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
                  <span className="badge badge-info">
                    score: {item.score.toFixed(3)}
                  </span>
                  {item.confidence !== undefined && item.confidence !== null && (
                    <span className="text-xs text-muted">
                      confidence: {(item.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                </div>
                <p>{item.content_text}</p>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
