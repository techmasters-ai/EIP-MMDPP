import React, { useCallback, useEffect, useState } from "react";
import {
  createWatchDir,
  deleteWatchDir,
  listSources,
  listWatchDirs,
  type Source,
  type WatchDir,
} from "../api/client";

function formatPatterns(patterns: string[]): string {
  return patterns.join(", ");
}

export function DirectoryMonitor() {
  const [watchDirs, setWatchDirs] = useState<WatchDir[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  // Form state
  const [formPath, setFormPath] = useState("");
  const [formSourceId, setFormSourceId] = useState("");
  const [formInterval, setFormInterval] = useState("30");
  const [formPatterns, setFormPatterns] = useState("*.pdf,*.docx,*.doc,*.pptx,*.ppt,*.xlsx,*.xls,*.html,*.htm,*.md,*.csv,*.txt,*.png,*.jpg,*.jpeg,*.tiff,*.tif,*.bmp,*.gif,*.webp");
  const [submitting, setSubmitting] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [dirs, srcs] = await Promise.all([listWatchDirs(), listSources()]);
      setWatchDirs(dirs);
      setSources(srcs);
      if (srcs.length > 0 && !formSourceId) {
        setFormSourceId(srcs[0].id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load watch directories");
    } finally {
      setLoading(false);
    }
  }, [formSourceId]);

  useEffect(() => {
    void refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    const path = formPath.trim();
    if (!path) { setError("Directory path is required."); return; }
    if (!formSourceId) { setError("Select a source to attach new documents to."); return; }

    setSubmitting(true);
    try {
      const dir = await createWatchDir({
        source_id: formSourceId,
        path,
        poll_interval_seconds: parseInt(formInterval, 10) || 30,
        file_patterns: formPatterns.split(",").map((p) => p.trim()).filter(Boolean),
      });
      setWatchDirs((prev) => [...prev, dir]);
      setFormPath("");
      setSuccess(`Watch directory "${path}" added.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to add watch directory");
    } finally {
      setSubmitting(false);
    }
  };

  const handleDelete = async (id: string, path: string) => {
    setDeleting(id);
    setError(null);
    setSuccess(null);
    try {
      await deleteWatchDir(id);
      setWatchDirs((prev) => prev.filter((d) => d.id !== id));
      setSuccess(`Watch directory "${path}" removed.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete watch directory");
    } finally {
      setDeleting(null);
    }
  };

  const sourceName = (id: string) =>
    sources.find((s) => s.id === id)?.name ?? id.slice(0, 8) + "…";

  return (
    <div>
      {error && <div className="alert alert-error">{error}</div>}
      {success && <div className="alert alert-success">{success}</div>}

      {/* Add form */}
      <div className="card card-body" style={{ marginBottom: "1.25rem" }}>
        <h3 className="text-bold" style={{ marginBottom: "1rem", fontSize: "0.95rem" }}>
          Add a monitored directory
        </h3>
        <form onSubmit={(e) => void handleAdd(e)}>
          <div className="field-row">
            <div className="field" style={{ flex: 2 }}>
              <label htmlFor="watch-path">Directory path</label>
              <input
                id="watch-path"
                type="text"
                placeholder="/data/incoming/documents"
                value={formPath}
                onChange={(e) => setFormPath(e.target.value)}
                required
              />
            </div>
            <div className="field">
              <label htmlFor="watch-source">Attach to source</label>
              <select
                id="watch-source"
                value={formSourceId}
                onChange={(e) => setFormSourceId(e.target.value)}
              >
                <option value="">— select —</option>
                {sources.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="field-row">
            <div className="field">
              <label htmlFor="watch-interval">Poll interval (seconds)</label>
              <input
                id="watch-interval"
                type="number"
                min={5}
                max={3600}
                value={formInterval}
                onChange={(e) => setFormInterval(e.target.value)}
              />
            </div>
            <div className="field" style={{ flex: 2 }}>
              <label htmlFor="watch-patterns">File patterns (comma-separated)</label>
              <input
                id="watch-patterns"
                type="text"
                value={formPatterns}
                onChange={(e) => setFormPatterns(e.target.value)}
              />
            </div>
          </div>

          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? <><span className="spinner" /> Adding…</> : "Add directory"}
          </button>
        </form>
      </div>

      {/* Directory list */}
      <div className="card">
        {loading ? (
          <div className="empty-state">
            <span className="spinner" />
          </div>
        ) : watchDirs.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📂</div>
            <div className="empty-state-title">No directories monitored</div>
            <div className="text-muted text-sm">
              Add a directory path above. The Beat worker will poll it every N seconds and
              automatically ingest new files.
            </div>
          </div>
        ) : (
          <table className="watch-table">
            <thead>
              <tr>
                <th>Path</th>
                <th>Source</th>
                <th>Interval</th>
                <th>Patterns</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {watchDirs.map((dir) => (
                <tr key={dir.id}>
                  <td>
                    <code style={{ fontFamily: "var(--font-mono)", fontSize: "0.82rem" }}>
                      {dir.path}
                    </code>
                  </td>
                  <td className="text-sm">{sourceName(dir.source_id)}</td>
                  <td className="text-sm">{dir.poll_interval_seconds}s</td>
                  <td className="text-xs text-muted">{formatPatterns(dir.file_patterns)}</td>
                  <td>
                    <span className={`badge ${dir.enabled ? "badge-complete" : "badge-pending"}`}>
                      {dir.enabled ? "Active" : "Disabled"}
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn btn-danger btn-sm"
                      disabled={deleting === dir.id}
                      onClick={() => void handleDelete(dir.id, dir.path)}
                    >
                      {deleting === dir.id ? <span className="spinner" /> : "Remove"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="api-info mt-md">
        Auto-ingest uses Celery Beat polling. Configured interval sets the maximum detection
        latency. Duplicate files (same SHA-256) are silently skipped.
      </div>
    </div>
  );
}
