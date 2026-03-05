import React, { useState } from "react";
import { ingestText } from "../api/client";

export function TextIngest() {
  const [text, setText] = useState("");
  const [modality, setModality] = useState("text");
  const [classification, setClassification] = useState("UNCLASSIFIED");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ chunk_ids: string[]; chunks_created: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await ingestText({ text: text.trim(), modality, classification });
      setResult(res);
      setText("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ingest failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card card-body">
        <form onSubmit={(e) => void handleSubmit(e)}>
          <div className="field">
            <label htmlFor="text-input">Text content</label>
            <textarea
              id="text-input"
              rows={8}
              placeholder="Paste text content to embed and store..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              style={{ width: "100%", fontFamily: "inherit", resize: "vertical" }}
            />
          </div>

          <div className="field-row" style={{ gap: "1rem" }}>
            <div className="field">
              <label htmlFor="modality-select">Modality</label>
              <select
                id="modality-select"
                value={modality}
                onChange={(e) => setModality(e.target.value)}
              >
                <option value="text">Text</option>
                <option value="table">Table</option>
                <option value="schematic">Schematic</option>
              </select>
            </div>
            <div className="field">
              <label htmlFor="classification-select">Classification</label>
              <select
                id="classification-select"
                value={classification}
                onChange={(e) => setClassification(e.target.value)}
              >
                <option value="UNCLASSIFIED">UNCLASSIFIED</option>
                <option value="CUI">CUI</option>
                <option value="CONFIDENTIAL">CONFIDENTIAL</option>
                <option value="SECRET">SECRET</option>
              </select>
            </div>
          </div>

          <button
            type="submit"
            className="btn btn-primary mt-sm"
            disabled={loading || !text.trim()}
          >
            {loading ? <><span className="spinner" /> Ingesting&hellip;</> : "Ingest Text"}
          </button>
        </form>
      </div>

      {error && <div className="alert alert-error mt-md">{error}</div>}

      {result && (
        <div className="alert alert-success mt-md">
          Created {result.chunks_created} chunk{result.chunks_created !== 1 ? "s" : ""}.
          IDs: {result.chunk_ids.map((id) => id.slice(0, 8)).join(", ")}
        </div>
      )}
    </div>
  );
}
