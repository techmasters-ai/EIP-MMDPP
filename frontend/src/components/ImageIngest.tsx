import React, { useState, useCallback } from "react";
import { ingestImage } from "../api/client";

export function ImageIngest() {
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [altText, setAltText] = useState("");
  const [classification, setClassification] = useState("UNCLASSIFIED");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setPreview(dataUrl);
      // Strip the data:image/...;base64, prefix
      const b64 = dataUrl.split(",")[1];
      setImageB64(b64);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleFile(file);
    },
    [handleFile],
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageB64) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await ingestImage({
        image: imageB64,
        alt_text: altText || undefined,
        classification,
      });
      setResult(res.chunk_id);
      setImageB64(null);
      setPreview(null);
      setAltText("");
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
          <div
            className="drop-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => document.getElementById("image-file-input")?.click()}
            style={{ cursor: "pointer", textAlign: "center", padding: "2rem" }}
          >
            {preview ? (
              <img
                src={preview}
                alt="Preview"
                style={{ maxWidth: "300px", maxHeight: "200px", borderRadius: "4px" }}
              />
            ) : (
              <div className="text-muted">
                Drop an image here or click to select (PNG, JPG, TIFF)
              </div>
            )}
            <input
              id="image-file-input"
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
          </div>

          <div className="field mt-md">
            <label htmlFor="alt-text">Alt text (optional)</label>
            <input
              id="alt-text"
              type="text"
              placeholder="Describe this image..."
              value={altText}
              onChange={(e) => setAltText(e.target.value)}
            />
          </div>

          <div className="field">
            <label htmlFor="img-classification">Classification</label>
            <select
              id="img-classification"
              value={classification}
              onChange={(e) => setClassification(e.target.value)}
            >
              <option value="UNCLASSIFIED">UNCLASSIFIED</option>
              <option value="CUI">CUI</option>
              <option value="CONFIDENTIAL">CONFIDENTIAL</option>
              <option value="SECRET">SECRET</option>
            </select>
          </div>

          <button
            type="submit"
            className="btn btn-primary mt-sm"
            disabled={loading || !imageB64}
          >
            {loading ? <><span className="spinner" /> Ingesting&hellip;</> : "Ingest Image"}
          </button>
        </form>
      </div>

      {error && <div className="alert alert-error mt-md">{error}</div>}

      {result && (
        <div className="alert alert-success mt-md">
          Image chunk created. ID: {result.slice(0, 8)}&hellip;
        </div>
      )}
    </div>
  );
}
