import { useState } from "react";

interface ProvenanceChipsProps {
  evidenceIds: string[];
  pageNumbers?: number[];
  /** Label prefix shown on the collapsible toggle. Defaults to "Evidence IDs". */
  label?: string;
}

/**
 * Collapsible list of Docling self_ref chips (e.g. "#/texts/12") with an optional
 * page-numbers badge. Used in QueryPage (chunk/entity hits) and GraphTooltip (edge
 * provenance).
 */
export function ProvenanceChips({ evidenceIds, pageNumbers, label = "Evidence IDs" }: ProvenanceChipsProps) {
  const [open, setOpen] = useState(false);

  if (!evidenceIds || evidenceIds.length === 0) return null;

  return (
    <div className="provenance-chips-root" style={{ marginTop: "0.35rem" }}>
      <button
        className="prov-toggle"
        onClick={() => setOpen((v) => !v)}
        style={{ fontSize: "0.8rem" }}
      >
        {open ? "▼" : "▶"} {label} ({evidenceIds.length})
        {pageNumbers && pageNumbers.length > 0 && (
          <span
            className="badge badge-info"
            style={{ marginLeft: "0.4rem", fontSize: "0.75rem", verticalAlign: "middle" }}
          >
            pp.&nbsp;{pageNumbers.join(", ")}
          </span>
        )}
      </button>

      {open && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem", marginTop: "0.3rem", paddingLeft: "1rem" }}>
          {evidenceIds.map((id) => (
            <code
              key={id}
              style={{
                fontSize: "0.75rem",
                padding: "0.1rem 0.35rem",
                background: "var(--color-surface-raised, #f0f0f0)",
                border: "1px solid var(--color-border, #d0d0d0)",
                borderRadius: "3px",
                color: "var(--color-text-muted, #666)",
                fontFamily: "monospace",
              }}
            >
              {id}
            </code>
          ))}
        </div>
      )}
    </div>
  );
}
