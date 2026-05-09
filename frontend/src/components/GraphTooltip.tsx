import { ProvenanceChips } from "./ProvenanceChips";

interface EdgeProvenance {
  evidence_ids?: string[];
  self_refs?: string[];
  page_numbers?: number[];
}

interface GraphTooltipProps {
  visible: boolean;
  x: number;
  y: number;
  data: Record<string, unknown>;
  containerRect?: DOMRect;
}

const HIDDEN_KEYS = new Set(["id", "label", "classes"]);
/** Keys that are rendered via dedicated provenance UI — suppress from the generic k/v dump. */
const PROVENANCE_KEYS = new Set(["provenance"]);

export function GraphTooltip({ visible, x, y, data, containerRect }: GraphTooltipProps) {
  if (!visible || !data || Object.keys(data).length === 0) return null;

  const provenance = data.provenance as EdgeProvenance | null | undefined;
  const evidenceIds = provenance?.evidence_ids ?? [];
  const selfRefs = provenance?.self_refs ?? [];
  const pageNumbers = provenance?.page_numbers ?? [];

  // Merge evidence_ids + self_refs (deduplicated)
  const allEvidenceIds = [...evidenceIds, ...selfRefs.filter((r) => !evidenceIds.includes(r))];

  const entries = Object.entries(data).filter(
    ([key, value]) =>
      !HIDDEN_KEYS.has(key) &&
      !PROVENANCE_KEYS.has(key) &&
      value !== null &&
      value !== undefined &&
      value !== "",
  );

  const hasContent = entries.length > 0 || allEvidenceIds.length > 0 || pageNumbers.length > 0;
  if (!hasContent) return null;

  let left = x + 12;
  let top = y + 12;
  if (containerRect) {
    // Convert from viewport coordinates to container-relative
    left = x - containerRect.left + 12;
    top = y - containerRect.top + 12;
    const tooltipWidth = 300;
    const tooltipHeight = entries.length * 28 + (allEvidenceIds.length > 0 ? 60 : 0) + 24;
    if (left + tooltipWidth > containerRect.width) left = x - containerRect.left - tooltipWidth - 12;
    if (top + tooltipHeight > containerRect.height) top = y - containerRect.top - tooltipHeight - 12;
    if (left < 0) left = 4;
    if (top < 0) top = 4;
  }

  return (
    <div className="graph-tooltip" style={{ position: "absolute", left, top, maxWidth: 320 }}>
      {entries.length > 0 && (
        <table>
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key}>
                <td className="graph-tooltip-key">{key}</td>
                <td className="graph-tooltip-value">
                  {typeof value === "object" ? JSON.stringify(value) : String(value)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {/* Edge provenance — evidence IDs + page numbers */}
      {allEvidenceIds.length > 0 && (
        <div style={{ padding: "0.25rem 0.5rem", borderTop: entries.length > 0 ? "1px solid var(--color-border, #ddd)" : undefined }}>
          <ProvenanceChips
            evidenceIds={allEvidenceIds}
            pageNumbers={pageNumbers.length > 0 ? pageNumbers : undefined}
            label="Evidence"
          />
        </div>
      )}
      {/* Show page numbers even when no evidence_ids present */}
      {allEvidenceIds.length === 0 && pageNumbers.length > 0 && (
        <div style={{ padding: "0.25rem 0.5rem", borderTop: entries.length > 0 ? "1px solid var(--color-border, #ddd)" : undefined, fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
          pp.&nbsp;{pageNumbers.join(", ")}
        </div>
      )}
    </div>
  );
}
