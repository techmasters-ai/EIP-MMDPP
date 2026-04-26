import type { QueryProfileFieldEvidence } from "../api/client";

interface FieldEvidencePopoverProps {
  evidence: QueryProfileFieldEvidence[];
  fieldName?: string;
  onClose: () => void;
}

function buildDeepLink(e: QueryProfileFieldEvidence): string {
  if (!e.document_id || !e.element_uid) return "#";
  return `/documents/${e.document_id}?element_uid=${encodeURIComponent(e.element_uid)}`;
}

export function FieldEvidencePopover({ evidence, fieldName, onClose }: FieldEvidencePopoverProps) {
  return (
    <div
      className="field-evidence-popover"
      role="dialog"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="field-evidence-popover-header">
        {fieldName && <strong>{fieldName} — sources</strong>}
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={onClose}
          aria-label="Close"
        >
          ✕
        </button>
      </div>
      {evidence.length === 0 ? (
        <p className="empty">No per-field evidence available.</p>
      ) : (
        <ul className="field-evidence-list">
          {evidence.map((e, idx) => (
            <li key={idx} className={e.element_uid ? "" : "unverified"}>
              {!e.element_uid && (
                <span className="badge badge-warning">Unverified source</span>
              )}
              <blockquote>{e.supporting_snippet}</blockquote>
              <div className="meta text-xs text-muted">
                {e.document_name && <span>📄 {e.document_name} </span>}
                {e.page_number != null && <span>p. {e.page_number} </span>}
                {e.element_uid && e.document_id && (
                  <a href={buildDeepLink(e)} target="_blank" rel="noreferrer">
                    Open in document viewer
                  </a>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
