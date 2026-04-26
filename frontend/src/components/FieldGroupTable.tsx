import { useState } from "react";
import type {
  QueryProfileFieldEntry,
  QueryProfileFieldEvidence,
  QueryProfileFieldGroup,
} from "../api/client";
import { FieldEvidencePopover } from "./FieldEvidencePopover";

interface FieldGroupTableProps {
  groups: QueryProfileFieldGroup[];
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return v.toLocaleString();
  return String(v);
}

function EvidenceChip({
  evidence,
  fieldName,
}: {
  evidence: QueryProfileFieldEvidence[];
  fieldName: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className="evidence-chip btn btn-ghost btn-sm"
        onClick={() => setOpen(true)}
        title={`Show ${evidence.length} source${evidence.length !== 1 ? "s" : ""} for ${fieldName}`}
      >
        📄 {evidence.length}
      </button>
      {open && (
        <FieldEvidencePopover
          evidence={evidence}
          fieldName={fieldName}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}

function FieldRow({ f }: { f: QueryProfileFieldEntry }) {
  return (
    <tr>
      <th title={f.description ?? undefined}>{f.label}</th>
      <td>{formatValue(f.value)}</td>
      <td className="evidence-cell">
        {f.evidence.length > 0 && (
          <EvidenceChip evidence={f.evidence} fieldName={f.label} />
        )}
      </td>
    </tr>
  );
}

export function FieldGroupTable({ groups }: FieldGroupTableProps) {
  if (groups.length === 0) {
    return <p className="empty">No properties extracted for this section.</p>;
  }
  return (
    <div className="field-group-table">
      {groups.map((g, idx) => (
        <details key={g.subgroup ?? `group-${idx}`} open={idx === 0}>
          <summary>{g.subgroup_label ?? "Other"}</summary>
          <table>
            <tbody>
              {g.fields.map((f) => (
                <FieldRow key={f.name} f={f} />
              ))}
            </tbody>
          </table>
        </details>
      ))}
    </div>
  );
}
