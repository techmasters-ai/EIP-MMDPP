import type { QueryProfileFieldGroup } from "../api/client";

interface FieldGroupTableProps {
  groups: QueryProfileFieldGroup[];
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return v.toLocaleString();
  return String(v);
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
                <tr key={f.name}>
                  <th title={f.description ?? undefined}>{f.label}</th>
                  <td>{formatValue(f.value)}</td>
                  <td className="evidence-cell">
                    {/* Phase 3 wires the chip; Phase 2 leaves empty */}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
    </div>
  );
}
