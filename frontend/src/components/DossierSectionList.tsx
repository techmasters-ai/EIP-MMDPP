import type {
  GraphProfileEntityResult,
  QueryProfileDossierSection,
} from "../api/client";
import { FieldGroupTable } from "./FieldGroupTable";

interface DossierSectionListProps {
  sections: QueryProfileDossierSection[];
}

function RelatedSystemsChips({ systems }: { systems: GraphProfileEntityResult[] }) {
  return (
    <div className="related-systems">
      <span>Related: </span>
      {systems.map((s) => (
        <span key={s.node_id ?? s.name} className="chip">
          {s.entity_type} / {s.name}
        </span>
      ))}
    </div>
  );
}

export function DossierSectionList({ sections }: DossierSectionListProps) {
  return (
    <div className="dossier-section-list">
      {sections.map((s) => (
        <section key={s.profile_id} className="dossier-section">
          <h3>{s.profile_label}</h3>
          {s.kind === "section_properties" ? (
            s.field_groups.length > 0 ? (
              <FieldGroupTable groups={s.field_groups} />
            ) : (
              <p className="empty">No data extracted for this section.</p>
            )
          ) : (
            <p>(legacy section: {s.items.length} items)</p>
          )}
          {s.related_systems.length > 0 && (
            <RelatedSystemsChips systems={s.related_systems} />
          )}
        </section>
      ))}
    </div>
  );
}
