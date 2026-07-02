/** Shared ontology helper functions used by GraphExplorer and QueryProfilesPage. */

export function uniqueSorted(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

export function normalizeNamedList(values: unknown): string[] {
  if (!Array.isArray(values)) return [];
  const names = values.flatMap((item) => {
    if (typeof item === "string") return [item.trim()];
    if (item && typeof item === "object") {
      const record = item as Record<string, unknown>;
      const candidate = record.name ?? record.id ?? record.value ?? record.label;
      return typeof candidate === "string" ? [candidate.trim()] : [];
    }
    return [];
  });
  return uniqueSorted(names);
}
