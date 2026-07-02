/** Shared ontology helper functions used by GraphExplorer and QueryProfilesPage. */

export function uniqueSorted(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
}
