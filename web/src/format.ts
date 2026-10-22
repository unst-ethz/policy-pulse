/** Display helpers shared by views. */

/**
 * Resolution titles from the Digital Library end in catalogue boilerplate
 * (" : resolution / adopted by the General Assembly"); drop it for display.
 */
export function cleanTitle(title: string | null | undefined): string {
  if (!title) return ''
  // "Assem\w*": the catalogue has the odd typo ("Assemly").
  return title.replace(/\s*:\s*resolution\s*\/\s*adopted by the General Assem\w*\.?\s*$/i, '').trim()
}

/** Thesaurus labels come in capitals ("SOCIAL CONDITIONS AND EQUITY"); show them in sentence case. */
export function sentenceCase(label: string): string {
  if (label !== label.toUpperCase() || label === label.toLowerCase()) return label
  const lower = label.toLocaleLowerCase()
  return lower.charAt(0).toLocaleUpperCase() + lower.slice(1)
}

/** First and last year of an ISO date range, as "1990–2020" (or a single year). */
export function yearSpan(start?: string | null, end?: string | null): string {
  const a = start?.slice(0, 4)
  const b = end?.slice(0, 4)
  if (a && b) return a === b ? a : `${a}–${b}`
  return a ?? b ?? ''
}

/** Fixed colours for a comparison list, by position (never re-assigned when an entry is removed). */
export const SERIES_SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'] as const
export const SERIES_SLOTS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'] as const
