/**
 * The resolution filter, kept in the URL so every view is shareable and the back button works.
 *
 * URL keys are short and human-readable (`from=1990&to=2020&subject=…&country=CHE&mode=voted&q=…`,
 * `sponsor=CHE&original=1`, `adopted=recorded&adopted=without_vote`);
 * `toApiParams` turns them into the API's filter parameters.
 */
import { computed } from 'vue'
import { type LocationQuery, useRoute, useRouter } from 'vue-router'

export type CountryMode = 'voted' | 'member' | 'none'
/** How a resolution was adopted, as the API names it. */
export type Adoption = 'recorded' | 'non_recorded' | 'without_vote'
export const ADOPTIONS: Adoption[] = ['recorded', 'non_recorded', 'without_vote']

export interface Filters {
  from: number | null
  to: number | null
  subjects: string[]
  country: string | null
  mode: CountryMode
  keyword: string
  sponsor: string | null
  /** Only resolutions `sponsor` was an original sponsor of. */
  original: boolean
  /** Any of these ways of adoption; empty = all. */
  adoption: Adoption[]
}

const MODES: CountryMode[] = ['voted', 'member', 'none']

function many(value: LocationQuery[string] | undefined): string[] {
  return (Array.isArray(value) ? value : value ? [value] : []).filter((v): v is string => !!v)
}

function one(value: LocationQuery[string] | undefined): string | null {
  const v = Array.isArray(value) ? value[0] : value
  return v ?? null
}

function year(value: LocationQuery[string] | undefined): number | null {
  const v = Number(one(value))
  return Number.isInteger(v) && v > 1900 && v < 2100 ? v : null
}

export function parseFilters(query: LocationQuery): Filters {
  const mode = one(query.mode) as CountryMode | null
  return {
    from: year(query.from),
    to: year(query.to),
    subjects: many(query.subject),
    country: one(query.country)?.toUpperCase() ?? null,
    mode: mode && MODES.includes(mode) ? mode : 'none',
    keyword: one(query.q) ?? '',
    sponsor: one(query.sponsor)?.toUpperCase() ?? null,
    original: one(query.original) === '1',
    adoption: many(query.adopted).filter((a): a is Adoption => ADOPTIONS.includes(a as Adoption)),
  }
}

export function serializeFilters(filters: Filters): Record<string, string | string[]> {
  const query: Record<string, string | string[]> = {}
  if (filters.from) query.from = String(filters.from)
  if (filters.to) query.to = String(filters.to)
  if (filters.subjects.length) query.subject = filters.subjects
  if (filters.country) query.country = filters.country
  if (filters.country && filters.mode !== 'none') query.mode = filters.mode
  if (filters.keyword.trim()) query.q = filters.keyword.trim()
  if (filters.sponsor) query.sponsor = filters.sponsor
  if (filters.sponsor && filters.original) query.original = '1'
  if (filters.adoption.length) query.adopted = filters.adoption
  return query
}

/** The API's shared filter parameters for `filters`. */
export function toApiParams(filters: Filters) {
  return {
    start_date: filters.from ? `${filters.from}-01-01` : undefined,
    end_date: filters.to ? `${filters.to}-12-31` : undefined,
    subject: filters.subjects.length ? filters.subjects : undefined,
    country: filters.country ?? undefined,
    country_mode: filters.country ? filters.mode : undefined,
    keyword: filters.keyword.trim() || undefined,
    sponsor: filters.sponsor ?? undefined,
    original_sponsor: filters.sponsor && filters.original ? true : undefined,
    adoption: filters.adoption.length ? filters.adoption : undefined,
  }
}

// shortcut: splits on every "&" and ignores "," (OR): an exact term containing either, or a
// term toggled into an OR expression, is handled wrongly. To fix with the backend's
// searchable_count, which builds the same expression.
const keywordParts = (keyword: string) =>
  keyword
    .split('&')
    .map((part) => part.trim())
    .filter(Boolean)

const normal = (term: string) => term.trim().toLocaleLowerCase()

/** The text of an exact ("quoted") part, or null; the search itself ignores case and spacing. */
function exact(part: string): string | null {
  return part.length > 1 && part.startsWith('"') && part.endsWith('"') ? part.slice(1, -1).trim() : null
}

/** The exact terms of a keyword expression, as typed. */
export function exactTerms(keyword: string): string[] {
  return keywordParts(keyword)
    .map(exact)
    .filter((term): term is string => !!term)
}

/** Whether the keyword expression holds `term` as an exact `&` term. */
export function hasTerm(keyword: string, term: string): boolean {
  return exactTerms(keyword).some((t) => normal(t) === normal(term))
}

/** The keyword expression with `term` added as an exact `&` term, or removed if already there. */
export function toggleTerm(keyword: string, term: string): string {
  const parts = keywordParts(keyword)
  const kept = parts.filter((part) => normal(exact(part) ?? '\0') !== normal(term))
  return (kept.length < parts.length ? kept : [...parts, `"${term}"`]).join(' & ')
}

export function countActive(filters: Filters): number {
  return (
    (filters.from || filters.to ? 1 : 0) +
    (filters.subjects.length ? 1 : 0) +
    (filters.country && filters.mode !== 'none' ? 1 : 0) +
    (filters.keyword.trim() ? 1 : 0) +
    (filters.sponsor ? 1 : 0) +
    (filters.adoption.length ? 1 : 0)
  )
}

export function useFilters() {
  const route = useRoute()
  const router = useRouter()

  const filters = computed(() => parseFilters(route.query))

  /** Merge a change into the URL. Other query keys (page, sort, view options) are kept. */
  function update(patch: Partial<Filters>, extra: Record<string, string | undefined> = {}) {
    const next = { ...filters.value, ...patch }
    const kept = Object.fromEntries(
      Object.entries(route.query).filter(
        ([key]) => !['from', 'to', 'subject', 'country', 'mode', 'q', 'sponsor', 'original', 'adopted', 'page'].includes(key),
      ),
    )
    const query: Record<string, unknown> = { ...kept, ...serializeFilters(next) }
    for (const [key, value] of Object.entries(extra)) {
      if (value === undefined) delete query[key]
      else query[key] = value
    }
    router.replace({ query: query as LocationQuery })
  }

  function clear() {
    update({ from: null, to: null, subjects: [], country: null, mode: 'none', keyword: '', sponsor: null, original: false, adoption: [] })
  }

  return { filters, update, clear }
}
