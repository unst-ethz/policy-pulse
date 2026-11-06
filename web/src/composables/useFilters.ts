/**
 * The resolution filter, kept in the URL so every view is shareable and the back button works.
 *
 * URL keys are short and human-readable (`from=1990&to=2020&subject=…&country=CHE&mode=voted&q=…`,
 * `sponsor=CHE&original=1`);
 * `toApiParams` turns them into the API's filter parameters.
 */
import { computed } from 'vue'
import { type LocationQuery, useRoute, useRouter } from 'vue-router'

export type CountryMode = 'voted' | 'member' | 'none'

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
}

const MODES: CountryMode[] = ['voted', 'member', 'none']

function one(value: LocationQuery[string] | undefined): string | null {
  const v = Array.isArray(value) ? value[0] : value
  return v ?? null
}

function year(value: LocationQuery[string] | undefined): number | null {
  const v = Number(one(value))
  return Number.isInteger(v) && v > 1900 && v < 2100 ? v : null
}

export function parseFilters(query: LocationQuery): Filters {
  const subjects = query.subject
  const mode = one(query.mode) as CountryMode | null
  return {
    from: year(query.from),
    to: year(query.to),
    subjects: (Array.isArray(subjects) ? subjects : subjects ? [subjects] : []).filter(
      (s): s is string => !!s,
    ),
    country: one(query.country)?.toUpperCase() ?? null,
    mode: mode && MODES.includes(mode) ? mode : 'none',
    keyword: one(query.q) ?? '',
    sponsor: one(query.sponsor)?.toUpperCase() ?? null,
    original: one(query.original) === '1',
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
  }
}

// shortcut: splits on every "&", so an exact term containing "&" is not recognised; fine for
// extracted keywords, which carry none.
const keywordParts = (keyword: string) =>
  keyword
    .split('&')
    .map((part) => part.trim())
    .filter(Boolean)

/** Whether the keyword expression holds `term` as an exact `&` term. */
export function hasTerm(keyword: string, term: string): boolean {
  return keywordParts(keyword).includes(`"${term}"`)
}

/** The keyword expression with `term` added as an exact `&` term, or removed if already there. */
export function toggleTerm(keyword: string, term: string): string {
  const exact = `"${term}"`
  const parts = keywordParts(keyword)
  return (parts.includes(exact) ? parts.filter((p) => p !== exact) : [...parts, exact]).join(' & ')
}

export function countActive(filters: Filters): number {
  return (
    (filters.from || filters.to ? 1 : 0) +
    (filters.subjects.length ? 1 : 0) +
    (filters.country && filters.mode !== 'none' ? 1 : 0) +
    (filters.keyword.trim() ? 1 : 0) +
    (filters.sponsor ? 1 : 0)
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
        ([key]) => !['from', 'to', 'subject', 'country', 'mode', 'q', 'sponsor', 'original', 'page'].includes(key),
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
    update({ from: null, to: null, subjects: [], country: null, mode: 'none', keyword: '', sponsor: null, original: false })
  }

  return { filters, update, clear }
}
