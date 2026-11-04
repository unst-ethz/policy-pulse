/**
 * One composable per API resource, on TanStack Query.
 *
 * Arguments are getters so the query key — and therefore the request — follows reactive state
 * (filters in the URL, the selected language). `placeholderData: keepPreviousData` keeps the
 * previous result on screen, dimmed, while the next one loads, instead of flashing a skeleton.
 */
import { keepPreviousData, useQuery } from '@tanstack/vue-query'
import { computed, type MaybeRefOrGetter, toValue } from 'vue'
import { useI18n } from 'vue-i18n'
import type { Filters } from '@/composables/useFilters'
import { toApiParams } from '@/composables/useFilters'
import { api, unwrap, type Vote } from './client'

export type Lang = 'en' | 'fr' | 'es' | 'ar' | 'zh' | 'ru'

function useLang() {
  const { locale } = useI18n({ useScope: 'global' })
  return computed(() => locale.value as Lang)
}

const STATIC = { staleTime: Infinity } as const

/** A reactive view of an argument, for use inside query keys. */
const r = <T>(value: MaybeRefOrGetter<T>) => computed(() => toValue(value))

export function useMeta() {
  return useQuery({
    queryKey: ['meta'],
    queryFn: () => unwrap(api.GET('/api/v1/meta')),
    ...STATIC,
  })
}

export function useCountries() {
  const lang = useLang()
  return useQuery({
    queryKey: ['countries', lang],
    queryFn: () => unwrap(api.GET('/api/v1/countries', { params: { query: { lang: lang.value } } })),
    ...STATIC,
  })
}

/** ISO3 -> country, for labelling codes everywhere. */
export function useCountryIndex() {
  const countries = useCountries()
  const index = computed(() => new Map((countries.data.value ?? []).map((c) => [c.iso3, c])))
  const name = (iso3: string) => index.value.get(iso3)?.name ?? iso3
  return { index, name, isLoading: countries.isLoading }
}

/**
 * `membersOn(allVotes, date)`: the votes of states that were UN members on `date`. Vote
 * columns exist for every code in the data, so a 2026 resolution has a "did not vote" for the
 * Soviet Union; those are not abstentions from the vote and should not be counted as such.
 */
export function useMembersOn() {
  const { index } = useCountryIndex()
  return (votes: Record<string, Vote>, date: string | null | undefined): Record<string, Vote> => {
    if (!date) return votes
    return Object.fromEntries(
      Object.entries(votes).filter(([iso3, vote]) => {
        if (vote !== 'X') return true
        const periods = index.value.get(iso3)?.membership ?? []
        return periods.some((p) => p.start <= date && (p.end == null || date <= p.end))
      }),
    )
  }
}

export function useSubjectTree() {
  const lang = useLang()
  return useQuery({
    queryKey: ['subjects', lang],
    queryFn: () =>
      unwrap(api.GET('/api/v1/subjects/tree', { params: { query: { lang: lang.value } } })),
    ...STATIC,
  })
}

export function useStats() {
  return useQuery({
    queryKey: ['stats'],
    queryFn: () => unwrap(api.GET('/api/v1/stats/overview')),
    ...STATIC,
  })
}

export function useRecentResolutions(limit: MaybeRefOrGetter<number>) {
  return useQuery({
    queryKey: ['recent', r(limit)],
    queryFn: () =>
      unwrap(api.GET('/api/v1/resolutions/recent', { params: { query: { limit: toValue(limit) } } })),
    placeholderData: keepPreviousData,
  })
}

export interface ListOptions {
  sort: 'date_desc' | 'date_asc' | 'consensus_desc' | 'consensus_asc'
  offset: number
  limit: number
  votes: string[]
}

export function useResolutions(
  filters: MaybeRefOrGetter<Filters>,
  options: MaybeRefOrGetter<ListOptions>,
) {
  return useQuery({
    queryKey: ['resolutions', r(filters), r(options)],
    queryFn: () => {
      const { sort, offset, limit, votes } = toValue(options)
      return unwrap(
        api.GET('/api/v1/resolutions', {
          params: {
            query: {
              ...toApiParams(toValue(filters)),
              sort,
              offset,
              limit,
              votes: votes.length ? votes : undefined,
            },
          },
        }),
      )
    },
    placeholderData: keepPreviousData,
  })
}

export function useResolution(undlId: MaybeRefOrGetter<string | null>) {
  const lang = useLang()
  return useQuery({
    queryKey: ['resolution', r(undlId), lang],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/resolutions/{undl_id}', {
          params: { path: { undl_id: toValue(undlId)! }, query: { lang: lang.value } },
        }),
      ),
    enabled: computed(() => !!toValue(undlId)),
  })
}

export function useAgreementMap(filters: MaybeRefOrGetter<Filters>, country: MaybeRefOrGetter<string>) {
  return useQuery({
    queryKey: ['agreement-map', r(filters), r(country)],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/agreement/map', {
          params: { query: { ...toApiParams(toValue(filters)), country: toValue(country) } },
        }),
      ),
    placeholderData: keepPreviousData,
  })
}

export function useTimeline(
  country: MaybeRefOrGetter<string>,
  compare: MaybeRefOrGetter<string[]>,
  specialSessions: MaybeRefOrGetter<boolean> = false,
) {
  return useQuery({
    queryKey: ['timeline', r(country), r(compare), r(specialSessions)],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/agreement/timeline', {
          params: {
            query: {
              country: toValue(country),
              compare: toValue(compare),
              special_sessions: toValue(specialSessions),
            },
          },
        }),
      ),
    enabled: computed(() => toValue(compare).length > 0),
    placeholderData: keepPreviousData,
  })
}

export function useAgreementBySubject(
  country: MaybeRefOrGetter<string>,
  compare: MaybeRefOrGetter<string | null>,
  range: MaybeRefOrGetter<{ from: number | null; to: number | null }>,
) {
  const lang = useLang()
  return useQuery({
    queryKey: ['by-subject', r(country), r(compare), r(range), lang],
    queryFn: () => {
      const { from, to } = toValue(range)
      return unwrap(
        api.GET('/api/v1/agreement/by-subject', {
          params: {
            query: {
              country: toValue(country),
              compare: toValue(compare)!,
              start_date: from ? `${from}-01-01` : undefined,
              end_date: to ? `${to}-12-31` : undefined,
              lang: lang.value,
            },
          },
        }),
      )
    },
    enabled: computed(() => !!toValue(compare)),
    placeholderData: keepPreviousData,
  })
}

export function useMultilateral(filters: MaybeRefOrGetter<Filters>, minVotes: MaybeRefOrGetter<number>) {
  const lang = useLang()
  return useQuery({
    queryKey: ['multilateral', r(filters), r(minVotes), lang],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/multilateral', {
          params: {
            query: { ...toApiParams(toValue(filters)), min_votes: toValue(minVotes), lang: lang.value },
          },
        }),
      ),
    placeholderData: keepPreviousData,
  })
}

export type CloudMode = 'default' | 'geopolitical' | 'thematic' | 'action' | 'category'

export function useWordCloud(filters: MaybeRefOrGetter<Filters>, mode: MaybeRefOrGetter<CloudMode>) {
  return useQuery({
    queryKey: ['cloud', r(filters), r(mode)],
    queryFn: () =>
      unwrap(
        api.GET('/api/v1/keywords/cloud', {
          params: { query: { ...toApiParams(toValue(filters)), mode: toValue(mode), limit: 40 } },
        }),
      ),
    placeholderData: keepPreviousData,
  })
}

export function useProfile(
  country: MaybeRefOrGetter<string>,
  range: MaybeRefOrGetter<{ from: number | null; to: number | null }>,
  compare: MaybeRefOrGetter<string[]>,
) {
  const lang = useLang()
  return useQuery({
    queryKey: ['profile', r(country), r(range), r(compare), lang],
    queryFn: () => {
      const { from, to } = toValue(range)
      return unwrap(
        api.GET('/api/v1/countries/{iso3}/profile', {
          params: {
            path: { iso3: toValue(country) },
            query: {
              start_date: from ? `${from}-01-01` : undefined,
              end_date: to ? `${to}-12-31` : undefined,
              compare: toValue(compare).length ? toValue(compare) : undefined,
              lang: lang.value,
            },
          },
        }),
      )
    },
    placeholderData: keepPreviousData,
  })
}

export function useSponsorship(
  country: MaybeRefOrGetter<string>,
  range: MaybeRefOrGetter<{ from: number | null; to: number | null }>,
) {
  const lang = useLang()
  return useQuery({
    queryKey: ['sponsorship', r(country), r(range), lang],
    queryFn: () => {
      const { from, to } = toValue(range)
      return unwrap(
        api.GET('/api/v1/countries/{iso3}/sponsorship', {
          params: {
            path: { iso3: toValue(country) },
            query: {
              start_date: from ? `${from}-01-01` : undefined,
              end_date: to ? `${to}-12-31` : undefined,
              lang: lang.value,
            },
          },
        }),
      )
    },
    placeholderData: keepPreviousData,
  })
}

/** Build the CSV download URL for the current filters (a plain link: the browser downloads it). */
export function csvUrl(filters: Filters, votes: string[], lang: Lang): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(toApiParams(filters))) {
    if (Array.isArray(value)) value.forEach((v) => params.append(key, v))
    else if (value) params.append(key, String(value))
  }
  votes.forEach((v) => params.append('votes', v))
  params.set('lang', lang)
  return `/api/v1/resolutions.csv?${params}`
}
