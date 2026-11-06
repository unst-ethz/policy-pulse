/**
 * Every facet a page can offer, in the order they stack. A page names the ones it allows (only
 * those that change what it shows) and the ones it starts with; the viewer adds the others.
 */
import type { Component } from 'vue'
import { exactTerms, type Filters } from '@/composables/useFilters'
import AdoptionFacet from './AdoptionFacet.vue'
import CountryFacet from './CountryFacet.vue'
import GroupFacet from './GroupFacet.vue'
import HistoricalFacet from './HistoricalFacet.vue'
import SponsorFacet from './SponsorFacet.vue'
import SubjectsFacet from './SubjectsFacet.vue'
import TermsFacet from './TermsFacet.vue'
import YearsFacet from './YearsFacet.vue'

export type FacetKey = 'years' | 'adoption' | 'subjects' | 'terms' | 'group' | 'country' | 'sponsor' | 'historical'

export interface Facet {
  key: FacetKey
  /** i18n key of its title */
  title: string
  component: Component
  /** Whether its filter is set: then it shows even if the viewer did not choose it. */
  active: (f: Filters) => boolean
}

export const FACETS: Facet[] = [
  { key: 'years', title: 'filters.years', component: YearsFacet, active: (f) => !!(f.from || f.to) },
  { key: 'adoption', title: 'facets.adoption', component: AdoptionFacet, active: (f) => f.adoption.length > 0 },
  { key: 'subjects', title: 'filters.subjects', component: SubjectsFacet, active: (f) => f.subjects.length > 0 },
  { key: 'terms', title: 'facets.terms', component: TermsFacet, active: (f) => exactTerms(f.keyword).length > 0 },
  { key: 'group', title: 'facets.group', component: GroupFacet, active: (f) => f.group.length > 0 },
  { key: 'country', title: 'filters.country', component: CountryFacet, active: (f) => !!f.country },
  { key: 'sponsor', title: 'filters.sponsor', component: SponsorFacet, active: (f) => !!f.sponsor },
  { key: 'historical', title: 'facets.historical', component: HistoricalFacet, active: (f) => f.historical },
]

export const ALL_FACETS = FACETS.map((f) => f.key)

/** How many of `keys`' facets have their filter set (for "Filters (n)"). */
export const activeCount = (keys: FacetKey[], filters: Filters) =>
  FACETS.filter((f) => keys.includes(f.key) && f.active(filters)).length
