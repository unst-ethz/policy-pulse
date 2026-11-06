<script setup lang="ts">
/**
 * The resolution list's filters as a column of facets. Each facet reads and writes the URL
 * filter on its own; this only decides which show. The viewer picks them (kept in this
 * browser), starting with years and subjects, and a facet whose filter is set always shows, so
 * a shared link never hides a filter that narrows the list.
 */
import { PopoverContent, PopoverPortal, PopoverRoot, PopoverTrigger } from 'reka-ui'
import { type Component, computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { countActive, exactTerms, type Filters, useFilters } from '@/composables/useFilters'
import AdoptionFacet from './AdoptionFacet.vue'
import CountryFacet from './CountryFacet.vue'
import GroupFacet from './GroupFacet.vue'
import SponsorFacet from './SponsorFacet.vue'
import SubjectsFacet from './SubjectsFacet.vue'
import TermsFacet from './TermsFacet.vue'
import YearsFacet from './YearsFacet.vue'

const STORAGE_KEY = 'policy-pulse:facets'
const DEFAULT = ['years', 'subjects']

const FACETS: { key: string; title: string; component: Component; active: (f: Filters) => boolean }[] = [
  { key: 'years', title: 'filters.years', component: YearsFacet, active: (f) => !!(f.from || f.to) },
  { key: 'adoption', title: 'facets.adoption', component: AdoptionFacet, active: (f) => f.adoption.length > 0 },
  { key: 'subjects', title: 'filters.subjects', component: SubjectsFacet, active: (f) => f.subjects.length > 0 },
  { key: 'terms', title: 'facets.terms', component: TermsFacet, active: (f) => exactTerms(f.keyword).length > 0 },
  { key: 'group', title: 'facets.group', component: GroupFacet, active: (f) => f.group.length > 0 },
  { key: 'country', title: 'filters.country', component: CountryFacet, active: (f) => !!f.country },
  { key: 'sponsor', title: 'filters.sponsor', component: SponsorFacet, active: (f) => !!f.sponsor },
]

const { t } = useI18n()
const { filters, clear } = useFilters()
const active = computed(() => countActive(filters.value))

function load(): string[] {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    return Array.isArray(saved) ? saved : DEFAULT
  } catch {
    return DEFAULT
  }
}
const chosen = ref<string[]>(load())
watch(chosen, (value) => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(value))
  } catch {
    // Storage blocked (private mode): the choice lasts for this visit only
  }
})

const shown = computed(() => FACETS.filter((f) => chosen.value.includes(f.key) || f.active(filters.value)))

function choose(key: string, on: boolean) {
  chosen.value = on ? [...chosen.value, key] : chosen.value.filter((k) => k !== key)
}
</script>

<template>
  <div class="facets" role="group" :aria-label="t('filters.label')">
    <div class="head">
      <h2>{{ t('facets.title') }}</h2>
      <PopoverRoot>
        <PopoverTrigger class="pick" :aria-label="t('facets.choose')" :title="t('facets.choose')">
          <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
            <path d="M2 4h7M13 4h1M2 12h1M7 12h7" stroke="currentColor" stroke-width="1.5" fill="none" />
            <circle cx="11" cy="4" r="2" stroke="currentColor" stroke-width="1.5" fill="none" />
            <circle cx="5" cy="12" r="2" stroke="currentColor" stroke-width="1.5" fill="none" />
          </svg>
        </PopoverTrigger>
        <PopoverPortal>
          <PopoverContent class="facet-chooser" side="bottom" align="start" :side-offset="6" :collision-padding="12">
            <p class="facet-chooser-title">{{ t('facets.choose') }}</p>
            <label v-for="f in FACETS" :key="f.key" class="facet-chooser-option">
              <input
                type="checkbox"
                :checked="chosen.includes(f.key) || f.active(filters)"
                :disabled="f.active(filters)"
                @change="choose(f.key, ($event.target as HTMLInputElement).checked)"
              />
              {{ t(f.title) }}
            </label>
            <p v-if="FACETS.some((f) => f.active(filters))" class="facet-chooser-note">{{ t('facets.chooseNote') }}</p>
          </PopoverContent>
        </PopoverPortal>
      </PopoverRoot>
      <button v-if="active" type="button" class="clear" @click="clear">
        {{ t('filters.clear', { n: active }) }}
      </button>
    </div>
    <component :is="f.component" v-for="f in shown" :key="f.key" />
  </div>
</template>

<style scoped>
.facets {
  display: grid;
  align-content: start;
  min-width: 0;
}

.head {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding-bottom: var(--space-3);
}

h2 {
  margin: 0;
  font-size: var(--text-lg);
}

.pick {
  display: inline-grid;
  place-items: center;
  width: 28px;
  height: 28px;
  padding: 0;
  border: 1px solid var(--rule-strong);
  border-radius: var(--radius-s);
  background: var(--surface);
  color: var(--ink-2);
  cursor: pointer;
}

.pick:hover,
.pick[data-state='open'] {
  border-color: var(--ink-2);
  color: var(--ink);
}

.clear {
  margin-inline-start: auto;
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  font-size: var(--text-sm);
  color: var(--accent);
  text-decoration: underline;
  cursor: pointer;
}
</style>

<style>
/* Portaled outside the component, so unscoped */
.facet-chooser {
  z-index: 300; /* above the filter drawer */
  display: grid;
  gap: 2px;
  min-width: 220px;
  max-width: min(300px, 90vw);
  padding: var(--space-3);
  background: var(--surface);
  border: 1px solid var(--rule-strong);
  border-radius: var(--radius-s);
  box-shadow: 0 4px 16px rgb(0 0 0 / 0.08);
  font-size: var(--text-sm);
}

.facet-chooser-title {
  margin: 0 0 var(--space-2);
  font-weight: 600;
}

.facet-chooser-option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding-block: 3px;
  cursor: pointer;
}

.facet-chooser-option input {
  margin: 0;
  accent-color: var(--ink);
}

.facet-chooser-option:has(input:disabled) {
  cursor: default;
  color: var(--ink-2);
}

.facet-chooser-note {
  margin: var(--space-2) 0 0;
  font-size: var(--text-xs);
  color: var(--muted);
}
</style>
