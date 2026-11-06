<script setup lang="ts">
/**
 * A page's filters as a column of facets. Each facet reads and writes the URL filter on its own;
 * this only decides which show: of the facets the page allows, those the viewer chose (kept in
 * this browser per page, starting with the page's defaults), and always those whose filter is
 * set, so a shared link never hides a filter that changes the page.
 */
import { PopoverContent, PopoverPortal, PopoverRoot, PopoverTrigger } from 'reka-ui'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFilters } from '@/composables/useFilters'
import { activeCount, FACETS, type FacetKey } from './registry'

const props = defineProps<{
  allowed: FacetKey[]
  defaults: FacetKey[]
  storageKey: string
  /** How many filters are set, when the page has some outside the facets (default: the facets'). */
  count?: number
}>()

const { t } = useI18n()
const { filters, clear } = useFilters()
const facets = computed(() => FACETS.filter((f) => props.allowed.includes(f.key)))
const active = computed(() => props.count ?? activeCount(props.allowed, filters.value))

function load(): string[] {
  try {
    const saved = JSON.parse(localStorage.getItem(props.storageKey) ?? 'null')
    return Array.isArray(saved) ? saved : props.defaults
  } catch {
    return props.defaults
  }
}
const chosen = ref<string[]>(load())
watch(chosen, (value) => {
  try {
    localStorage.setItem(props.storageKey, JSON.stringify(value))
  } catch {
    // Storage blocked (private mode): the choice lasts for this visit only
  }
})

const shown = computed(() => facets.value.filter((f) => chosen.value.includes(f.key) || f.active(filters.value)))
// Nothing to choose when the page starts with every facet it has
const choosable = computed(() => facets.value.some((f) => !props.defaults.includes(f.key)))

function choose(key: string, on: boolean) {
  chosen.value = on ? [...chosen.value, key] : chosen.value.filter((k) => k !== key)
}
</script>

<template>
  <div class="facets" role="group" :aria-label="t('filters.label')">
    <div class="head">
      <h2>{{ t('facets.title') }}</h2>
      <PopoverRoot v-if="choosable">
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
            <label v-for="f in facets" :key="f.key" class="facet-chooser-option">
              <input
                type="checkbox"
                :checked="chosen.includes(f.key) || f.active(filters)"
                :disabled="f.active(filters)"
                @change="choose(f.key, ($event.target as HTMLInputElement).checked)"
              />
              {{ t(f.title) }}
            </label>
            <p v-if="facets.some((f) => f.active(filters))" class="facet-chooser-note">{{ t('facets.chooseNote') }}</p>
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
