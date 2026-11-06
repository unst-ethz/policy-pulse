<script setup lang="ts">
/**
 * How a resolution was adopted: by a recorded vote (each state's vote known), by a vote without
 * a per-state record, or without a vote. Counts apply every other filter.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFacets } from '@/api/queries'
import { ADOPTIONS, type Adoption, useFilters } from '@/composables/useFilters'
import FacetPanel from './FacetPanel.vue'

const { t, n } = useI18n()
const { filters, update } = useFilters()
const facets = useFacets(filters)

const counts = computed(
  () => new Map((facets.data.value?.adoption ?? []).map((a) => [a.adoption, a.resolutions])),
)

function toggle(kind: Adoption) {
  const now = filters.value.adoption
  update({ adoption: now.includes(kind) ? now.filter((a) => a !== kind) : [...now, kind] })
}
</script>

<template>
  <FacetPanel
    :info="t('facets.info.adoption')"
    :title="t('facets.adoption')"
    :active="filters.adoption.length > 0"
    @clear="update({ adoption: [] })"
  >
    <ul class="kinds" :class="{ fetching: facets.isFetching.value }">
      <li v-for="kind in ADOPTIONS" :key="kind">
        <label class="row" :class="{ empty: !counts.get(kind) }">
          <input type="checkbox" :checked="filters.adoption.includes(kind)" @change="toggle(kind)" />
          <span>{{ t(`facets.adoptions.${kind}`) }}</span>
          <span class="count">{{ n(counts.get(kind) ?? 0) }}</span>
        </label>
      </li>
    </ul>
  </FacetPanel>
</template>

<style scoped>
.kinds {
  display: grid;
  gap: 2px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: var(--text-sm);
  transition: opacity 0.15s;
}

.kinds.fetching {
  opacity: 0.6;
}

.row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  column-gap: var(--space-2);
  padding-block: 3px;
  cursor: pointer;
}

.row input {
  margin: 0;
  accent-color: var(--ink);
}

.row.empty {
  color: var(--muted);
}

.count {
  color: var(--ink-2);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}
</style>
