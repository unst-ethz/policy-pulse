<script setup lang="ts">
/**
 * Subjects: the largest thesaurus domains with how many resolutions each would give, under every
 * filter but the subjects. Picked subjects below the domain level show as chips; "More subjects"
 * opens the whole tree, with search and the levels below.
 */
import { PopoverTrigger } from 'reka-ui'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFacets, useSubjectTree } from '@/api/queries'
import { useFilters, usePageFilters } from '@/composables/useFilters'
import { flatSubjects } from '@/format'
import SubjectPicker from '../SubjectPicker.vue'
import FacetPanel from './FacetPanel.vue'

const NO_SUBJECT = '__no_subject__'
const SHOWN = 5

const { t, n } = useI18n()
const { filters, update } = useFilters()
const facets = useFacets(usePageFilters())
const tree = useSubjectTree()

const subjects = computed({
  get: () => filters.value.subjects,
  set: (value: string[]) => update({ subjects: value }),
})

// Every domain, largest first; empty ones count as such rather than vanish
const domains = computed(() => {
  const counts = new Map((facets.data.value?.domains ?? []).map((d) => [d.subject_id, d.resolutions]))
  const rows = flatSubjects(tree.data.value ?? [])
    .filter((s) => !s.domain)
    .map((s) => ({ id: s.id, label: s.label, count: counts.get(s.id) ?? 0 }))
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
  const max = Math.max(1, ...rows.map((r) => r.count))
  return rows.map((r) => ({ ...r, width: (r.count / max) * 100 }))
})

// The largest few, plus any picked domain further down so it can be unticked here
const shown = computed(() =>
  domains.value.filter((d, i) => i < SHOWN || subjects.value.includes(d.id)),
)

const deeper = computed(() => {
  const labels = new Map(flatSubjects(tree.data.value ?? []).map((s) => [s.id, s.label]))
  const top = new Set(domains.value.map((d) => d.id))
  return subjects.value
    .filter((id) => !top.has(id))
    .map((id) => ({ id, label: id === NO_SUBJECT ? t('subjects.none') : (labels.get(id) ?? id) }))
})

function toggle(id: string) {
  subjects.value = subjects.value.includes(id) ? subjects.value.filter((s) => s !== id) : [...subjects.value, id]
}
</script>

<template>
  <FacetPanel
    :info="t('facets.info.subjects')" :title="t('filters.subjects')" :active="subjects.length > 0" @clear="subjects = []">
    <div v-if="deeper.length" class="chips">
      <button
        v-for="s in deeper"
        :key="s.id"
        type="button"
        class="chip"
        :title="t('facets.removeSubject', { name: s.label })"
        @click="toggle(s.id)"
      >
        {{ s.label }} <span aria-hidden="true">×</span>
      </button>
    </div>
    <ul class="domains" :class="{ fetching: facets.isFetching.value }">
      <li v-for="d in shown" :key="d.id">
        <label class="row" :class="{ empty: !d.count }">
          <input type="checkbox" :checked="subjects.includes(d.id)" @change="toggle(d.id)" />
          <span class="name">{{ d.label }}</span>
          <span class="count">{{ n(d.count) }}</span>
          <span class="bar" aria-hidden="true"><span :style="{ width: `${d.width}%` }" /></span>
        </label>
      </li>
    </ul>
    <SubjectPicker v-model="subjects" :label="t('filters.subjects')">
      <template #trigger>
        <PopoverTrigger class="more">{{ t('facets.moreSubjects') }}</PopoverTrigger>
      </template>
    </SubjectPicker>
  </FacetPanel>
</template>

<style scoped>
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.chip {
  padding: 3px 10px;
  border: 1px solid var(--ink);
  border-radius: 999px;
  background: var(--ink);
  color: var(--surface);
  font: inherit;
  font-size: var(--text-sm);
  cursor: pointer;
}

.domains {
  display: grid;
  gap: 2px;
  margin: 0;
  padding: 0;
  list-style: none;
  font-size: var(--text-sm);
  transition: opacity 0.15s;
}

.domains.fetching {
  opacity: 0.6;
}

.row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: baseline;
  column-gap: var(--space-2);
  padding-block: 3px;
  cursor: pointer;
}

.row input {
  margin: 0;
  accent-color: var(--ink);
  align-self: center;
}

.row.empty .name,
.row.empty .count {
  color: var(--muted);
}

.count {
  color: var(--ink-2);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

.bar {
  grid-column: 2 / -1;
  height: 3px;
  margin-top: 2px;
  background: var(--rule);
}

.bar > span {
  display: block;
  height: 100%;
  background: var(--ink-2);
}

.more {
  justify-self: start;
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
