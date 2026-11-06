<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useDomains } from '@/api/queries'
import FilterBar from '@/components/FilterBar.vue'
import QueryState from '@/components/QueryState.vue'
import { serializeFilters, useFilters } from '@/composables/useFilters'
import { sentenceCase } from '@/format'

const { t, n } = useI18n()
const { filters } = useFilters()

const range = computed(() => ({ from: filters.value.from, to: filters.value.to }))
const domains = useDomains(range)

// Every card's sparkline spans the same years, so their shapes compare
const years = computed(() => (domains.data.value?.per_year ?? []).map((y) => y.year))
const yearsQuery = computed(() => serializeFilters({ ...filters.value, subjects: [], country: null, keyword: '', sponsor: null }))

const cards = computed(() =>
  (domains.data.value?.domains ?? []).map((d) => {
    const counts = new Map(d.per_year.map((y) => [y.year, y.resolutions]))
    const max = Math.max(1, ...counts.values())
    return {
      ...d,
      label: sentenceCase(d.label),
      bars: years.value.map((year, i) => ({ year, x: i, h: (counts.get(year) ?? 0) / max })),
      recordedShare: d.resolutions ? d.recorded / d.resolutions : 0,
    }
  }),
)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('subject.title') }}</h1>
      <p>{{ t('subject.lead') }}</p>
    </div>

    <FilterBar :fields="['years']" />

    <QueryState
      :loading="domains.isLoading.value"
      :fetching="domains.isFetching.value"
      :error="domains.error.value"
      :empty="cards.length === 0"
      :empty-text="t('subject.empty')"
      min-height="320px"
      @retry="domains.refetch()"
    >
      <ul class="cards">
        <li v-for="card in cards" :key="card.subject_id">
          <RouterLink class="card panel" :to="{ name: 'subject', params: { id: card.subject_id }, query: yearsQuery }">
            <h2>{{ card.label }}</h2>
            <p class="figures">
              <span class="figure">{{ t('subject.count', { n: n(card.resolutions) }) }}</span>
              <span class="caption">{{ t('subject.recordedShare', { pct: n(card.recordedShare, 'percentShort') }) }}</span>
            </p>
            <svg
              class="spark"
              :viewBox="`0 0 ${Math.max(1, card.bars.length)} 1`"
              preserveAspectRatio="none"
              role="img"
              :aria-label="t('subject.perYear')"
            >
              <rect
                v-for="bar in card.bars"
                :key="bar.year"
                :x="bar.x + 0.15"
                :y="1 - bar.h"
                width="0.7"
                :height="bar.h"
              />
            </svg>
          </RouterLink>
        </li>
      </ul>
    </QueryState>

    <p class="hint">{{ t('subject.coverage') }}</p>
  </div>
</template>

<style scoped>
.cards {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 260px), 1fr));
  gap: var(--space-4);
}

.card {
  display: grid;
  gap: var(--space-2);
  height: 100%;
  color: inherit;
  text-decoration: none;
  transition: border-color 0.1s;
}

.card:hover,
.card:focus-visible {
  border-color: var(--rule-strong);
}

.card:hover h2 {
  text-decoration: underline;
}

.card h2 {
  font-size: var(--text-md);
  font-weight: 500;
}

.figures {
  display: grid;
  margin: 0;
}

.figure {
  font-size: var(--text-lg);
  font-weight: 500;
}

.caption,
.hint {
  color: var(--muted);
  font-size: var(--text-xs);
}

.spark {
  width: 100%;
  height: 40px;
  margin-top: auto;
  fill: var(--muted);
}
</style>
