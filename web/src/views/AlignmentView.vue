<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { useMultilateral } from '@/api/queries'
import AlignmentScatter from '@/components/AlignmentScatter.vue'
import ChartFrame from '@/components/ChartFrame.vue'
import CountryPicker from '@/components/CountryPicker.vue'
import FilterBar from '@/components/FilterBar.vue'
import QueryState from '@/components/QueryState.vue'
import type { ScatterPoint } from '@/charts/types'
import { useFilters } from '@/composables/useFilters'

type Metric = 'yes_rate' | 'no_rate' | 'abstention_rate'
const METRICS: Metric[] = ['yes_rate', 'no_rate', 'abstention_rate']
const REGIONS = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']

const { t, n } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()

function param<T extends string>(key: string, allowed: readonly T[], fallback: T | null) {
  return computed<T | null>({
    get: () => (allowed.includes(route.query[key] as T) ? (route.query[key] as T) : fallback),
    set: (value) => router.replace({ query: { ...route.query, [key]: value ?? undefined } }),
  })
}

const metric = param<Metric>('y', METRICS, 'abstention_rate')
const region = param<string>('region', REGIONS, null)
const focus = computed<string | null>({
  get: () => (route.query.focus as string | undefined) ?? null,
  set: (value) => router.replace({ query: { ...route.query, focus: value ?? undefined } }),
})

const listFilters = computed(() => ({ ...filters.value, country: null, mode: 'none' as const }))
const result = useMultilateral(listFilters, 10)

const points = computed<ScatterPoint[]>(() =>
  (result.data.value?.countries ?? [])
    .filter((c) => c[metric.value!] != null)
    .map((c) => ({
      country: c.country,
      name: c.name,
      region: c.region,
      x: c.alignment,
      y: c[metric.value!]!,
      votes: c.votes_cast,
    })),
)
const yLabel = computed(() => t(`alignment.metrics.${metric.value}`))
const rows = computed(() => [...(result.data.value?.countries ?? [])].sort((a, b) => b.alignment - a.alignment))
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('alignment.title') }}</h1>
      <p>{{ t('alignment.lead') }}</p>
    </div>

    <FilterBar :fields="['years', 'subjects', 'keyword']" />

    <ChartFrame
      :title="t('alignment.chartTitle', { metric: yLabel.toLocaleLowerCase() })"
      :description="result.data.value ? t('alignment.chartText', { n: n(result.data.value.resolution_count) }) : undefined"
    >
      <template #controls>
        <label class="field inline">
          <span>{{ t('alignment.y') }}</span>
          <select class="input" :value="metric" @change="metric = ($event.target as HTMLSelectElement).value as Metric">
            <option v-for="m in METRICS" :key="m" :value="m">{{ t(`alignment.metrics.${m}`) }}</option>
          </select>
        </label>
      </template>
      <div class="emphasis">
        <div class="regions" role="group" :aria-label="t('alignment.region')">
          <span>{{ t('alignment.region') }}</span>
          <button
            v-for="r in REGIONS"
            :key="r"
            type="button"
            :aria-pressed="region === r"
            @click="region = region === r ? null : r"
          >
            {{ t(`regions.${r}`) }}
          </button>
        </div>
        <CountryPicker v-model="focus" :label="t('alignment.focus')" class="focus" />
      </div>
      <QueryState
        :loading="result.isLoading.value"
        :fetching="result.isFetching.value"
        :error="result.error.value"
        :empty="points.length === 0"
        :empty-text="t('alignment.empty')"
        min-height="520px"
        @retry="result.refetch()"
      >
        <AlignmentScatter
          :points="points"
          :y-label="yLabel"
          :mean-x="result.data.value?.mean_alignment ?? null"
          :region="region"
          :selected="focus"
          @select="(c) => (focus = c)"
        />
      </QueryState>
      <template #table>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">{{ t('country.country') }}</th>
              <th scope="col">{{ t('alignment.regionCol') }}</th>
              <th scope="col" class="numeric">{{ t('alignment.x') }}</th>
              <th v-for="m in METRICS" :key="m" scope="col" class="numeric">{{ t(`alignment.metrics.${m}`) }}</th>
              <th scope="col" class="numeric">{{ t('alignment.votesCol') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="c in rows" :key="c.country">
              <td><RouterLink :to="{ name: 'country', params: { iso3: c.country } }">{{ c.name }}</RouterLink></td>
              <td>{{ t(`regions.${c.region}`, c.region) }}</td>
              <td class="numeric">{{ n(c.alignment, 'score') }}</td>
              <td v-for="m in METRICS" :key="m" class="numeric">{{ c[m] == null ? '' : n(c[m]!, 'percent') }}</td>
              <td class="numeric">{{ n(c.votes_cast) }}</td>
            </tr>
          </tbody>
        </table>
      </template>
      <template #note>
        <p>{{ t('alignment.note', { n: result.data.value?.min_votes ?? 10 }) }}</p>
      </template>
    </ChartFrame>
  </div>
</template>

<style scoped>
.inline {
  grid-auto-flow: column;
  align-items: center;
  gap: var(--space-2);
}

.emphasis {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  justify-content: space-between;
  gap: var(--space-3);
  margin-bottom: var(--space-3);
}

.regions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  color: var(--ink-2);
}

.regions button {
  padding: 3px 12px;
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  background: var(--surface);
  font-size: var(--text-sm);
  cursor: pointer;
}

.regions button[aria-pressed='true'] {
  border-color: var(--yes);
  box-shadow: inset 0 0 0 1px var(--yes);
  font-weight: 600;
}

/* The pressed chip doubles as the legend for the emphasised (blue) dots. */
.regions button[aria-pressed='true']::before {
  content: '';
  display: inline-block;
  width: 8px;
  height: 8px;
  margin-inline-end: 6px;
  border-radius: 50%;
  background: var(--yes);
}

.focus {
  width: 240px;
}
</style>
