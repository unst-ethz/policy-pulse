<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { useHistoricalStates, useMultilateral } from '@/api/queries'
import AlignmentScatter from '@/components/AlignmentScatter.vue'
import ChartFrame from '@/components/ChartFrame.vue'
import CountryList from '@/components/CountryList.vue'
import FacetLayout from '@/components/facets/FacetLayout.vue'
import QueryState from '@/components/QueryState.vue'
import { REGIONS, type ScatterPoint } from '@/charts/types'
import { onlyFilters, useFilters } from '@/composables/useFilters'
import { useThemeColors } from '@/composables/useThemeColors'

type Metric = 'yes_rate' | 'no_rate' | 'abstention_rate'
const METRICS: Metric[] = ['yes_rate', 'no_rate', 'abstention_rate']
// The permanent Security Council members: known reference points for a first look
const P5 = ['USA', 'CHN', 'RUS', 'GBR', 'FRA']
const MAX_MARKED = 5

const { t, n } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()
const colors = useThemeColors()

function param<T extends string>(key: string, allowed: readonly T[], fallback: T | null) {
  return computed<T | null>({
    get: () => (allowed.includes(route.query[key] as T) ? (route.query[key] as T) : fallback),
    set: (value) => router.replace({ query: { ...route.query, [key]: value ?? undefined } }),
  })
}

const metric = param<Metric>('y', METRICS, 'abstention_rate')
const region = param<string>('region', REGIONS, null)
// `mark` absent means the P5; an empty `mark=` keeps an emptied list empty
const marked = computed<string[]>({
  get: () => {
    const raw = route.query.mark
    if (raw === undefined) return P5
    return ([] as (string | null)[]).concat(raw).filter((c): c is string => !!c)
  },
  set: (value) => router.replace({ query: { ...route.query, mark: value.length ? value : '' } }),
})
function toggle(iso3: string) {
  if (marked.value.includes(iso3)) marked.value = marked.value.filter((c) => c !== iso3)
  else if (marked.value.length < MAX_MARKED) marked.value = [...marked.value, iso3]
}

// Only the sidebar's filters apply (a keyword or group carried over in the URL would filter
// invisibly); `historical` only hides points, so it stays out of the request.
const listFilters = computed(() => ({
  ...onlyFilters(filters.value, ['from', 'to', 'subjects', 'sponsor', 'original']),
  mode: 'none' as const,
}))
const result = useMultilateral(listFilters, 10)

const historical = useHistoricalStates()
const countries = computed(() =>
  (result.data.value?.countries ?? []).filter((c) => filters.value.historical || !historical.value.has(c.country)),
)

// Over the countries shown, so hidden historical States don't shift it
const meanAlignment = computed(() =>
  countries.value.length ? countries.value.reduce((sum, c) => sum + c.alignment, 0) / countries.value.length : null,
)

const regionColor = (r: string) =>
  (REGIONS as readonly string[]).includes(r) ? colors[`region-${r.toLowerCase()}` as keyof typeof colors] : colors.abstain
const markedColors = computed(() =>
  Object.fromEntries(countries.value.filter((c) => marked.value.includes(c.country)).map((c) => [c.country, regionColor(c.region)])),
)

const points = computed<ScatterPoint[]>(() =>
  countries.value
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
const rows = computed(() => [...countries.value].sort((a, b) => b.alignment - a.alignment))

// One sentence per marked country: which side of the average it falls on
const explanations = computed(() => {
  const mean = meanAlignment.value
  if (mean == null) return []
  return marked.value.flatMap((iso3) => {
    const p = points.value.find((q) => q.country === iso3)
    if (!p) return []
    return [{
      country: iso3,
      color: regionColor(p.region),
      text: t(p.x >= mean ? 'alignment.explainWith' : 'alignment.explainAgainst', {
        name: p.name,
        x: n(p.x, 'score'),
        mean: n(mean, 'score'),
      }),
    }]
  })
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('alignment.title') }}</h1>
      <p>{{ t('alignment.lead') }}</p>
    </div>

    <FacetLayout
      :allowed="['years', 'subjects', 'sponsor', 'historical']"
      :defaults="['years', 'subjects']"
      storage-key="policy-pulse:facets:alignment"
      :applied="listFilters"
    >
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
            :style="{ '--chip': regionColor(r) }"
            @click="region = region === r ? null : r"
          >
            {{ t(`regions.${r}`) }}
          </button>
        </div>
        <div class="marked">
          <CountryList v-model="marked" :label="t('alignment.marked')" :max="MAX_MARKED" :colors="markedColors" />
        </div>
      </div>
      <QueryState
        :loading="result.isLoading.value"
        :fetching="result.isFetching.value"
        :error="result.error.value"
        :empty="points.length === 0"
        :empty-text="t('alignment.empty')"
        min-height="560px"
        @retry="result.refetch()"
      >
        <AlignmentScatter
          :points="points"
          :y-label="yLabel"
          :mean-x="meanAlignment"
          :region="region"
          :marked="marked"
          @select="toggle"
        />
        <ul v-if="explanations.length" class="explain">
          <li v-for="e in explanations" :key="e.country">
            <span class="dot" :style="{ background: e.color }" aria-hidden="true" />{{ e.text }}
          </li>
        </ul>
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
    </FacetLayout>
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
  border-color: var(--chip);
  box-shadow: inset 0 0 0 1px var(--chip);
  font-weight: 600;
}

.marked {
  flex-basis: 100%;
}

.explain {
  display: grid;
  gap: var(--space-1);
  margin: var(--space-3) 0 0;
  padding: 0;
  list-style: none;
  font-size: var(--text-sm);
  color: var(--ink-2);
}

.explain .dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-inline-end: var(--space-2);
  border-radius: 50%;
  border: 2px solid var(--ink);
}
</style>
