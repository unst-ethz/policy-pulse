<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import {
  useAgreementBySubject,
  useAgreementMap,
  useCountryIndex,
  useProfile,
  useTimeline,
} from '@/api/queries'
import AgreementBars from '@/components/AgreementBars.vue'
import ChartFrame from '@/components/ChartFrame.vue'
import CountryList from '@/components/CountryList.vue'
import CountryPicker from '@/components/CountryPicker.vue'
import FilterBar from '@/components/FilterBar.vue'
import QueryState from '@/components/QueryState.vue'
import ScaleLegend from '@/components/ScaleLegend.vue'
import TimelineChart from '@/components/TimelineChart.vue'
import type { TimelineSeries } from '@/charts/types'
import VoteBar from '@/components/VoteBar.vue'
import VoteChip from '@/components/VoteChip.vue'
import WorldMap from '@/components/WorldMap.vue'
import { divergingScale } from '@/charts/scales'
import { useFilters } from '@/composables/useFilters'
import { useThemeColors } from '@/composables/useThemeColors'
import { cleanTitle, sentenceCase, SERIES_SLOTS, SERIES_SLOTS_DARK, yearSpan } from '@/format'

const props = defineProps<{ iso3: string }>()

const { t, n, d } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()
const { index, name } = useCountryIndex()
const colors = useThemeColors()

const country = computed(() => index.value.get(props.iso3))
const range = computed(() => ({ from: filters.value.from, to: filters.value.to }))

function listParam(key: string): string[] {
  const v = route.query[key]
  return (Array.isArray(v) ? v : v ? [v] : []).filter((x): x is string => !!x)
}
function setParam(key: string, value: string | string[] | undefined) {
  const query = { ...route.query, [key]: value }
  if (value === undefined || (Array.isArray(value) && !value.length)) delete query[key]
  router.replace({ query })
}

const requested = computed({
  get: () => listParam('compare'),
  set: (value: string[]) => setParam('compare', value),
})
const profile = useProfile(() => props.iso3, range, requested)
const p = computed(() => profile.data.value)

// What the comparison charts use: what was asked for, else the profile's default (the P5).
const compare = computed(() =>
  requested.value.length ? requested.value : (p.value?.comparison ?? []).filter((c) => c !== props.iso3),
)
const compareModel = computed({
  get: () => compare.value,
  set: (value: string[]) => (requested.value = value),
})

// --- agreement map -------------------------------------------------------------------------
const mapFilters = computed(() => ({ ...filters.value, subjects: [], keyword: '', mode: 'none' as const, country: null }))
const map = useAgreementMap(mapFilters, () => props.iso3)
// Centred by default: most pairs agree 0.7-0.95, which on a fixed 0-1 scale is all one blue.
const centred = ref(true)
const scores = computed(() => new Map((map.data.value?.scores ?? []).map((s) => [s.country, s.score])))
// The midpoint when centred: this country's average agreement over the countries on the map, so
// blue reads "votes more like it than most" and red "less". (Not the mean consensus the API also
// returns: that compares with the world's agreement overall, and turns nearly every country red
// for a country that often votes with a minority.)
const average = computed(() => {
  const values = [...scores.value.values()].filter((v): v is number => v != null)
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null
})
const scale = computed(() =>
  divergingScale(colors.no, colors['scale-mid'], colors.yes, [0, 1], centred.value && average.value != null ? average.value : 0.5),
)
const fill = (iso3: string) => {
  if (iso3 === props.iso3) return colors.accent
  const score = scores.value.get(iso3)
  return score == null ? null : scale.value(score)
}
const mapTooltip = (iso3: string) => {
  if (iso3 === props.iso3) return `<b>${name(iso3)}</b>`
  const score = scores.value.get(iso3)
  return `<b>${name(iso3)}</b><br>${score == null ? t('country.noSharedVote') : t('country.agreement', { value: n(score, 'score') })}`
}
const mapRows = computed(() =>
  (map.data.value?.scores ?? [])
    .filter((s) => s.score != null)
    .map((s) => ({ ...s, name: name(s.country) }))
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0)),
)

// --- timeline -----------------------------------------------------------------------------
const timeline = useTimeline(() => props.iso3, compare)
const isDark = computed(() => colors.surface.toLowerCase() !== '#ffffff')
const seriesColor = (i: number) => (isDark.value ? SERIES_SLOTS_DARK : SERIES_SLOTS)[i % SERIES_SLOTS.length]!
const seriesColors = computed(() => Object.fromEntries(compare.value.map((c, i) => [c, seriesColor(i)])))
const series = computed<TimelineSeries[]>(() =>
  (timeline.data.value?.compare ?? []).map((c) => ({
    country: c,
    label: name(c),
    color: seriesColors.value[c] ?? seriesColor(0),
    points: (timeline.data.value?.sessions ?? []).map((s) => ({
      year: s.year,
      session: s.session,
      agreement: s.values[c]?.agreement ?? null,
      votes: s.values[c]?.shared_votes ?? 0,
    })),
  })),
)

// --- by topic -----------------------------------------------------------------------------
const topicWith = computed({
  get: () => (route.query.with as string | undefined) ?? compare.value[0] ?? null,
  set: (value: string | null) => setParam('with', value ?? undefined),
})
const bySubject = useAgreementBySubject(() => props.iso3, topicWith, range)
const subjectRows = computed(() =>
  (bySubject.data.value?.subjects ?? []).map((s) => ({
    key: s.subject_id,
    label: sentenceCase(s.label),
    value: s.agreement,
    detail: t('country.sharedVotes', { n: s.shared_votes }),
    to: { name: 'resolutions', query: { subject: s.subject_id, country: props.iso3, mode: 'voted' } },
  })),
)

const period = computed(() => (p.value ? yearSpan(p.value.start_date, p.value.end_date) : ''))
const membershipText = computed(() => {
  const m = country.value?.membership ?? []
  const first = m[0]
  const last = m[m.length - 1]
  if (!first || !last) return ''
  return last.end
    ? t('country.memberFromTo', { from: first.start.slice(0, 4), to: last.end.slice(0, 4) })
    : t('country.memberSince', { date: d(new Date(first.start), 'long') })
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ country?.name ?? iso3 }}</h1>
      <p class="sub">
        <span v-if="country?.subregion">{{ country.subregion }}</span>
        <span v-if="membershipText">{{ membershipText }}</span>
        <span v-if="country?.historical_names.length">
          {{ t('countries.formerly', { names: country.historical_names.join(', ') }) }}
        </span>
      </p>
    </div>

    <FilterBar :fields="['years']" />

    <QueryState
      :loading="profile.isLoading.value"
      :fetching="profile.isFetching.value"
      :error="profile.error.value"
      min-height="200px"
      @retry="profile.refetch()"
    >
      <section v-if="p" class="overview panel">
        <div class="figures">
          <div>
            <span class="figure">{{ n(p.total_resolutions) }}</span>
            <span class="caption">{{ t('country.resolutionsIn', { period }) }}</span>
          </div>
          <div v-if="p.multilateral_rank">
            <span class="figure">{{ n(p.multilateral_alignment ?? 0, 'score') }}</span>
            <span class="caption">
              {{ t('country.alignmentRank', { rank: p.multilateral_rank, of: p.multilateral_ranked }) }}
            </span>
          </div>
        </div>
        <VoteBar
          :yes="p.votes.Y ?? 0"
          :no="p.votes.N ?? 0"
          :abstain="p.votes.A ?? 0"
          :absent="p.votes.absent ?? 0"
          :absent-label="t('country.noVoteRecorded')"
        />
        <p class="hint">{{ t('country.noVoteRecordedHelp') }}</p>
      </section>
    </QueryState>

    <section class="section">
      <div class="grid-agreement">
        <ChartFrame :title="t('country.mapTitle', { name: name(iso3) })" :description="t('country.mapText')">
          <template #controls>
            <label class="check">
              <input v-model="centred" type="checkbox" :disabled="average == null" />
              {{ t('country.centre') }}
            </label>
          </template>
          <QueryState
            :loading="map.isLoading.value"
            :fetching="map.isFetching.value"
            :error="map.error.value"
            @retry="map.refetch()"
          >
            <WorldMap
              :fill="fill"
              :tooltip="mapTooltip"
              :highlight="iso3"
              @select="(c) => router.push({ name: 'country', params: { iso3: c }, query: route.query })"
            />
          </QueryState>
          <ScaleLegend
            :color="scale"
            :low="t('country.alwaysOpposed')"
            :high="t('country.alwaysAgreed')"
            :center="centred && average != null ? { value: average, label: t('country.average', { name: name(iso3), value: n(average, 'score') }) } : null"
            :no-data="t('country.noSharedVote')"
          />
          <template #table>
            <table class="data-table">
              <thead>
                <tr>
                  <th scope="col">{{ t('country.country') }}</th>
                  <th scope="col" class="numeric">{{ t('country.agreementCol') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in mapRows" :key="row.country">
                  <td>{{ row.name }}</td>
                  <td class="numeric">{{ n(row.score ?? 0, 'score') }}</td>
                </tr>
              </tbody>
            </table>
          </template>
          <template #note>
            <p>{{ t('country.mapNote') }} {{ t('map.disclaimer') }}</p>
          </template>
        </ChartFrame>

        <section v-if="p" class="panel ranks">
          <h2>{{ t('country.closest') }}</h2>
          <AgreementBars
            :rows="p.most_aligned.map((r) => ({ key: r.country, label: name(r.country), value: r.agreement, to: { name: 'country', params: { iso3: r.country }, query: route.query } }))"
          />
          <h2>{{ t('country.furthest') }}</h2>
          <AgreementBars
            :rows="p.least_aligned.map((r) => ({ key: r.country, label: name(r.country), value: r.agreement, to: { name: 'country', params: { iso3: r.country }, query: route.query } }))"
          />
          <p class="hint">{{ t('country.ranksNote', { n: p.min_joint_votes }) }}</p>
        </section>
      </div>
    </section>

    <ChartFrame :title="t('country.timelineTitle')" :description="t('country.timelineText', { name: name(iso3) })">
      <CountryList v-model="compareModel" :label="t('country.compareWith')" :colors="seriesColors" :exclude="[iso3]" />
      <QueryState
        :loading="timeline.isLoading.value"
        :fetching="timeline.isFetching.value"
        :error="timeline.error.value"
        :empty="!compare.length"
        :empty-text="t('country.addCountry')"
        min-height="360px"
        @retry="timeline.refetch()"
      >
        <TimelineChart :series="series" />
      </QueryState>
      <template #table>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">{{ t('country.sessionCol') }}</th>
              <th scope="col">{{ t('country.yearCol') }}</th>
              <th v-for="s in series" :key="s.country" scope="col" class="numeric">{{ s.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="session in timeline.data.value?.sessions ?? []" :key="session.session">
              <td>{{ session.session }}</td>
              <td>{{ session.year }}</td>
              <td v-for="s in series" :key="s.country" class="numeric">
                {{ session.values[s.country]?.agreement == null ? '' : n(session.values[s.country]!.agreement!, 'score') }}
              </td>
            </tr>
          </tbody>
        </table>
      </template>
      <template #note>
        <p>{{ t('country.timelineNote', { n: timeline.data.value?.min_votes ?? 3 }) }}</p>
      </template>
    </ChartFrame>

    <section class="panel section">
      <div class="section-head">
        <div>
          <h2>{{ t('country.topicsTitle') }}</h2>
          <p>{{ t('country.topicsText') }}</p>
        </div>
        <CountryPicker v-model="topicWith" :label="t('country.compareWith')" :exclude="[iso3]" class="with" />
      </div>
      <QueryState
        :loading="bySubject.isLoading.value"
        :fetching="bySubject.isFetching.value"
        :error="bySubject.error.value"
        :empty="!topicWith || subjectRows.length === 0"
        :empty-text="topicWith ? t('country.topicsEmpty') : t('country.pickCountry')"
        @retry="bySubject.refetch()"
      >
        <AgreementBars :rows="subjectRows" />
      </QueryState>
    </section>

    <section v-if="p" class="section">
      <h2>{{ t('country.notable') }}</h2>
      <div class="grid-2">
        <div class="panel notable">
          <h3>{{ t('country.opposed') }}</h3>
          <p class="hint">{{ t('country.opposedText') }}</p>
          <ol>
            <li v-for="v in p.opposed_high_consensus" :key="v.undl_id">
              <VoteChip :vote="v.vote" />
              <span>
                <RouterLink :to="{ name: 'resolution', params: { id: v.undl_id } }">{{ cleanTitle(v.title) }}</RouterLink>
                <small>{{ v.symbol }}, {{ t('country.consensusOf', { value: n(v.consensus_score, 'score') }) }}</small>
              </span>
            </li>
          </ol>
        </div>
        <div class="panel notable">
          <h3>{{ t('country.supported') }}</h3>
          <p class="hint">{{ t('country.supportedText') }}</p>
          <ol>
            <li v-for="v in p.supported_low_consensus" :key="v.undl_id">
              <VoteChip :vote="v.vote" />
              <span>
                <RouterLink :to="{ name: 'resolution', params: { id: v.undl_id } }">{{ cleanTitle(v.title) }}</RouterLink>
                <small>{{ v.symbol }}, {{ t('country.consensusOf', { value: n(v.consensus_score, 'score') }) }}</small>
              </span>
            </li>
          </ol>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.sub {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-5);
  color: var(--ink-2);
}

.overview {
  display: grid;
  gap: var(--space-4);
}

.figures {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-4) var(--space-7);
}

.figures > div {
  display: grid;
}

.figure {
  font-size: var(--text-2xl);
  font-weight: 500;
  line-height: 1.1;
}

.caption {
  color: var(--ink-2);
  font-size: var(--text-sm);
}

.hint {
  color: var(--muted);
  font-size: var(--text-xs);
}

.grid-agreement {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr);
  gap: var(--space-5);
  align-items: start;
}

.ranks {
  display: grid;
  gap: var(--space-3);
}

.ranks h2 {
  font-size: var(--text-lg);
}

.ranks h2 + :deep(.bars) {
  margin-bottom: var(--space-3);
}

.check {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  color: var(--ink-2);
}

.check input {
  accent-color: var(--ink);
}

.with {
  width: 240px;
}

.notable {
  display: grid;
  gap: var(--space-2);
  align-content: start;
}

.notable ol {
  list-style: none;
  margin: var(--space-2) 0 0;
  padding: 0;
  display: grid;
  gap: var(--space-3);
}

.notable li {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: var(--space-3);
  align-items: start;
  font-size: var(--text-sm);
}

.notable li span {
  display: grid;
}

.notable small {
  color: var(--muted);
}

@media (max-width: 960px) {
  .grid-agreement {
    grid-template-columns: 1fr;
  }
}
</style>
