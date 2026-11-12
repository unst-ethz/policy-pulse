<script setup lang="ts">
/**
 * One area of the UNBIS Thesaurus: how much the Assembly adopts on it, who sponsors it, how states
 * vote on it and which of its resolutions split the Assembly most. Only the years filter applies;
 * states that no longer exist are left out of the per-country lists unless the viewer asks.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import {
  useCountryIndex,
  useDomains,
  useHistoricalStates,
  useMultilateral,
  useResolutions,
  useSubjectSponsors,
} from '@/api/queries'
import AgreementBars from '@/components/AgreementBars.vue'
import ChartFrame from '@/components/ChartFrame.vue'
import FacetLayout from '@/components/facets/FacetLayout.vue'
import QueryState from '@/components/QueryState.vue'
import ResolutionTable from '@/components/ResolutionTable.vue'
import ScaleLegend from '@/components/ScaleLegend.vue'
import WorldMap from '@/components/WorldMap.vue'
import YearBars from '@/components/YearBars.vue'
import { divergingScale } from '@/charts/scales'
import { type Filters, onlyFilters, serializeFilters, useFilters } from '@/composables/useFilters'
import { provideReport } from '@/composables/useReport'
import { useThemeColors } from '@/composables/useThemeColors'
import { slug } from '@/export'
import { sentenceCase } from '@/format'

// A state's vote shares count only above this many votes cast (as on the Alignment page)
const MIN_VOTES = 10
// "Most focused" ranks states by share, which means little for a handful of sponsorships
const FOCUS_MIN = 10
// Below this many recorded votes, the vote shares get a warning
const THIN = 30
const TOP = 10

const props = defineProps<{ id: string }>()
const { t, n } = useI18n()
const router = useRouter()
const { filters } = useFilters()
const { name } = useCountryIndex()
const colors = useThemeColors()

const range = computed(() => ({ from: filters.value.from, to: filters.value.to }))
const yearsOnly = computed<Filters>(() => onlyFilters(filters.value, ['from', 'to']))
const yearsQuery = computed(() => serializeFilters(yearsOnly.value))
// Back to the overview, keeping the historical-states choice as well
const overviewQuery = computed(() => serializeFilters(onlyFilters(filters.value, ['from', 'to', 'historical'])))

// States that no longer exist (the region tree's "historical" branch), left out unless asked for.
// Dropping their rows is enough: every figure per state here (vote shares, sponsorships) is the
// state's own, and the map's midpoint is the mean over the rows shown.
const historical = useHistoricalStates()
const shown = (iso3: string) => filters.value.historical || !historical.value.has(iso3)
const selection = computed<Filters>(() => ({ ...yearsOnly.value, subjects: [props.id] }))
const listLink = computed(() => ({ name: 'resolutions', query: serializeFilters(selection.value) }))

// --- headline and volume ---------------------------------------------------------------------
const domains = useDomains(range)
const sponsors = useSubjectSponsors(() => props.id, range)
const domain = computed(() => domains.data.value?.domains.find((d) => d.subject_id === props.id))
const label = computed(() => sentenceCase(sponsors.data.value?.label ?? domain.value?.label ?? props.id))
const total = computed(() => domains.data.value?.resolutions ?? 0)
const count = computed(() => domain.value?.resolutions ?? 0)
const recorded = computed(() => domain.value?.recorded ?? 0)

const showShare = ref(false)
const volume = computed(() => {
  const own = new Map((domain.value?.per_year ?? []).map((y) => [y.year, y]))
  const first = domain.value?.per_year[0]?.year
  return (domains.data.value?.per_year ?? [])
    .filter((y) => first != null && y.year >= first)
    .map((all) => {
      const y = own.get(all.year)
      return {
        year: all.year,
        recorded: y?.recorded ?? 0,
        other: (y?.resolutions ?? 0) - (y?.recorded ?? 0),
        resolutions: y?.resolutions ?? 0,
        share: all.resolutions ? (y?.resolutions ?? 0) / all.resolutions : 0,
      }
    })
})
const volumeBars = computed(() =>
  showShare.value
    ? [{ key: 'share', label: t('subject.shareOfYear'), color: 'ink' as const, stack: 'all' }]
    : [
        { key: 'recorded', label: t('subject.recordedVote'), color: 'ink' as const, stack: 'all' },
        { key: 'other', label: t('subject.withoutVote'), color: 'muted' as const, stack: 'all' },
      ],
)
const percent = (value: number) => n(value, 'percent')

// --- sponsors --------------------------------------------------------------------------------
const countryLink = (iso3: string) => ({ name: 'country', params: { iso3 }, query: yearsQuery.value })
const sponsorRows = computed(() => (sponsors.data.value?.sponsors ?? []).filter((s) => shown(s.country)))
const mostActive = computed(() => {
  const rows = sponsorRows.value.slice(0, TOP)
  const max = Math.max(1, ...rows.map((s) => s.sponsored))
  return rows.map((s) => ({
    key: s.country,
    label: name(s.country),
    value: s.sponsored / max,
    display: n(s.sponsored),
    to: countryLink(s.country),
  }))
})
const focused = computed(() => {
  const rows = sponsorRows.value
    .filter((s) => s.sponsored >= FOCUS_MIN)
    .map((s) => ({ ...s, share: s.sponsored / s.own_total }))
    .sort((a, b) => b.share - a.share)
    .slice(0, TOP)
  // Shares are small (a few percent): bars relative to the top one, the share as the value
  const max = rows[0]?.share || 1
  return rows.map((s) => ({
      key: s.country,
      label: name(s.country),
      value: s.share / max,
      display: n(s.share, 'percentShort'),
      detail: t('subject.ofOwn', { n: n(s.sponsored), total: n(s.own_total) }),
      to: countryLink(s.country),
    }))
})

// --- votes -----------------------------------------------------------------------------------
const votes = useMultilateral(selection, MIN_VOTES)
const voters = computed(() => (votes.data.value?.countries ?? []).filter((c) => shown(c.country)))
const byCountry = computed(() => new Map(voters.value.map((c) => [c.country, c])))
const averageYes = computed(() => {
  const rates = voters.value.map((c) => c.yes_rate).filter((r): r is number => r != null)
  return rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null
})
const scale = computed(() => divergingScale(colors.no, colors['scale-mid'], colors.yes, [0, 1], averageYes.value ?? 0.5))
const fill = (iso3: string) => {
  const rate = byCountry.value.get(iso3)?.yes_rate
  return rate == null ? null : scale.value(rate)
}
const mapTooltip = (iso3: string) => {
  const c = byCountry.value.get(iso3)
  if (!c) return `<b>${name(iso3)}</b><br>${t('subject.fewVotes', { n: MIN_VOTES })}`
  return `<b>${name(iso3)}</b><br>${t('subject.mapTooltip', {
    yes: n(c.yes_rate ?? 0, 'percentShort'),
    no: n(c.no_rate ?? 0, 'percentShort'),
    abstain: n(c.abstention_rate ?? 0, 'percentShort'),
    n: n(c.votes_cast),
  })}`
}
function ranked(key: 'no_rate' | 'abstention_rate') {
  return voters.value
    .filter((c) => (c[key] ?? 0) > 0)
    .sort((a, b) => (b[key] ?? 0) - (a[key] ?? 0) || b.votes_cast - a.votes_cast)
    .slice(0, TOP)
    .map((c) => ({
      key: c.country,
      label: name(c.country),
      value: c[key] ?? 0,
      display: n(c[key] ?? 0, 'percentShort'),
      detail: t('subject.votesCast') + ': ' + n(c.votes_cast),
      to: countryLink(c.country),
    }))
}
const mostNo = computed(() => ranked('no_rate'))
const mostAbstain = computed(() => ranked('abstention_rate'))
const tableRows = computed(() => [...voters.value].sort((a, b) => (a.yes_rate ?? 0) - (b.yes_rate ?? 0)))

// --- exports ------------------------------------------------------------------------------
// File names and export titles name the subject: an exported chart has no page around it
const years = computed(() =>
  filters.value.from || filters.value.to ? `${filters.value.from ?? ''}-${filters.value.to ?? ''}` : '',
)
provideReport(() => [slug(label.value), years.value].filter(Boolean).join('_'))
const volumeLegend = computed(() => ({
  items: volumeBars.value.map((b) => ({ label: b.label, color: colors[b.color] })),
}))
const mapLegend = computed(() => ({
  scale: {
    colors: Array.from({ length: 11 }, (_, i) => scale.value(i / 10)),
    low: t('subject.neverYes'),
    high: t('subject.alwaysYes'),
    mark:
      averageYes.value != null
        ? { at: averageYes.value, label: t('subject.average', { value: n(averageYes.value, 'percentShort') }) }
        : undefined,
    noData: { color: colors.surface, label: t('subject.fewVotes', { n: MIN_VOTES }) },
  },
}))

// --- contested -------------------------------------------------------------------------------
const contested = useResolutions(selection, () => ({ sort: 'consensus_asc', offset: 0, limit: TOP, votes: [] }))
const contestedItems = computed(() => (contested.data.value?.items ?? []).filter((r) => r.consensus_score != null))
const contestedLink = computed(() => ({ ...listLink.value, query: { ...listLink.value.query, sort: 'consensus_asc' } }))
</script>

<template>
  <div class="page">
    <div class="page-head">
      <p class="crumb">
        <RouterLink :to="{ name: 'subjects', query: overviewQuery }">{{ t('subject.all') }}</RouterLink>
      </p>
      <h1>{{ label }}</h1>
      <p>{{ t('subject.detailLead') }}</p>
    </div>

    <FacetLayout
      :allowed="['years', 'historical']"
      :defaults="['years', 'historical']"
      storage-key="policy-pulse:facets:subject"
      :applied="selection"
    >
    <QueryState
      :loading="domains.isLoading.value || sponsors.isLoading.value"
      :fetching="domains.isFetching.value"
      :error="sponsors.error.value ?? domains.error.value"
      min-height="120px"
      @retry="domains.refetch(), sponsors.refetch()"
    >
      <section class="panel overview">
        <div class="figures">
          <div>
            <span class="figure">{{ n(count) }}</span>
            <span class="caption">{{ t('subject.resolutions') }}</span>
          </div>
          <div v-if="total">
            <span class="figure">{{ n(count / total, 'percent') }}</span>
            <span class="caption">{{ t('subject.shareOfAll') }}</span>
          </div>
          <div>
            <span class="figure">{{ n(recorded) }}</span>
            <span class="caption">{{ t('subject.recorded') }}</span>
          </div>
        </div>
        <p><RouterLink :to="listLink">{{ t('subject.list') }}</RouterLink></p>
      </section>
    </QueryState>

    <ChartFrame
      v-if="volume.length"
      id="volume"
      :export-title="t(showShare ? 'subject.volumeShareExportTitle' : 'subject.volumeExportTitle', { subject: label })"
      :legend="volumeLegend"
      :title="t('subject.volumeTitle')"
      :description="showShare ? t('subject.volumeShareText') : t('subject.volumeText')"
    >
      <template #controls>
        <div class="modes" role="group" :aria-label="t('subject.show')">
          <button type="button" :aria-pressed="!showShare" @click="showShare = false">{{ t('subject.counts') }}</button>
          <button type="button" :aria-pressed="showShare" @click="showShare = true">{{ t('subject.shares') }}</button>
        </div>
      </template>
      <YearBars
        :years="volume"
        :bars="volumeBars"
        :label="t('subject.volumeTitle')"
        :format="showShare ? percent : undefined"
      />
      <template #table>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">{{ t('subject.yearCol') }}</th>
              <th scope="col" class="numeric">{{ t('subject.recordedVote') }}</th>
              <th scope="col" class="numeric">{{ t('subject.withoutVote') }}</th>
              <th scope="col" class="numeric">{{ t('subject.shareOfYear') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="y in volume" :key="y.year">
              <td>{{ y.year }}</td>
              <td class="numeric" :data-value="y.recorded">{{ n(y.recorded) }}</td>
              <td class="numeric" :data-value="y.other">{{ n(y.other) }}</td>
              <td class="numeric" :data-value="y.share">{{ n(y.share, 'percent') }}</td>
            </tr>
          </tbody>
        </table>
      </template>
      <template #note>
        <p>{{ t('subject.volumeNote') }}</p>
      </template>
    </ChartFrame>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>{{ t('subject.sponsorsTitle') }}</h2>
          <p v-if="sponsors.data.value">
            {{ t('subject.sponsorsLead', { n: n(sponsors.data.value.with_data), total: n(sponsors.data.value.resolutions) }) }}
          </p>
        </div>
      </div>
      <div v-if="sponsorRows.length" class="grid-2">
        <div class="panel notable">
          <h3>{{ t('subject.mostActive') }}</h3>
          <p class="hint">{{ t('subject.mostActiveText') }}</p>
          <AgreementBars :rows="mostActive" />
        </div>
        <div class="panel notable">
          <h3>{{ t('subject.focused') }}</h3>
          <p class="hint">{{ t('subject.focusedText', { n: FOCUS_MIN }) }}</p>
          <AgreementBars v-if="focused.length" :rows="focused" />
          <p v-else class="hint">{{ t('subject.focusedNone', { n: FOCUS_MIN }) }}</p>
        </div>
      </div>
      <p v-else-if="sponsors.data.value" class="hint">{{ t('subject.noSponsors') }}</p>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>{{ t('subject.votesTitle') }}</h2>
          <p v-if="recorded">{{ t('subject.votesLead', { n: n(recorded) }) }}</p>
        </div>
      </div>
      <p v-if="recorded && recorded < THIN" class="warning">{{ t('subject.thin', { n: recorded }) }}</p>
      <p v-if="domains.data.value && !recorded" class="hint">{{ t('subject.noVotes') }}</p>
      <template v-else-if="recorded">
        <p v-if="votes.data.value && !voters.length" class="hint">{{ t('subject.minVotes', { n: MIN_VOTES }) }}</p>
        <ChartFrame
          v-if="voters.length || !votes.data.value"
          id="map"
          :export-title="t('subject.mapExportTitle', { subject: label })"
          :legend="mapLegend"
          :title="t('subject.mapTitle')" :description="t('subject.mapText')">
          <QueryState
            :loading="votes.isLoading.value"
            :fetching="votes.isFetching.value"
            :error="votes.error.value"
            @retry="votes.refetch()"
          >
            <WorldMap
              :fill="fill"
              :tooltip="mapTooltip"
              @select="(c) => router.push(countryLink(c))"
            />
          </QueryState>
          <ScaleLegend
            :color="scale"
            :low="t('subject.neverYes')"
            :high="t('subject.alwaysYes')"
            :center="averageYes != null ? { value: averageYes, label: t('subject.average', { value: n(averageYes, 'percentShort') }) } : null"
            :no-data="t('subject.fewVotes', { n: MIN_VOTES })"
          />
          <template #table>
            <table class="data-table">
              <thead>
                <tr>
                  <th scope="col">{{ t('subject.country') }}</th>
                  <th scope="col" class="numeric">{{ t('vote.Y') }}</th>
                  <th scope="col" class="numeric">{{ t('vote.N') }}</th>
                  <th scope="col" class="numeric">{{ t('vote.A') }}</th>
                  <th scope="col" class="numeric">{{ t('subject.votesCast') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="c in tableRows" :key="c.country">
                  <td><RouterLink :to="countryLink(c.country)">{{ name(c.country) }}</RouterLink></td>
                  <td class="numeric" :data-value="c.yes_rate ?? ''">{{ n(c.yes_rate ?? 0, 'percentShort') }}</td>
                  <td class="numeric" :data-value="c.no_rate ?? ''">{{ n(c.no_rate ?? 0, 'percentShort') }}</td>
                  <td class="numeric" :data-value="c.abstention_rate ?? ''">{{ n(c.abstention_rate ?? 0, 'percentShort') }}</td>
                  <td class="numeric" :data-value="c.votes_cast">{{ n(c.votes_cast) }}</td>
                </tr>
              </tbody>
            </table>
          </template>
          <template #note>
            <p>{{ t('subject.minVotes', { n: MIN_VOTES }) }} {{ t('map.disclaimer') }}</p>
          </template>
        </ChartFrame>

        <div v-if="voters.length" class="grid-2">
          <div class="panel notable">
            <h3>{{ t('subject.mostNo') }}</h3>
            <p class="hint">{{ t('subject.mostNoText') }}</p>
            <AgreementBars :rows="mostNo" />
          </div>
          <div class="panel notable">
            <h3>{{ t('subject.mostAbstain') }}</h3>
            <p class="hint">{{ t('subject.mostAbstainText') }}</p>
            <AgreementBars :rows="mostAbstain" />
          </div>
        </div>

        <section class="section">
          <div class="section-head">
            <div>
              <h2>{{ t('subject.contestedTitle') }}</h2>
              <p>{{ t('subject.contestedText') }}</p>
            </div>
            <RouterLink :to="contestedLink">{{ t('subject.contestedAll') }}</RouterLink>
          </div>
          <div class="panel flush">
            <QueryState
              :loading="contested.isLoading.value"
              :fetching="contested.isFetching.value"
              :error="contested.error.value"
              @retry="contested.refetch()"
            >
              <ResolutionTable :items="contestedItems" />
            </QueryState>
          </div>
        </section>
      </template>
    </section>
    </FacetLayout>
  </div>
</template>

<style scoped>
.crumb {
  font-size: var(--text-sm);
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

.warning {
  font-size: var(--text-sm);
  color: var(--ink-2);
  border-inline-start: 3px solid var(--rule-strong);
  padding-inline-start: var(--space-3);
}

.notable {
  display: grid;
  gap: var(--space-2);
  align-content: start;
}

.notable h3 {
  font-size: var(--text-lg);
}

.flush {
  padding: 0;
  overflow: hidden;
}

.modes {
  display: inline-flex;
  gap: 2px;
  padding: 2px;
  background: var(--surface-sunk);
  border: 1px solid var(--rule);
  border-radius: var(--radius-s);
}

.modes button {
  border: 0;
  background: transparent;
  padding: 4px 10px;
  border-radius: 2px;
  font-size: var(--text-xs);
  color: var(--ink-2);
  cursor: pointer;
}

.modes button[aria-pressed='true'] {
  background: var(--surface);
  color: var(--ink);
  box-shadow: 0 0 0 1px var(--rule-strong);
  font-weight: 500;
}
</style>
