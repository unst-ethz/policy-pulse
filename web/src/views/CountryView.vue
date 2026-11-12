<script setup lang="ts">
import { PopoverContent, PopoverPortal, PopoverRoot, PopoverTrigger } from 'reka-ui'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import {
  csvUrl,
  type Lang,
  useAgreementBySubject,
  useAgreementMap,
  useCountryIndex,
  useHistoricalStates,
  useMeta,
  useProfile,
  useSponsorship,
  useSubjectTree,
  useTimeline,
} from '@/api/queries'
import AgreementBars from '@/components/AgreementBars.vue'
import ChartFrame from '@/components/ChartFrame.vue'
import CountryList from '@/components/CountryList.vue'
import CountryPicker from '@/components/CountryPicker.vue'
import FacetLayout from '@/components/facets/FacetLayout.vue'
import QueryState from '@/components/QueryState.vue'
import ScaleLegend from '@/components/ScaleLegend.vue'
import TimelineChart from '@/components/TimelineChart.vue'
import type { TimelineSeries } from '@/charts/types'
import VoteBar from '@/components/VoteBar.vue'
import VoteChip from '@/components/VoteChip.vue'
import WorldMap from '@/components/WorldMap.vue'
import YearBars from '@/components/YearBars.vue'
import { divergingScale } from '@/charts/scales'
import { onlyFilters, useFilters } from '@/composables/useFilters'
import { provideReport } from '@/composables/useReport'
import { useThemeColors } from '@/composables/useThemeColors'
import { download, printPage, svgToPng, zip, type ZipEntry } from '@/export'
import { cleanTitle, flatSubjects, sentenceCase, SERIES_SLOTS, SERIES_SLOTS_DARK, yearSpan } from '@/format'

const props = defineProps<{ iso3: string }>()

const { t, n, d, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()
const { index, name } = useCountryIndex()
const colors = useThemeColors()

const country = computed(() => index.value.get(props.iso3))
const range = computed(() => ({ from: filters.value.from, to: filters.value.to }))
// Years and subjects narrow the page, except the timeline (whole history by design) and the
// agreement by topic, which compares the subjects themselves
const scope = computed(() => ({ ...range.value, subjects: filters.value.subjects, historical: filters.value.historical }))
// States that no longer exist stay out of the per-state lists unless asked for (never this one).
// The profile's rankings leave them out on the server; the map and reception lists here.
const historical = useHistoricalStates()
const shown = (iso3: string) => filters.value.historical || iso3 === props.iso3 || !historical.value.has(iso3)
const pageFilters = computed(() => onlyFilters(filters.value, ['from', 'to', 'subjects']))
// What the facets count: the same filters, over the resolutions adopted while it was a member
const applied = computed(() => ({ ...pageFilters.value, country: props.iso3, mode: 'member' as const }))

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
const profile = useProfile(() => props.iso3, scope, requested)
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
const map = useAgreementMap(pageFilters, () => props.iso3)
// Centred by default: most pairs agree 0.7-0.95, which on a fixed 0-1 scale is all one blue.
const centred = ref(true)
const mapScores = computed(() => (map.data.value?.scores ?? []).filter((s) => shown(s.country)))
const scores = computed(() => new Map(mapScores.value.map((s) => [s.country, s.score])))
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
  mapScores.value
    .filter((s) => s.score != null)
    .map((s) => ({ ...s, name: name(s.country) }))
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0)),
)

// --- timeline -----------------------------------------------------------------------------
const specialSessions = ref(false)
const timeline = useTimeline(() => props.iso3, compare, specialSessions)
// Fractional year of an ISO date, so special sessions sit within their year, not on top of it
const fractionalYear = (iso: string) => {
  const d = new Date(`${iso}T00:00:00Z`)
  const start = Date.UTC(d.getUTCFullYear(), 0, 1)
  return d.getUTCFullYear() + (d.getTime() - start) / (365.25 * 86_400_000)
}
const isDark = computed(() => colors.surface.toLowerCase() !== '#ffffff')
const seriesColor = (i: number) => (isDark.value ? SERIES_SLOTS_DARK : SERIES_SLOTS)[i % SERIES_SLOTS.length]!
const seriesColors = computed(() => Object.fromEntries(compare.value.map((c, i) => [c, seriesColor(i)])))
const series = computed<TimelineSeries[]>(() =>
  (timeline.data.value?.compare ?? []).map((c) => ({
    country: c,
    label: name(c),
    color: seriesColors.value[c] ?? seriesColor(0),
    points: (timeline.data.value?.sessions ?? [])
      .map((s) => ({
        x: fractionalYear(s.date),
        year: s.year,
        session: s.session,
        special: s.special,
        agreement: s.values[c]?.agreement ?? null,
        votes: s.values[c]?.shared_votes ?? 0,
      }))
      // A special session too small for a value is left out rather than cutting the line
      .filter((pt) => !(pt.special && pt.agreement == null)),
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

// --- sponsorship ---------------------------------------------------------------------------
const sponsorship = useSponsorship(() => props.iso3, scope)
const sp = computed(() => sponsorship.data.value)
// Printed, the reception table keeps its first rows (the CSS cut-off below matches): in full it runs over pages
const PRINTED_ROWS = 15
const reception = computed(() => (sp.value?.reception ?? []).filter((r) => shown(r.country)))
const years = computed(() => ({
  from: filters.value.from ? String(filters.value.from) : undefined,
  to: filters.value.to ? String(filters.value.to) : undefined,
}))
const sponsoredLink = computed(() => ({
  name: 'resolutions',
  query: { ...years.value, subject: filters.value.subjects, sponsor: props.iso3 },
}))
const domainRows = computed(() => {
  const rows = sp.value?.by_subject ?? []
  const max = Math.max(1, ...rows.map((s) => s.count))
  return rows.map((s) => ({
    key: s.subject_id,
    label: sentenceCase(s.label),
    value: s.count / max,
    display: n(s.count),
    to: { name: 'resolutions', query: { ...years.value, sponsor: props.iso3, subject: s.subject_id } },
  }))
})

// --- sections and report ------------------------------------------------------------------
const SECTIONS = ['overview', 'map', 'timeline', 'topics', 'sponsorship', 'notable'] as const
type Section = (typeof SECTIONS)[number]
// Hidden sections live in the URL like the filters, so a shared link shows (and exports) the same
const hidden = computed(() => new Set(listParam('hide')))
const show = (section: Section) => !hidden.value.has(section)
function toggleSection(section: Section, visible: boolean) {
  setParam('hide', SECTIONS.filter((s) => (s === section ? !visible : hidden.value.has(s))))
}

const meta = useMeta()
const prefix = () => [props.iso3, filters.value.from || filters.value.to ? `${filters.value.from ?? ''}-${filters.value.to ?? ''}` : ''].filter(Boolean).join('_')
const report = provideReport(prefix)

// --- legends of the exported charts (the page's own are HTML, outside the chart) -----------
const mapLegend = computed(() => ({
  items: [{ label: name(props.iso3), color: colors.accent }],
  scale: {
    colors: Array.from({ length: 11 }, (_, i) => scale.value(i / 10)),
    low: t('country.alwaysOpposed'),
    high: t('country.alwaysAgreed'),
    mark:
      centred.value && average.value != null
        ? { at: average.value, label: t('country.average', { name: name(props.iso3), value: n(average.value, 'score') }) }
        : undefined,
    noData: { color: colors.surface, label: t('country.noSharedVote') },
  },
}))
const timelineLegend = computed(() => ({
  items: series.value.map((s) => ({ label: s.label, color: s.color, shape: 'line' as const })),
}))
const sponsorBars = computed(() => [
  { key: 'with_data', label: t('sponsorship.withData'), color: 'rule' as const },
  { key: 'original', label: t('sponsorship.original'), color: 'ink' as const, stack: 'sponsored' },
  { key: 'additional', label: t('sponsorship.additional'), color: 'muted' as const, stack: 'sponsored' },
])
const sponsorLegend = computed(() => ({
  items: sponsorBars.value.map((b) => ({ label: b.label, color: colors[b.color] })),
}))
// The resolutions behind the page: in its years and subjects, adopted while it was a member,
// with its vote and those of the countries it is compared with
const resolutionsUrl = computed(
  () => `${csvUrl(applied.value, compare.value, locale.value as Lang)}&apply_country=true`,
)
// For the printed cover: what the PDF shows, and the link back to it
const tree = useSubjectTree()
const subjectLabels = computed(() => {
  const labels = new Map(flatSubjects(tree.data.value ?? []).map((s) => [s.id, s.label]))
  return filters.value.subjects.map((id) => sentenceCase(labels.get(id) ?? id))
})
const pageUrl = ref('')
const printing = ref(false)
async function exportPdf() {
  printing.value = true
  pageUrl.value = window.location.href
  try {
    await printPage()
  } finally {
    printing.value = false
  }
}

const exporting = ref(false)
const exportError = ref<string | null>(null)

async function downloadReport() {
  exporting.value = true
  exportError.value = null
  try {
    const files: ZipEntry[] = []
    const charts = []
    for (const [id, chart] of report.charts) {
      const written: string[] = []
      const image = await chart.image()
      if (image) {
        const png = new Uint8Array(await (await svgToPng(image.svg, image.width, image.height)).arrayBuffer())
        files.push({ name: `charts/${id}.svg`, data: image.svg }, { name: `charts/${id}.png`, data: png })
        written.push(`charts/${id}.svg`, `charts/${id}.png`)
      }
      const table = chart.csv()
      if (table) {
        files.push({ name: `tables/${id}.csv`, data: table })
        written.push(`tables/${id}.csv`)
      }
      charts.push({ id, title: chart.title(), files: written })
    }
    const response = await fetch(resolutionsUrl.value)
    if (!response.ok) throw new Error(`resolutions.csv: HTTP ${response.status}`)
    files.push({ name: 'resolutions.csv', data: new Uint8Array(await response.arrayBuffer()) })

    const metadata = {
      page: 'country',
      country: { iso3: props.iso3, name: name(props.iso3) },
      url: window.location.href,
      exported_at: new Date().toISOString(),
      data_version: meta.data.value?.data_version ?? null,
      language: locale.value,
      filters: {
        from: filters.value.from,
        to: filters.value.to,
        subjects: filters.value.subjects,
        historical_states: filters.value.historical,
      },
      settings: {
        compare_with: compare.value,
        topics_compared_with: topicWith.value,
        special_sessions: specialSessions.value,
        map_centred_on_average: centred.value,
      },
      sections: SECTIONS.filter(show),
      charts,
      resolutions: {
        file: 'resolutions.csv',
        description:
          'Resolutions in the selected years and subjects adopted while the country was a UN member, with its vote and those of the compared countries. The timeline covers the whole history and the topic chart every subject, whatever the filters.',
        api: new URL(resolutionsUrl.value, window.location.href).href,
      },
    }
    files.push({ name: 'metadata.json', data: JSON.stringify(metadata, null, 2) })
    download(zip(files), `policy-pulse_${prefix() || props.iso3}.zip`)
  } catch (error) {
    exportError.value = error instanceof Error ? error.message : String(error)
  } finally {
    exporting.value = false
  }
}

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
      <div class="page-actions no-print">
        <PopoverRoot>
          <PopoverTrigger class="button" :title="t('report.sectionsHint')">
            <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
              <path d="M2 4h7M13 4h1M2 12h1M7 12h7" stroke="currentColor" stroke-width="1.5" fill="none" />
              <circle cx="11" cy="4" r="2" stroke="currentColor" stroke-width="1.5" fill="none" />
              <circle cx="5" cy="12" r="2" stroke="currentColor" stroke-width="1.5" fill="none" />
            </svg>
            {{ t('report.customize') }}
          </PopoverTrigger>
          <PopoverPortal>
            <PopoverContent class="facet-chooser" side="bottom" align="end" :side-offset="6" :collision-padding="12">
              <p class="facet-chooser-title">{{ t('report.sections') }}</p>
              <label v-for="section in SECTIONS" :key="section" class="facet-chooser-option">
                <input
                  type="checkbox"
                  :checked="show(section)"
                  @change="toggleSection(section, ($event.target as HTMLInputElement).checked)"
                />
                {{ t(`report.${section}`) }}
              </label>
              <p class="facet-chooser-note">{{ t('report.sectionsHint') }}</p>
            </PopoverContent>
          </PopoverPortal>
        </PopoverRoot>
        <button type="button" class="button" :disabled="printing" :title="t('report.pdfHint')" @click="exportPdf">
          {{ printing ? t('report.preparing') : t('report.pdf') }}
        </button>
        <button
          type="button"
          class="button primary"
          :disabled="exporting"
          :title="t('report.contents')"
          @click="downloadReport"
        >
          {{ exporting ? t('report.preparing') : t('report.download') }}
        </button>
        <p v-if="exportError" class="error" role="alert">{{ t('report.failed', { error: exportError }) }}</p>
      </div>
      <h1>{{ country?.name ?? iso3 }}</h1>
      <p class="sub">
        <span v-if="country?.subregion">{{ country.subregion }}</span>
        <span v-if="membershipText">{{ membershipText }}</span>
        <span v-if="country?.historical_names.length">
          {{ t('countries.formerly', { names: country.historical_names.join(', ') }) }}
        </span>
      </p>
      <dl class="print-only cover">
        <dt>{{ t('report.years') }}</dt>
        <dd>{{ filters.from || filters.to ? `${filters.from ?? '…'}–${filters.to ?? '…'}` : t('report.allYears') }}</dd>
        <dt>{{ t('report.subjects') }}</dt>
        <dd>{{ subjectLabels.length ? subjectLabels.join('; ') : t('report.allSubjects') }}</dd>
        <dt>{{ t('report.comparedWith') }}</dt>
        <dd>{{ compare.map(name).join(', ') || '–' }}</dd>
        <dt>{{ t('report.topicsWith') }}</dt>
        <dd>{{ topicWith ? name(topicWith) : '–' }}</dd>
        <dt>{{ t('report.dataAsOf') }}</dt>
        <dd>{{ meta.data.value?.data_version.slice(0, 10) ?? '?' }}</dd>
        <dt>{{ t('report.link') }}</dt>
        <dd class="link">{{ pageUrl }}</dd>
      </dl>
    </div>

    <FacetLayout
      :allowed="['years', 'subjects', 'historical']"
      :defaults="['years', 'historical']"
      storage-key="policy-pulse:facets:country"
      :applied="applied"
    >

    <QueryState
      v-if="show('overview')"
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
        <div class="tally">
          <template v-if="p.recorded_votes">
            <p class="hint">{{ t('country.recordedVotes', { name: name(iso3), n: n(p.recorded_votes) }) }}</p>
            <VoteBar
              :yes="p.votes.Y ?? 0"
              :no="p.votes.N ?? 0"
              :abstain="p.votes.A ?? 0"
              :absent="p.votes.absent ?? 0"
            />
          </template>
          <p v-if="!p.recorded_votes" class="hint">{{ t('country.noRecordedVotes') }}</p>
          <p v-else-if="p.total_resolutions > p.recorded_votes" class="hint">
            {{
              t('country.notRecorded', {
                n: n(p.total_resolutions - p.recorded_votes),
                without: n(p.adopted_without_vote),
                unrecorded: n(p.non_recorded_votes),
              })
            }}
          </p>
        </div>
      </section>
    </QueryState>

    <section v-if="show('map')" class="section">
      <div class="grid-agreement">
        <ChartFrame id="map" :legend="mapLegend" :title="t('country.mapTitle', { name: name(iso3) })" :description="t('country.mapText')">
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
                  <td class="numeric" :data-value="row.score ?? ''">{{ n(row.score ?? 0, 'score') }}</td>
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

    <ChartFrame
      v-if="show('timeline')"
      id="timeline"
      :export-title="t('country.timelineExportTitle', { name: name(iso3) })"
      :legend="timelineLegend"
      :title="t('country.timelineTitle')"
      :description="t('country.timelineText', { name: name(iso3) })"
    >
      <template #controls>
        <label class="check">
          <input v-model="specialSessions" type="checkbox" />
          {{ t('country.specialSessions') }}
        </label>
      </template>
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
              <td
                v-for="s in series"
                :key="s.country"
                class="numeric"
                :data-value="session.values[s.country]?.agreement ?? ''"
              >
                {{ session.values[s.country]?.agreement == null ? '' : n(session.values[s.country]!.agreement!, 'score') }}
              </td>
            </tr>
          </tbody>
        </table>
      </template>
      <template #note>
        <p>
          {{ t('country.timelineNote', { n: timeline.data.value?.min_votes ?? 3 }) }}
          <template v-if="!specialSessions">{{ t('country.specialNote') }}</template>
        </p>
      </template>
    </ChartFrame>

    <ChartFrame
      v-if="show('topics')"
      id="topics"
      :title="t('country.topicsTitle')"
      :description="t('country.topicsText')"
    >
      <template #controls>
        <CountryPicker v-model="topicWith" :label="t('country.compareWith')" :exclude="[iso3]" class="with" />
      </template>
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
      <template #table>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">{{ t('report.subjectCol') }}</th>
              <th scope="col" class="numeric">{{ t('country.agreementCol') }}</th>
              <th scope="col" class="numeric">{{ t('report.sharedVotesCol') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="s in bySubject.data.value?.subjects ?? []" :key="s.subject_id">
              <td>{{ sentenceCase(s.label) }}</td>
              <td class="numeric" :data-value="s.agreement ?? ''">{{ s.agreement == null ? '' : n(s.agreement, 'score') }}</td>
              <td class="numeric" :data-value="s.shared_votes">{{ n(s.shared_votes) }}</td>
            </tr>
          </tbody>
        </table>
      </template>
    </ChartFrame>

    <section v-if="show('sponsorship')" class="section">
      <div class="section-head">
        <div>
          <h2>{{ t('sponsorship.title') }}</h2>
          <p>{{ t('sponsorship.lead') }}</p>
        </div>
      </div>
      <QueryState
        :loading="sponsorship.isLoading.value"
        :fetching="sponsorship.isFetching.value"
        :error="sponsorship.error.value"
        min-height="200px"
        @retry="sponsorship.refetch()"
      >
        <div v-if="sp" class="section">
          <div class="panel overview">
            <div class="figures">
              <div>
                <span class="figure">{{ n(sp.sponsored) }}</span>
                <span class="caption">{{ t('sponsorship.sponsoredOf', { n: n(sp.with_data) }) }}</span>
              </div>
              <div>
                <span class="figure">{{ n(sp.original) }}</span>
                <span class="caption">{{ t('sponsorship.asOriginal') }}</span>
              </div>
            </div>
            <p class="hint">
              {{ t('sponsorship.coverage', { n: n(sp.with_data), total: n(sp.resolutions) }) }}
            </p>
            <p>
              <RouterLink :to="sponsoredLink">{{ t('sponsorship.list', { name: name(iso3) }) }}</RouterLink>
            </p>
          </div>

          <ChartFrame
            id="sponsorship"
            :export-title="t('sponsorship.chartExportTitle', { name: name(iso3) })"
            :legend="sponsorLegend" :title="t('sponsorship.chartTitle')" :description="t('sponsorship.chartText', { name: name(iso3) })">
            <YearBars
              :years="sp.per_year"
              :label="t('sponsorship.chartTitle')"
              :bars="sponsorBars"
            />
            <template #table>
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">{{ t('country.yearCol') }}</th>
                    <th scope="col" class="numeric">{{ t('sponsorship.original') }}</th>
                    <th scope="col" class="numeric">{{ t('sponsorship.additional') }}</th>
                    <th scope="col" class="numeric">{{ t('sponsorship.withData') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="y in sp.per_year" :key="y.year">
                    <td>{{ y.year }}</td>
                    <td class="numeric" :data-value="y.original">{{ n(y.original) }}</td>
                    <td class="numeric" :data-value="y.additional">{{ n(y.additional) }}</td>
                    <td class="numeric" :data-value="y.with_data">{{ n(y.with_data) }}</td>
                  </tr>
                </tbody>
              </table>
            </template>
            <template #note>
              <p>{{ t('sponsorship.chartNote') }}</p>
            </template>
          </ChartFrame>

          <div class="grid-2">
            <div class="panel notable">
              <h3>{{ t('sponsorship.domains') }}</h3>
              <p class="hint">{{ t('sponsorship.domainsText') }}</p>
              <AgreementBars v-if="domainRows.length" :rows="domainRows" />
              <p v-else class="hint">{{ t('sponsorship.none') }}</p>
            </div>
            <div class="panel notable">
              <h3>{{ t('sponsorship.reception') }}</h3>
              <p class="hint">{{ t('sponsorship.receptionText', { name: name(iso3) }) }}</p>
              <div v-if="reception.length" class="reception">
                <table class="data-table">
                  <thead>
                    <tr>
                      <th scope="col">{{ t('country.country') }}</th>
                      <th scope="col" class="numeric">{{ t('vote.N') }}</th>
                      <th scope="col" class="numeric">{{ t('vote.A') }}</th>
                      <th scope="col" class="numeric">{{ t('vote.Y') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in reception" :key="row.country">
                      <td>
                        <RouterLink :to="{ name: 'country', params: { iso3: row.country }, query: route.query }">
                          {{ name(row.country) }}
                        </RouterLink>
                      </td>
                      <td class="numeric">{{ n(row.no) }}</td>
                      <td class="numeric">{{ n(row.abstain) }}</td>
                      <td class="numeric">{{ n(row.yes) }}</td>
                    </tr>
                  </tbody>
                </table>
                <p v-if="reception.length > PRINTED_ROWS" class="print-only hint">
                  {{ t('report.printedRows', { n: PRINTED_ROWS, total: reception.length }) }}
                </p>
              </div>
              <p v-else class="hint">{{ t('sponsorship.none') }}</p>
            </div>
          </div>
        </div>
      </QueryState>
    </section>

    <section v-if="p && show('notable')" class="section">
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
        <div v-if="sp" class="panel notable">
          <h3>{{ t('sponsorship.notYes') }}</h3>
          <p class="hint">{{ t('sponsorship.notYesText') }}</p>
          <ol v-if="sp.not_yes.length">
            <li v-for="v in sp.not_yes" :key="v.undl_id">
              <VoteChip :vote="v.vote" />
              <span>
                <RouterLink :to="{ name: 'resolution', params: { id: v.undl_id } }">{{ cleanTitle(v.title) }}</RouterLink>
                <small>{{ v.symbol }}, {{ t('country.consensusOf', { value: n(v.consensus_score, 'score') }) }}</small>
              </span>
            </li>
          </ol>
          <p v-else class="hint">{{ t('sponsorship.notYesNone') }}</p>
        </div>
      </div>
    </section>
    </FacetLayout>
  </div>
</template>

<style scoped>
.page-head {
  grid-template-columns: minmax(0, 1fr) auto;
}

.page-head > :not(.page-actions) {
  grid-column: 1;
}

.page-actions {
  grid-column: 2;
  grid-row: 1;
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  align-items: center;
  gap: var(--space-2);
}

/* Shown only when printing (.print-only); laid out as a grid then */
.cover {
  grid-template-columns: max-content minmax(0, 1fr);
  gap: var(--space-1) var(--space-4);
  margin: var(--space-3) 0 0;
  padding-top: var(--space-3);
  border-top: 1px solid var(--rule);
  font-size: var(--text-sm);
}

html.printing .cover {
  display: grid;
}

.cover dt {
  color: var(--muted);
}

.cover dd {
  margin: 0;
}

.cover .link {
  overflow-wrap: anywhere;
}

.page-actions .error {
  flex-basis: 100%;
  text-align: end;
  color: var(--danger-ink);
  font-size: var(--text-sm);
}
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

.tally {
  display: grid;
  gap: var(--space-2);
}

.tally .hint {
  margin: 0;
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

.reception {
  max-height: 360px;
  overflow: auto;
}

html.printing .reception {
  max-height: none;
  overflow: visible;
}

/* After PRINTED_ROWS */
html.printing .reception tbody tr:nth-child(n + 16) {
  display: none;
}

.reception thead th {
  position: sticky;
  top: 0;
  background: var(--surface);
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
