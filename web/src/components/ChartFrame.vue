<script setup lang="ts">
/**
 * A chart with its heading, a "how to read this" line, and a table twin. The table carries every
 * value the chart shows, so nothing depends on hovering or on colour vision.
 *
 * Both download: an ECharts chart as SVG or PNG, the table as CSV. With an `id`, the frame also
 * registers with the page's report (useReport), if the page keeps one.
 */
import { PopoverContent, PopoverPortal, PopoverRoot, PopoverTrigger } from 'reka-ui'
import { ref, useSlots, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMeta } from '@/api/queries'
import { type ChartImage, useReportChart } from '@/composables/useReport'
import { type ChartLegend, composeChart, download, slug, svgToPng, tableCsv } from '@/export'

const props = withDefaults(
  defineProps<{
    title: string
    description?: string
    id?: string
    /** The title of the exported image when it needs more context than the page gives, e.g. the country. */
    exportTitle?: string
    /** What the exported image shows as its legend (the page's own legend is HTML). */
    legend?: ChartLegend
  }>(),
  { description: undefined, id: undefined, exportTitle: undefined, legend: undefined },
)
const view = ref<'chart' | 'table'>('chart')
const { t } = useI18n()
const slots = useSlots()
const meta = useMeta()

const frame = ref<HTMLElement>()
const chartBody = ref<HTMLElement>()
const tableBody = ref<HTMLElement>()

// vue-echarts marks its element; the instance is ECharts' own, so no chart needs to expose it
const chartElement = () => chartBody.value?.querySelector<HTMLElement>('[_echarts_instance_]') ?? null
const exportTitle = () => props.exportTitle ?? props.title

async function chartImage(): Promise<ChartImage | null> {
  const el = chartElement()
  if (!el) return null
  const { echarts } = await import('@/charts/echarts')
  const chart = echarts.getInstanceByDom(el)
  if (!chart) return null
  const style = getComputedStyle(frame.value!)
  const token = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback
  return composeChart(
    { svg: chart.renderToSVGString(), width: chart.getWidth(), height: chart.getHeight() },
    {
      title: exportTitle(),
      legend: props.legend,
      // Where the figure comes from: the data's date, and the link that reproduces it
      footer: t('report.footer', {
        date: meta.data.value?.data_version.slice(0, 10) ?? '?',
        url: window.location.href,
      }),
      // The page's theme; the background matters, as ECharts draws on a transparent canvas
      colors: {
        background: token('--surface', '#ffffff'),
        ink: token('--ink', '#172033'),
        muted: token('--muted', '#646d7e'),
        rule: token('--rule-strong', '#c3c9d3'),
      },
      font: token('--font-sans', 'sans-serif'),
    },
  )
}

function csv(): string | null {
  const table = tableBody.value?.querySelector('table')
  return table ? tableCsv(table) : null
}

const report = useReportChart(props.id, { title: exportTitle, image: chartImage, csv })
const filename = (ext: string) =>
  `${[report?.prefix(), props.id ?? slug(props.title)].filter(Boolean).join('_')}.${ext}`

const exportOpen = ref(false)
// Image formats only for a chart ECharts drew (not an HTML one, nor one still loading)
const hasChart = ref(false)
watch(exportOpen, (open) => open && (hasChart.value = !!chartElement()))

async function save(kind: 'svg' | 'png' | 'csv') {
  exportOpen.value = false
  if (kind === 'csv') {
    const data = csv()
    if (data) download(data, filename('csv'))
    return
  }
  const img = await chartImage()
  if (!img) return
  if (kind === 'svg') download(img.svg, filename('svg'), 'image/svg+xml')
  else download(await svgToPng(img.svg, img.width, img.height), filename('png'))
}
</script>

<template>
  <section ref="frame" class="frame">
    <div class="head">
      <div class="head-main">
        <div class="titles">
          <h2>{{ title }}</h2>
          <p v-if="description" class="description">{{ description }}</p>
        </div>
        <div class="controls">
          <slot name="controls" />
          <div class="toggle no-print" role="group" :aria-label="t('chart.view')">
            <button
              type="button"
              :aria-pressed="view === 'chart'"
              @click="view = 'chart'"
            >
              {{ t('chart.chart') }}
            </button>
            <button
              type="button"
              :aria-pressed="view === 'table'"
              @click="view = 'table'"
            >
              {{ t('chart.table') }}
            </button>
          </div>
        </div>
      </div>
      <PopoverRoot v-model:open="exportOpen">
        <PopoverTrigger class="export no-print" :aria-label="t('chart.download')" :title="t('chart.download')">
          <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
            <path d="M8 2v8M4.5 6.5 8 10l3.5-3.5M2.5 13.5h11" stroke="currentColor" stroke-width="1.5" fill="none" />
          </svg>
        </PopoverTrigger>
        <PopoverPortal>
          <PopoverContent class="facet-chooser" side="bottom" align="end" :side-offset="6" :collision-padding="12">
            <p class="facet-chooser-title">{{ t('chart.download') }}</p>
            <template v-if="hasChart">
              <button type="button" class="facet-chooser-option export-option" @click="save('svg')">
                {{ t('chart.svg') }}
              </button>
              <button type="button" class="facet-chooser-option export-option" @click="save('png')">
                {{ t('chart.png') }}
              </button>
            </template>
            <button v-if="slots.table" type="button" class="facet-chooser-option export-option" @click="save('csv')">
              {{ t('chart.csv') }}
            </button>
          </PopoverContent>
        </PopoverPortal>
      </PopoverRoot>
    </div>
    <div v-show="view === 'chart'" ref="chartBody" class="body"><slot /></div>
    <!-- Mounted while hidden too, so the CSV download reads it in either view -->
    <div v-show="view === 'table'" ref="tableBody" class="body table-body"><slot name="table" /></div>
    <div v-if="$slots.note" class="note"><slot name="note" /></div>
  </section>
</template>

<style scoped>
.frame {
  display: grid;
  gap: var(--space-4);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius-m);
  padding: var(--space-5);
  min-width: 0;
}

/* The export button stays top right, however the titles and controls wrap */
.head {
  display: flex;
  align-items: flex-start;
  gap: var(--space-3);
}

.head-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3) var(--space-5);
}

.titles {
  display: grid;
  gap: var(--space-1);
  max-width: 64ch;
}

.description {
  color: var(--ink-2);
  font-size: var(--text-sm);
}

.controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3);
}

.toggle {
  display: inline-flex;
  border: 1px solid var(--rule-strong);
  border-radius: var(--radius-s);
  overflow: hidden;
}

.toggle button {
  border: 0;
  background: var(--surface);
  padding: 4px 12px;
  font-size: var(--text-xs);
  cursor: pointer;
  color: var(--ink-2);
}

.toggle button[aria-pressed='true'] {
  background: var(--ink);
  color: var(--surface);
}

.export {
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

.export:hover,
.export[data-state='open'] {
  border-color: var(--ink-2);
  color: var(--ink);
}

.body {
  min-width: 0;
}

.table-body {
  max-height: 520px;
  overflow: auto;
}

.note {
  font-size: var(--text-xs);
  color: var(--muted);
  border-top: 1px solid var(--rule);
  padding-top: var(--space-3);
}

.note :deep(p) {
  max-width: 100ch;
}
</style>

<style>
/* Portaled outside the component, so unscoped; the rest comes from the facet chooser's styles */
.export-option {
  border: 0;
  background: none;
  font: inherit;
  color: var(--ink);
  text-align: start;
  border-radius: var(--radius-s);
  padding-inline: var(--space-2);
}

.export-option:hover {
  background: var(--surface-sunk);
}
</style>
