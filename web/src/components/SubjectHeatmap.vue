<script setup lang="ts">
/**
 * Agreement by topic with several countries: one row per subject, one column per compared
 * country, each cell coloured on the map's red–blue scale and labelled with its value. Column
 * heads carry the country's colour from the comparison list, as in the timeline. An empty cell
 * means too few shared votes on that subject. Clicking a cell selects its subject and country.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'

export interface HeatmapColumn {
  key: string
  label: string
  color: string
}
export interface HeatmapRow {
  key: string
  label: string
  values: Record<string, { agreement: number | null; shared_votes: number } | undefined>
}

const props = defineProps<{ columns: HeatmapColumn[]; rows: HeatmapRow[]; color: (value: number) => string }>()
const emit = defineEmits<{ select: [row: string, column: string] }>()
const { t, n } = useI18n()
const colors = useThemeColors()

const ROW = 34
const HEAD = 44
const height = computed(() => `${props.rows.length * ROW + HEAD + 8}px`)
// Room for the topic labels and a readable cell per country; narrower screens scroll sideways
const minWidth = computed(() => `${160 + props.columns.length * 90}px`)

// Labels get what the drawn width leaves them, cut short only when it runs out
const box = ref<HTMLElement>()
const width = ref(800)
let observer: ResizeObserver | null = null
onMounted(() => {
  observer = new ResizeObserver((entries) => {
    const next = Math.round(entries[0]!.contentRect.width)
    if (next && Math.abs(next - width.value) > 8) width.value = next
  })
  if (box.value) {
    width.value = box.value.clientWidth || 800
    observer.observe(box.value)
  }
})
onBeforeUnmount(() => observer?.disconnect())
const rowLabelWidth = computed(() => Math.min(260, Math.round(width.value * 0.32)))
const columnLabelWidth = computed(() =>
  Math.max(40, Math.floor((width.value - rowLabelWidth.value - 24) / Math.max(1, props.columns.length)) - 12),
)

type Datum = { value: [number, number, number]; label: { color: string } }

// Dark text on the pale middle of the scale, light text on its saturated ends
const textOn = (value: number) => (Math.abs(value - 0.5) > 0.3 ? '#ffffff' : colors.ink)

const option = computed(() => {
  const text = { color: colors['ink-2'], fontFamily: colors['font-sans'], fontSize: 12 }
  const data: Datum[] = []
  props.rows.forEach((row, y) =>
    props.columns.forEach((col, x) => {
      const v = row.values[col.key]?.agreement
      if (v != null) data.push({ value: [x, y, v], label: { color: textOn(v) } })
    }),
  )
  return {
    animation: false,
    grid: { left: 8, right: 8, top: HEAD, bottom: 8, containLabel: true },
    tooltip: {
      // On the page, not in the chart: inside the sideways-scroll wrapper, a tooltip near an edge
      // makes the wrapper overflow and show a scrollbar
      appendTo: 'body',
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: ({ data: d }: { data: Datum }) => {
        const [x, y, v] = d.value
        const row = props.rows[y]!
        const col = props.columns[x]!
        const votes = row.values[col.key]?.shared_votes ?? 0
        return `<div style="color:${colors.muted}">${row.label}</div>${col.label}: <b>${n(v, 'score')}</b><br><span style="color:${colors.muted}">${t('country.sharedVotes', { n: votes })}</span>`
      },
    },
    xAxis: {
      type: 'category',
      position: 'top',
      data: props.columns.map((c) => c.label),
      axisLine: { show: false },
      axisTick: { show: false },
      splitArea: { show: false },
      axisLabel: {
        ...text,
        interval: 0,
        // shortcut: long names are cut to the column width; the tooltip and table have them in full
        width: columnLabelWidth.value,
        overflow: 'truncate',
        formatter: (label: string, i: number) => `{c${i}|}  ${label}`,
        rich: Object.fromEntries(
          props.columns.map((c, i) => [`c${i}`, { width: 10, height: 3, backgroundColor: c.color }]),
        ),
      },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: props.rows.map((r) => r.label),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { ...text, width: rowLabelWidth.value, overflow: 'truncate' },
    },
    // ECharts colours heatmap cells only through a visual map: hidden, on the page's own scale
    visualMap: {
      show: false,
      min: 0,
      max: 1,
      dimension: 2,
      inRange: { color: Array.from({ length: 11 }, (_, i) => props.color(i / 10)) },
    },
    series: [
      {
        type: 'heatmap',
        data,
        itemStyle: { borderColor: colors.surface, borderWidth: 2 },
        label: { show: true, fontFamily: colors['font-sans'], fontSize: 12, formatter: ({ data: d }: { data: Datum }) => n(d.value[2], 'score') },
        emphasis: { itemStyle: { borderColor: colors.ink, borderWidth: 1 } },
      },
    ],
  }
})

function onClick(event: { data?: unknown }) {
  const d = event.data as Datum | undefined
  if (!d) return
  const [x, y] = d.value
  emit('select', props.rows[y]!.key, props.columns[x]!.key)
}
</script>

<template>
  <div class="scroll">
    <div ref="box" class="heatmap" :style="{ height, minWidth }">
      <VChart :option="option" autoresize :aria-label="t('country.topicsTitle')" @click="onClick" />
    </div>
  </div>
</template>

<style scoped>
.scroll {
  overflow-x: auto;
}

.heatmap {
  width: 100%;
}
</style>
