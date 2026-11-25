<script setup lang="ts">
/**
 * Each country's multilateral alignment (x) against a vote share (y), one colour per region with a
 * legend. Emphasising a region fades the others; marked countries get an ink ring and a direct
 * label. Above the plot two labels point away from the average line (with / against the majority);
 * below it the average's value and the axis title. Those are HTML so they wrap, lined up with the
 * plot through fixed grid margins (`GRID`), which turn the average into a CSS position.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'
import { REGIONS, type ScatterPoint } from '@/charts/types'

const props = defineProps<{
  points: ScatterPoint[]
  yLabel: string
  meanX: number | null
  region: string | null
  marked: string[]
}>()
const emit = defineEmits<{ select: [iso3: string] }>()

const { t, n } = useI18n()
const colors = useThemeColors()

// Plot margins in px; the HTML rows around the chart use the same ones to line up with it
const GRID = { left: 48, right: 24, top: 12, bottom: 28 }

const regionColor = (region: string) =>
  (REGIONS as readonly string[]).includes(region)
    ? colors[`region-${region.toLowerCase()}` as keyof typeof colors]
    : colors.abstain

const xs = computed(() => props.points.map((p) => p.x))
const xMin = computed(() => (xs.value.length ? Math.floor(Math.min(...xs.value) * 20) / 20 : 0))
const xMax = computed(() => (xs.value.length ? Math.ceil(Math.max(...xs.value) * 20) / 20 : 1))
const regions = computed(() => [
  ...REGIONS,
  ...new Set(props.points.map((p) => p.region).filter((r) => !(REGIONS as readonly string[]).includes(r))),
])

// The average as a fraction of the plot width
const meanAt = computed(() =>
  props.meanX == null ? null : (props.meanX - xMin.value) / (xMax.value - xMin.value || 1),
)
const position = computed(() => ({
  '--plot-left': `${GRID.left}px`,
  '--plot-right': `${GRID.right}px`,
  '--mean': meanAt.value ?? 0.5,
}))

const option = computed(() => {
  const axis = {
    axisLine: { lineStyle: { color: colors['rule-strong'] } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: colors['font-sans'], fontSize: 12 },
    splitLine: { lineStyle: { color: colors.rule } },
    nameTextStyle: { color: colors['ink-2'], fontFamily: colors['font-sans'], fontSize: 12 },
  }
  const toData = (p: ScatterPoint) => ({ value: [p.x, p.y], name: p.name, country: p.country, votes: p.votes })
  const faded = (region: string) => props.region != null && props.region !== region
  const regionSeries = regions.value.map((region, i) => ({
    type: 'scatter',
    name: t(`regions.${region}`, region),
    symbolSize: 9,
    itemStyle: {
      color: regionColor(region),
      opacity: faded(region) ? 0.15 : 0.85,
      borderColor: colors.surface,
      borderWidth: 1,
    },
    data: props.points.filter((p) => p.region === region && !props.marked.includes(p.country)).map(toData),
    // The average line rides on the first series
    ...(i === 0 && props.meanX != null
      ? {
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: colors['ink-2'], type: 'solid', width: 1 },
            label: { show: false },
            data: [{ xAxis: props.meanX }],
          },
        }
      : {}),
  }))

  return {
    animation: false,
    grid: { ...GRID, containLabel: false },
    tooltip: {
      trigger: 'item',
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: (p: { data: { name: string; value: [number, number]; votes: number } }) =>
        `<b>${p.data.name}</b><br>${t('alignment.x')}: ${n(p.data.value[0], 'score')}<br>${props.yLabel}: ${n(p.data.value[1], 'percent')}<br>${t('alignment.votes', { n: p.data.votes })}`,
    },
    xAxis: { type: 'value', min: xMin.value, max: xMax.value, ...axis },
    // No axis name: the chart title names the vertical axis, and a name here gets clipped.
    yAxis: {
      type: 'value',
      min: 0,
      ...axis,
      axisLabel: { ...axis.axisLabel, formatter: (v: number) => n(v, 'percentShort') },
    },
    series: [
      ...regionSeries,
      {
        type: 'scatter',
        symbolSize: 14,
        z: 3,
        itemStyle: { borderColor: colors.ink, borderWidth: 2 },
        // A halo in the surface colour keeps the name readable over the dots it crosses
        label: {
          show: true,
          position: 'right',
          formatter: '{b}',
          color: colors.ink,
          fontSize: 13,
          fontWeight: 600,
          textBorderColor: colors.surface,
          textBorderWidth: 3,
        },
        labelLayout: { hideOverlap: false, moveOverlap: 'shiftY' },
        data: props.points
          .filter((p) => props.marked.includes(p.country))
          // Names on the right half go left of their dot, so they stay inside the chart
          .map((p) => ({
            ...toData(p),
            itemStyle: { color: regionColor(p.region) },
            label: { position: p.x > (xMin.value + xMax.value) / 2 ? 'left' : 'right' },
          })),
      },
    ],
  }
})

function onClick(params: { data?: unknown }) {
  const country = (params.data as { country?: string } | null)?.country
  if (country) emit('select', country)
}
</script>

<template>
  <div class="figure" :style="position">
    <ul class="legend">
      <li v-for="r in regions" :key="r" :class="{ faded: region != null && region !== r }">
        <span class="dot" :style="{ background: regionColor(r) }" aria-hidden="true" />{{ t(`regions.${r}`, r) }}
      </li>
    </ul>
    <p v-if="meanAt != null" class="direction">
      <span>{{ t('alignment.against') }}</span>
      <span>{{ t('alignment.with') }}</span>
    </p>
    <div class="scatter">
      <VChart :option="option" autoresize :aria-label="t('alignment.label')" @click="onClick" />
    </div>
    <p v-if="meanAt != null" class="mean">
      <span>{{ t('alignment.meanValue', { value: n(meanX!, 'score') }) }}</span>
    </p>
    <p class="axis-title">{{ t('alignment.x') }}</p>
  </div>
</template>

<style scoped>
.scatter {
  width: 100%;
  height: 560px;
}

/* Spans the plot, the margins included, so `--plot-left/right` line up with the chart's grid */
.direction,
.mean,
.axis-title {
  max-width: none;
  margin: 0;
  font-size: var(--text-xs);
  color: var(--muted);
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-3);
  margin: 0 0 var(--space-3);
  padding: 0;
  list-style: none;
  font-size: var(--text-xs);
  color: var(--ink-2);
}

.legend .faded {
  opacity: 0.4;
}

.legend .dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-inline-end: 6px;
  border-radius: 50%;
}

/* Two columns that meet at the average line, each label pointing away from it. The meeting point
   only moves off the line when a side would get narrower than 110px (on phones, or when filters
   push the average to an edge), so both labels stay readable. */
.direction {
  display: grid;
  grid-template-columns:
    clamp(
      calc(var(--plot-left) + 110px),
      calc(var(--plot-left) + (100% - var(--plot-left) - var(--plot-right)) * var(--mean)),
      calc(100% - 110px)
    )
    1fr;
  align-items: end;
}

.direction span:first-child {
  padding-inline: var(--plot-left) var(--space-2);
  text-align: end;
}

.direction span:last-child {
  padding-inline: var(--space-2) 0;
}

/* The average's value, centred under its line */
.mean {
  position: relative;
  height: 1.5em;
}

.mean span {
  position: absolute;
  left: calc(var(--plot-left) + (100% - var(--plot-left) - var(--plot-right)) * var(--mean));
  transform: translateX(-50%);
  white-space: nowrap;
  color: var(--ink-2);
}

.axis-title {
  padding-inline: var(--plot-left) var(--plot-right);
  text-align: center;
  color: var(--ink-2);
}
</style>
