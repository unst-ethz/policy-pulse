<script setup lang="ts">
/**
 * Each country's multilateral alignment (x) against a vote share (y), one colour per region with a
 * legend. Emphasising a region fades the others; marked countries get an ink ring and a direct
 * label. A line under the x axis says which way is with the majority, so it reads without help.
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

const regionColor = (region: string) =>
  (REGIONS as readonly string[]).includes(region)
    ? colors[`region-${region.toLowerCase()}` as keyof typeof colors]
    : colors.abstain

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
  const xs = props.points.map((p) => p.x)
  const xMin = xs.length ? Math.floor(Math.min(...xs) * 20) / 20 : 0
  const xMax = xs.length ? Math.ceil(Math.max(...xs) * 20) / 20 : 1
  const others = [...new Set(props.points.map((p) => p.region))].filter((r) => !(REGIONS as readonly string[]).includes(r))

  const regionSeries = [...REGIONS, ...others].map((region, i) => ({
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
            label: { formatter: t('alignment.mean'), color: colors['ink-2'], fontSize: 12 },
            data: [{ xAxis: props.meanX }],
          },
        }
      : {}),
  }))

  return {
    animation: false,
    legend: {
      top: 0,
      left: 0,
      icon: 'circle',
      itemWidth: 10,
      itemHeight: 10,
      selectedMode: false,
      textStyle: { color: colors['ink-2'], fontFamily: colors['font-sans'], fontSize: 12 },
      data: regionSeries.map((s) => s.name),
    },
    grid: { left: 8, right: 24, top: 56, bottom: 8, containLabel: true },
    tooltip: {
      trigger: 'item',
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: (p: { data: { name: string; value: [number, number]; votes: number } }) =>
        `<b>${p.data.name}</b><br>${t('alignment.x')}: ${n(p.data.value[0], 'score')}<br>${props.yLabel}: ${n(p.data.value[1], 'percent')}<br>${t('alignment.votes', { n: p.data.votes })}`,
    },
    xAxis: { type: 'value', name: t('alignment.x'), nameLocation: 'middle', nameGap: 28, min: xMin, max: xMax, ...axis },
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
            label: { position: p.x > (xMin + xMax) / 2 ? 'left' : 'right' },
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
  <div class="scatter">
    <VChart :option="option" autoresize :aria-label="t('alignment.label')" @click="onClick" />
  </div>
  <p class="direction">
    <span>{{ t('alignment.against') }}</span>
    <span>{{ t('alignment.with') }}</span>
  </p>
</template>

<style scoped>
.scatter {
  width: 100%;
  height: 560px;
}

.direction {
  display: flex;
  max-width: none;
  justify-content: space-between;
  gap: var(--space-4);
  margin: var(--space-1) 0 0;
  font-size: var(--text-xs);
  color: var(--muted);
}

.direction span:last-child {
  text-align: end;
}
</style>
