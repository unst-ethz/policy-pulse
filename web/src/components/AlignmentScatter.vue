<script setup lang="ts">
/**
 * Each country's multilateral alignment (x) against a vote share (y). Everyone is grey; one
 * region can be emphasised in blue and one country is ringed in gold with a direct label —
 * emphasis rather than a colour per region, which no palette keeps distinguishable at 5+ hues.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'
import type { ScatterPoint } from '@/charts/types'


const props = defineProps<{
  points: ScatterPoint[]
  yLabel: string
  meanX: number | null
  region: string | null
  selected: string | null
}>()
const emit = defineEmits<{ select: [iso3: string] }>()

const { t, n } = useI18n()
const colors = useThemeColors()

const option = computed(() => {
  const axis = {
    axisLine: { lineStyle: { color: colors['rule-strong'] } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: colors['font-sans'], fontSize: 12 },
    splitLine: { lineStyle: { color: colors.rule } },
    nameTextStyle: { color: colors['ink-2'], fontFamily: colors['font-sans'], fontSize: 12 },
  }
  const background = props.points.filter((p) => p.region !== props.region && p.country !== props.selected)
  const emphasised = props.points.filter((p) => p.region === props.region && p.country !== props.selected)
  const chosen = props.points.filter((p) => p.country === props.selected)
  const toData = (p: ScatterPoint) => ({ value: [p.x, p.y], name: p.name, country: p.country, votes: p.votes })
  const xs = props.points.map((p) => p.x)
  return {
    animation: false,
    grid: { left: 8, right: 24, top: 24, bottom: 8, containLabel: true },
    tooltip: {
      trigger: 'item',
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: (p: { data: { name: string; value: [number, number]; votes: number } }) =>
        `<b>${p.data.name}</b><br>${t('alignment.x')}: ${n(p.data.value[0], 'score')}<br>${props.yLabel}: ${n(p.data.value[1], 'percent')}<br>${t('alignment.votes', { n: p.data.votes })}`,
    },
    xAxis: {
      type: 'value',
      name: t('alignment.x'),
      nameLocation: 'middle',
      nameGap: 28,
      min: xs.length ? Math.floor(Math.min(...xs) * 20) / 20 : 0,
      max: xs.length ? Math.ceil(Math.max(...xs) * 20) / 20 : 1,
      ...axis,
    },
    // No axis name: the chart title names the vertical axis, and a name here gets clipped.
    yAxis: {
      type: 'value',
      min: 0,
      ...axis,
      axisLabel: { ...axis.axisLabel, formatter: (v: number) => n(v, 'percentShort') },
    },
    series: [
      {
        type: 'scatter',
        symbolSize: 9,
        itemStyle: { color: colors.abstain, opacity: 0.55, borderColor: colors.surface, borderWidth: 1 },
        data: background.map(toData),
        markLine: props.meanX == null ? undefined : {
          silent: true,
          symbol: 'none',
          lineStyle: { color: colors['ink-2'], type: 'solid', width: 1 },
          label: { formatter: t('alignment.mean'), color: colors['ink-2'], fontSize: 12 },
          data: [{ xAxis: props.meanX }],
        },
      },
      {
        type: 'scatter',
        symbolSize: 10,
        itemStyle: { color: colors.yes, borderColor: colors.surface, borderWidth: 2 },
        data: emphasised.map(toData),
      },
      {
        type: 'scatter',
        symbolSize: 14,
        itemStyle: { color: colors.accent, borderColor: colors.ink, borderWidth: 2 },
        label: { show: true, position: 'right', formatter: '{b}', color: colors.ink, fontSize: 13, fontWeight: 600 },
        data: chosen.map(toData),
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
</template>

<style scoped>
.scatter {
  width: 100%;
  height: 520px;
}
</style>
