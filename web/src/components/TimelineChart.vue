<script setup lang="ts">
/**
 * Agreement per General Assembly session, one line per compared country. Lines keep the colour
 * of their country's position in the comparison list (never re-coloured when one is removed),
 * carry a direct label at their last point, and leave gaps where a session has too few votes.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'
import type { TimelineSeries } from '@/charts/types'


const props = defineProps<{ series: TimelineSeries[]; height?: string }>()
const { t, n } = useI18n()
const colors = useThemeColors()

// Start where the data does: a country admitted in 2002 has no values for 1946-2001.
const firstYear = computed(() => {
  const years = props.series.flatMap((s) => s.points.filter((p) => p.agreement != null).map((p) => p.year))
  return years.length ? Math.min(...years) : 'dataMin'
})

const option = computed(() => {
  const axis = {
    axisLine: { lineStyle: { color: colors['rule-strong'] } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: colors['font-sans'], fontSize: 12 },
    splitLine: { lineStyle: { color: colors.rule, width: 1 } },
  }
  return {
    animation: false,
    grid: { left: 8, right: 96, top: 16, bottom: 8, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'line', lineStyle: { color: colors['rule-strong'] } },
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: (items: { seriesName: string; color: string; data: [number, number | null, string, number] }[]) => {
        if (!items.length) return ''
        const [year, , session] = items[0]!.data
        const rows = items
          .map(
            (i) =>
              `<div style="display:flex;gap:8px;align-items:center"><span style="width:10px;height:2px;background:${i.color}"></span>${i.seriesName}<b style="margin-inline-start:auto">${i.data[1] == null ? t('chart.noValue') : n(i.data[1], 'score')}</b></div>`,
          )
          .join('')
        return `<div style="min-width:180px"><div style="color:${colors.muted}">${t('timeline.session', { session, year })}</div>${rows}</div>`
      },
    },
    xAxis: { type: 'value', min: firstYear.value, max: 'dataMax', ...axis, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(v) }, splitLine: { show: false } },
    yAxis: { type: 'value', min: 0, max: 1, interval: 0.25, ...axis },
    series: props.series.map((s) => ({
      name: s.label,
      type: 'line',
      showSymbol: false,
      symbolSize: 8,
      connectNulls: false,
      lineStyle: { width: 2, color: s.color },
      itemStyle: { color: s.color },
      emphasis: { focus: 'series' },
      // Direct labels only while they fit; with more lines the chips above are the legend.
      endLabel: {
        show: props.series.length <= 3,
        formatter: s.label,
        color: colors['ink-2'],
        fontFamily: colors['font-sans'],
        fontSize: 12,
        distance: 6,
      },
      data: s.points.map((p) => [p.year, p.agreement, p.session, p.votes]),
    })),
  }
})
</script>

<template>
  <div class="timeline" :style="{ height: height ?? '360px' }">
    <VChart :option="option" autoresize :aria-label="t('timeline.label')" />
  </div>
</template>

<style scoped>
.timeline {
  width: 100%;
}
</style>
