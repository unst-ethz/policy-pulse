<script setup lang="ts">
/**
 * A country's sponsorships per year, original and additional stacked, in front of a pale bar for
 * the resolutions with sponsorship data that year: the coverage stays visible, so a year without
 * data never reads as a year without sponsoring.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'

const props = defineProps<{
  years: { year: number; with_data: number; original: number; additional: number }[]
}>()
const { t, n } = useI18n()
const colors = useThemeColors()

// Drawn in HTML, not by ECharts: ECharts' legend overlaps its text in right-to-left pages.
const legend = computed(() => [
  { label: t('sponsorship.original'), color: colors.ink },
  { label: t('sponsorship.additional'), color: colors.muted },
  { label: t('sponsorship.withData'), color: colors.rule },
])

const option = computed(() => {
  const axis = {
    axisLine: { lineStyle: { color: colors['rule-strong'] } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: colors['font-sans'], fontSize: 12 },
    splitLine: { lineStyle: { color: colors.rule, width: 1 } },
  }
  const bar = (name: string, key: 'with_data' | 'original' | 'additional', color: string, stack?: string) => ({
    name,
    type: 'bar',
    stack,
    barGap: '-100%',
    barCategoryGap: '20%',
    itemStyle: { color },
    emphasis: { disabled: true },
    data: props.years.map((y) => y[key]),
  })
  return {
    animation: false,
    grid: { left: 8, right: 8, top: 16, bottom: 8, containLabel: true },
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: (items: { dataIndex: number }[]) => {
        const y = props.years[items[0]?.dataIndex ?? 0]
        if (!y) return ''
        return `<b>${y.year}</b><br>${t('sponsorship.original')}: ${n(y.original)}<br>${t('sponsorship.additional')}: ${n(y.additional)}<br><span style="color:${colors.muted}">${t('sponsorship.withData')}: ${n(y.with_data)}</span>`
      },
    },
    xAxis: { type: 'category', data: props.years.map((y) => String(y.year)), ...axis, splitLine: { show: false } },
    yAxis: { type: 'value', minInterval: 1, ...axis },
    series: [
      bar(t('sponsorship.withData'), 'with_data', colors.rule),
      bar(t('sponsorship.original'), 'original', colors.ink, 'sponsored'),
      bar(t('sponsorship.additional'), 'additional', colors.muted, 'sponsored'),
    ],
  }
})
</script>

<template>
  <ul class="legend">
    <li v-for="item in legend" :key="item.label">
      <span class="swatch" :style="{ background: item.color }" aria-hidden="true" />{{ item.label }}
    </li>
  </ul>
  <div class="chart">
    <VChart :option="option" autoresize :aria-label="t('sponsorship.chartTitle')" />
  </div>
</template>

<style scoped>
.legend {
  list-style: none;
  margin: 0 0 var(--space-2);
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-4);
  font-size: var(--text-xs);
  color: var(--ink-2);
}

.legend li {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
}

.swatch {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.chart {
  width: 100%;
  height: 320px;
}
</style>
