<script setup lang="ts">
/**
 * Counts per year as bars. A bar with `stack` stacks onto the others of that stack; a bar without
 * one is drawn behind, e.g. a pale bar for the resolutions with sponsorship data that year, so a
 * year without data never reads as a year without sponsoring.
 */
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'

type Row = { year: number } & Record<string, number>

const props = defineProps<{
  years: Row[]
  bars: { key: string; label: string; color: 'ink' | 'muted' | 'rule' | 'accent'; stack?: string }[]
  label: string
  /** How values read in the tooltip and on the axis (default: a plain number). */
  format?: (value: number) => string
}>()
const { n } = useI18n()
const colors = useThemeColors()
const fmt = (value: number) => (props.format ? props.format(value) : n(value))

// Drawn in HTML, not by ECharts: ECharts' legend overlaps its text in right-to-left pages.
const legend = computed(() => props.bars.map((b) => ({ label: b.label, color: colors[b.color] })))

const option = computed(() => {
  const axis = {
    axisLine: { lineStyle: { color: colors['rule-strong'] } },
    axisTick: { show: false },
    axisLabel: { color: colors.muted, fontFamily: colors['font-sans'], fontSize: 12 },
    splitLine: { lineStyle: { color: colors.rule, width: 1 } },
  }
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
        // Stacked bars first, in order; the background bar last and muted
        const lines = [...props.bars]
          .sort((a, b) => Number(!a.stack) - Number(!b.stack))
          .map((b) => {
            const line = `${b.label}: ${fmt(y[b.key] ?? 0)}`
            return b.stack ? line : `<span style="color:${colors.muted}">${line}</span>`
          })
        return `<b>${y.year}</b><br>${lines.join('<br>')}`
      },
    },
    xAxis: { type: 'category', data: props.years.map((y) => String(y.year)), ...axis, splitLine: { show: false } },
    yAxis: { type: 'value', minInterval: props.format ? undefined : 1, ...axis, axisLabel: { ...axis.axisLabel, formatter: fmt } },
    series: props.bars.map((b) => ({
      name: b.label,
      type: 'bar',
      stack: b.stack,
      barGap: '-100%',
      barCategoryGap: '20%',
      itemStyle: { color: colors[b.color] },
      emphasis: { disabled: true },
      data: props.years.map((y) => y[b.key] ?? 0),
    })),
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
    <VChart :option="option" autoresize :aria-label="label" />
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
