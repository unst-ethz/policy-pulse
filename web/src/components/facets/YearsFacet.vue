<script setup lang="ts">
/**
 * The period: resolutions per year as bars over a two-thumb slider. The bars apply every other
 * filter but not the period itself, so they show what widening or moving the range would add.
 * The URL changes when a thumb is let go, not on every step of a drag.
 */
import { SliderRange, SliderRoot, SliderThumb, SliderTrack } from 'reka-ui'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFacets, useMeta } from '@/api/queries'
import { useFilters, usePageFilters } from '@/composables/useFilters'
import FacetPanel from './FacetPanel.vue'

const { t, n } = useI18n()
const { filters, update } = useFilters()
const meta = useMeta()
const facets = useFacets(usePageFilters())

const first = computed(() => Number(meta.data.value?.earliest_date?.slice(0, 4) ?? 1946))
const last = computed(() => Number(meta.data.value?.latest_date?.slice(0, 4) ?? new Date().getFullYear()))

const range = ref([0, 0])
watch(
  [() => filters.value.from, () => filters.value.to, first, last],
  () => (range.value = [filters.value.from ?? first.value, filters.value.to ?? last.value]),
  { immediate: true },
)

const bars = computed(() => {
  const counts = new Map((facets.data.value?.years ?? []).map((y) => [y.year, y.resolutions]))
  const max = Math.max(1, ...counts.values())
  return Array.from({ length: last.value - first.value + 1 }, (_, i) => {
    const year = first.value + i
    const count = counts.get(year) ?? 0
    return { year, count, height: (count / max) * 100, inside: year >= range.value[0]! && year <= range.value[1]! }
  })
})

function commit([from, to]: number[]) {
  update({ from: from === first.value ? null : from!, to: to === last.value ? null : to! })
}
</script>

<template>
  <FacetPanel
    :info="t('facets.info.years')"
    :title="t('filters.years')"
    :active="!!(filters.from || filters.to)"
    @clear="update({ from: null, to: null })"
  >
    <div class="bars" :class="{ fetching: facets.isFetching.value }" aria-hidden="true">
      <div
        v-for="bar in bars"
        :key="bar.year"
        class="bar"
        :class="{ inside: bar.inside }"
        :style="{ height: `${bar.height}%` }"
        :title="t('facets.yearTitle', { year: bar.year, n: n(bar.count) })"
      />
    </div>
    <SliderRoot
      v-model="range"
      class="slider"
      :min="first"
      :max="last"
      :step="1"
      :min-steps-between-thumbs="0"
      @value-commit="commit"
    >
      <SliderTrack class="track">
        <SliderRange class="range" />
      </SliderTrack>
      <SliderThumb class="thumb" :aria-label="t('filters.from')" />
      <SliderThumb class="thumb" :aria-label="t('filters.to')" />
    </SliderRoot>
    <p class="span">
      <span>{{ range[0] }}</span>
      <span>{{ range[1] }}</span>
    </p>
  </FacetPanel>
</template>

<style scoped>
.bars {
  display: flex;
  align-items: flex-end;
  gap: 1px;
  height: 64px;
  transition: opacity 0.15s;
}

.bars.fetching {
  opacity: 0.6;
}

.bar {
  flex: 1;
  min-width: 0;
  background: var(--rule-strong);
}

.bar.inside {
  background: var(--ink-2);
}

.slider {
  position: relative;
  display: flex;
  align-items: center;
  height: 20px;
  touch-action: none;
  user-select: none;
}

.track {
  position: relative;
  flex: 1;
  height: 4px;
  border-radius: 2px;
  background: var(--rule-strong);
}

.range {
  position: absolute;
  height: 100%;
  border-radius: 2px;
  background: var(--ink);
}

.thumb {
  display: block;
  width: 16px;
  height: 16px;
  border: 2px solid var(--ink);
  border-radius: 50%;
  background: var(--surface);
  cursor: grab;
}

.thumb:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}

.span {
  display: flex;
  justify-content: space-between;
  margin: 0;
  font-size: var(--text-xs);
  color: var(--ink-2);
  font-variant-numeric: tabular-nums;
}
</style>
