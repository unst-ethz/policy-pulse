<script setup lang="ts">
/**
 * A world map with one fill per country. The caller decides what the colour means (a vote, an
 * agreement score) by passing `fill`; the map only draws and reports hovers/clicks.
 */
import { computed, onMounted, ref } from 'vue'
import VChart from 'vue-echarts'
import { useI18n } from 'vue-i18n'
import { echarts, loadWorld, mapProjection } from '@/charts/echarts'
import { useThemeColors } from '@/composables/useThemeColors'

const props = withDefaults(
  defineProps<{
    fill: (iso3: string) => string | null
    tooltip: (iso3: string) => string
    highlight?: string | null
    /** Width / height; the Equal Earth world is about twice as wide as tall. */
    aspect?: number
  }>(),
  { highlight: null, aspect: 2.05 },
)
const emit = defineEmits<{ select: [iso3: string] }>()

const { t } = useI18n()
const colors = useThemeColors()
const ready = ref(false)
const isos = ref<string[]>([])

onMounted(async () => {
  await loadWorld()
  const map = echarts.getMap('world') as unknown as {
    geoJSON: { features: { properties: { iso3: string } }[] }
  }
  isos.value = map.geoJSON.features.map((f) => f.properties.iso3)
  ready.value = true
})

const option = computed(() => {
  const regions = isos.value.map((iso3) => {
    const fill = props.fill(iso3)
    const selected = iso3 === props.highlight
    return {
      name: iso3,
      itemStyle: {
        areaColor: fill ?? colors['surface'],
        borderColor: selected ? colors.accent : colors['rule-strong'],
        borderWidth: selected ? 2 : 0.5,
      },
      emphasis: { itemStyle: { areaColor: fill ?? colors['surface'], borderColor: colors.ink, borderWidth: 1.5 } },
    }
  })
  return {
    animation: false,
    tooltip: {
      trigger: 'item',
      backgroundColor: colors.surface,
      borderColor: colors['rule-strong'],
      textStyle: { color: colors.ink, fontFamily: colors['font-sans'], fontSize: 13 },
      formatter: (p: { name: string }) => props.tooltip(p.name),
    },
    geo: {
      map: 'world',
      nameProperty: 'iso3',
      projection: mapProjection,
      roam: true,
      scaleLimit: { min: 1, max: 8 },
      // Fill the box: layoutSize would size against the shorter side and leave the map small.
      left: 8,
      right: 8,
      top: 8,
      bottom: 8,
      aspectScale: 1,
      label: { show: false },
      emphasis: { label: { show: false } },
      select: { disabled: true },
      itemStyle: { areaColor: colors.surface, borderColor: colors['rule-strong'], borderWidth: 0.5 },
      regions,
    },
    series: [],
  }
})

function onClick(params: { name?: string; componentType?: string }) {
  if (params.componentType === 'geo' && params.name) emit('select', params.name)
}
</script>

<template>
  <div class="map" :style="{ aspectRatio: aspect }">
    <VChart v-if="ready" :option="option" autoresize :aria-label="t('map.label')" @click="onClick" />
    <p v-else class="loading">{{ t('state.loading') }}</p>
  </div>
</template>

<style scoped>
.map {
  position: relative;
  width: 100%;
  background: var(--surface-sunk);
  border-radius: var(--radius-s);
  overflow: hidden;
}

.loading {
  padding: var(--space-4);
  color: var(--muted);
  font-size: var(--text-sm);
}
</style>
