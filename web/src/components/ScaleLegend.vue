<script setup lang="ts">
/** The legend of a continuous diverging scale: a gradient with labelled ends (and centre). */
import { computed } from 'vue'

const props = defineProps<{
  color: (value: number) => string
  low: string
  high: string
  center?: { value: number; label: string } | null
  noData?: string
}>()

const stops = computed(() =>
  Array.from({ length: 11 }, (_, i) => `${props.color(i / 10)} ${i * 10}%`).join(', '),
)
</script>

<template>
  <div class="legend">
    <div class="scale">
      <div class="ramp" :style="{ background: `linear-gradient(to right, ${stops})` }" />
      <div
        v-if="center"
        class="center"
        :style="{ insetInlineStart: `${center.value * 100}%` }"
        aria-hidden="true"
      />
      <div class="labels">
        <span>{{ low }}</span>
        <span>{{ high }}</span>
      </div>
      <p v-if="center" class="center-label">
        <span class="tick" aria-hidden="true" />{{ center.label }}
      </p>
    </div>
    <span v-if="noData" class="no-data"><span class="swatch" aria-hidden="true" />{{ noData }}</span>
  </div>
</template>

<style scoped>
.legend {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: var(--space-2) var(--space-5);
  font-size: var(--text-xs);
  color: var(--ink-2);
}

.scale {
  position: relative;
  width: min(100%, 360px);
}

.ramp {
  height: 10px;
  border-radius: 2px;
}

[dir='rtl'] .ramp {
  transform: scaleX(-1);
}

.center {
  position: absolute;
  top: -3px;
  width: 2px;
  height: 16px;
  background: var(--ink);
  translate: -1px 0;
}

.labels {
  position: relative;
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
}

.center-label {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
  color: var(--ink);
}

.tick {
  width: 2px;
  height: 12px;
  background: var(--ink);
}

.no-data {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.swatch {
  width: 14px;
  height: 10px;
  border: 1px solid var(--rule-strong);
  background: var(--surface);
  border-radius: 2px;
}
</style>
