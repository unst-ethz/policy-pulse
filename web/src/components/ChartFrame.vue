<script setup lang="ts">
/**
 * A chart with its heading, a "how to read this" line, and a table twin. The table carries every
 * value the chart shows, so nothing depends on hovering or on colour vision.
 */
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

defineProps<{ title: string; description?: string }>()
const view = ref<'chart' | 'table'>('chart')
const { t } = useI18n()
</script>

<template>
  <section class="frame">
    <div class="head">
      <div class="titles">
        <h2>{{ title }}</h2>
        <p v-if="description" class="description">{{ description }}</p>
      </div>
      <div class="controls">
        <slot name="controls" />
        <div class="toggle" role="group" :aria-label="t('chart.view')">
          <button
            type="button"
            :aria-pressed="view === 'chart'"
            @click="view = 'chart'"
          >
            {{ t('chart.chart') }}
          </button>
          <button
            type="button"
            :aria-pressed="view === 'table'"
            @click="view = 'table'"
          >
            {{ t('chart.table') }}
          </button>
        </div>
      </div>
    </div>
    <div v-show="view === 'chart'" class="body"><slot /></div>
    <div v-if="view === 'table'" class="body table-body"><slot name="table" /></div>
    <div v-if="$slots.note" class="note"><slot name="note" /></div>
  </section>
</template>

<style scoped>
.frame {
  display: grid;
  gap: var(--space-4);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius-m);
  padding: var(--space-5);
  min-width: 0;
}

.head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--space-3) var(--space-5);
}

.titles {
  display: grid;
  gap: var(--space-1);
  max-width: 64ch;
}

.description {
  color: var(--ink-2);
  font-size: var(--text-sm);
}

.controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3);
}

.toggle {
  display: inline-flex;
  border: 1px solid var(--rule-strong);
  border-radius: var(--radius-s);
  overflow: hidden;
}

.toggle button {
  border: 0;
  background: var(--surface);
  padding: 4px 12px;
  font-size: var(--text-xs);
  cursor: pointer;
  color: var(--ink-2);
}

.toggle button[aria-pressed='true'] {
  background: var(--ink);
  color: var(--surface);
}

.body {
  min-width: 0;
}

.table-body {
  max-height: 520px;
  overflow: auto;
}

.note {
  font-size: var(--text-xs);
  color: var(--muted);
  border-top: 1px solid var(--rule);
  padding-top: var(--space-3);
}

.note :deep(p) {
  max-width: 100ch;
}
</style>
