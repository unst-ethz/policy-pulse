<script setup lang="ts">
/**
 * Agreement scores (0..1) as labelled horizontal bars, best first. Plain HTML: the labels are
 * long (thesaurus domains, country names), and HTML wraps and translates them properly.
 * `display` replaces the score as the shown value (e.g. a count, with `value` its share of the bar).
 */
import { useI18n } from 'vue-i18n'

defineProps<{
  rows: { key: string; label: string; value: number; display?: string; detail?: string; to?: object }[]
}>()
const { n } = useI18n()
</script>

<template>
  <ol class="bars">
    <li v-for="row in rows" :key="row.key" class="row">
      <span class="label">
        <RouterLink v-if="row.to" :to="row.to">{{ row.label }}</RouterLink>
        <template v-else>{{ row.label }}</template>
        <small v-if="row.detail">{{ row.detail }}</small>
      </span>
      <span class="track" aria-hidden="true">
        <span class="fill" :style="{ inlineSize: `${row.value * 100}%` }" />
      </span>
      <span class="value numeric">{{ row.display ?? n(row.value, 'score') }}</span>
    </li>
  </ol>
</template>

<style scoped>
.bars {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: var(--space-2);
}

.row {
  display: grid;
  grid-template-columns: minmax(8rem, 2fr) 3fr 3.5rem;
  align-items: center;
  gap: var(--space-3);
  font-size: var(--text-sm);
}

.label {
  display: grid;
  line-height: 1.3;
}

.label small {
  color: var(--muted);
  font-size: var(--text-xs);
}

.track {
  height: 10px;
  background: var(--surface-sunk);
  border-radius: 0 4px 4px 0;
}

.fill {
  display: block;
  height: 100%;
  background: var(--yes);
  border-start-end-radius: 4px;
  border-end-end-radius: 4px;
}

.value {
  text-align: end;
  font-weight: 500;
}

@media (max-width: 520px) {
  .row {
    grid-template-columns: 1fr 3.5rem;
  }

  .track {
    grid-column: 1 / -1;
    grid-row: 2;
  }
}
</style>
