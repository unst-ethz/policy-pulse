<script setup lang="ts">
/** One country's vote, as a small labelled chip (letter + colour, full word for screen readers). */
import { useI18n } from 'vue-i18n'
import type { Vote } from '@/api/client'

defineProps<{ vote: Vote | undefined; label?: string }>()
const { t } = useI18n()
</script>

<template>
  <span v-if="vote" class="chip" :data-vote="vote" :title="t(`vote.${vote}`)">
    <span v-if="label" class="label">{{ label }}</span>
    <span aria-hidden="true" class="letter">{{ vote === 'X' ? '–' : vote }}</span>
    <span class="visually-hidden">{{ t(`vote.${vote}`) }}</span>
  </span>
</template>

<style scoped>
.chip {
  display: inline-flex;
  align-items: stretch;
  border-radius: var(--radius-s);
  overflow: hidden;
  font-family: var(--font-condensed);
  font-size: var(--text-xs);
  line-height: 1.6;
  border: 1px solid var(--fill);
  white-space: nowrap;
}

.label {
  padding: 0 6px;
  color: var(--ink-2);
  background: var(--surface);
}

.letter {
  min-width: 20px;
  padding: 0 4px;
  text-align: center;
  font-weight: 600;
  background: var(--fill);
  color: #fff;
}

[data-vote='Y'] {
  --fill: var(--yes-cell);
}

[data-vote='N'] {
  --fill: var(--no-cell);
}

[data-vote='A'] {
  --fill: var(--abstain-cell);
}

[data-vote='X'] {
  --fill: var(--rule-strong);
}

[data-vote='X'] .letter {
  background: var(--surface-sunk);
  color: var(--muted);
}
</style>
