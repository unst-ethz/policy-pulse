<script setup lang="ts">
/** A vote tally as one stacked bar with a labelled legend: Yes, No, Abstain, (not voting). */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const props = withDefaults(
  defineProps<{
    yes: number
    no: number
    abstain: number
    absent?: number
    absentLabel?: string
  }>(),
  { absent: 0, absentLabel: undefined },
)

const { t, n } = useI18n()

const parts = computed(() => {
  const items = [
    { key: 'Y', label: t('vote.Y'), value: props.yes },
    { key: 'N', label: t('vote.N'), value: props.no },
    { key: 'A', label: t('vote.A'), value: props.abstain },
    { key: 'X', label: props.absentLabel ?? t('vote.X'), value: props.absent },
  ].filter((p) => p.value > 0)
  const total = items.reduce((sum, p) => sum + p.value, 0) || 1
  return items.map((p) => ({ ...p, share: p.value / total }))
})
</script>

<template>
  <figure class="vote-bar">
    <div class="bar" role="img" :aria-label="parts.map((p) => `${p.label} ${p.value}`).join(', ')">
      <span
        v-for="part in parts"
        :key="part.key"
        class="segment"
        :data-vote="part.key"
        :style="{ flexGrow: part.value }"
      />
    </div>
    <figcaption class="legend">
      <span v-for="part in parts" :key="part.key" class="item" :data-vote="part.key">
        <span class="swatch" aria-hidden="true" />
        {{ part.label }}
        <strong class="numeric">{{ n(part.value) }}</strong>
        <span class="share">{{ n(part.share, 'percent') }}</span>
      </span>
    </figcaption>
  </figure>
</template>

<style scoped>
.vote-bar {
  margin: 0;
  display: grid;
  gap: var(--space-2);
}

.bar {
  display: flex;
  gap: 2px;
  height: 12px;
}

.segment {
  flex-basis: 0;
  min-width: 2px;
  background: var(--fill);
  border: 1px solid var(--edge, transparent);
}

.segment:first-child {
  border-start-start-radius: 4px;
  border-end-start-radius: 4px;
}

.segment:last-child {
  border-start-end-radius: 4px;
  border-end-end-radius: 4px;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-4);
  font-size: var(--text-sm);
  color: var(--ink-2);
}

.item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.item strong {
  color: var(--ink);
  font-weight: 600;
}

.share {
  color: var(--muted);
}

.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  background: var(--fill);
  border: 1px solid var(--edge, transparent);
}

[data-vote='Y'] {
  --fill: var(--yes);
}

[data-vote='N'] {
  --fill: var(--no);
}

[data-vote='A'] {
  --fill: var(--abstain);
}

[data-vote='X'] {
  --fill: var(--surface-sunk);
  --edge: var(--rule-strong);
}
</style>
