<script setup lang="ts">
/**
 * Every member state's vote on one resolution, as on the voting board in the General Assembly
 * Hall: one cell per country, grouped by vote and ordered by code (the label each cell shows).
 * Grouping, group labels and counts carry the vote as well as colour, so the board reads without
 * colour vision too.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { Vote } from '@/api/client'

const props = withDefaults(
  defineProps<{
    votes: Record<string, Vote>
    name: (iso3: string) => string
    highlight?: string | null
    /** Countries to mark with a dot (e.g. the sponsors), named by `markedLabel` for screen readers. */
    marked?: string[]
    markedLabel?: string
    /** Hide the not-voting group (it includes states that were not yet members). */
    showAbsent?: boolean
  }>(),
  { highlight: null, showAbsent: true, marked: () => [], markedLabel: '' },
)

const { t } = useI18n()

const label = (iso3: string, vote: Vote) =>
  `${props.name(iso3)}: ${t(`vote.${vote}`)}${props.marked.includes(iso3) ? `, ${props.markedLabel}` : ''}`

const ORDER: Vote[] = ['Y', 'N', 'A', 'X']

const groups = computed(() =>
  ORDER.filter((vote) => vote !== 'X' || props.showAbsent)
    .map((vote) => ({
      vote,
      countries: Object.keys(props.votes)
        .filter((iso3) => props.votes[iso3] === vote)
        .sort(),
    }))
    .filter((group) => group.countries.length > 0),
)
</script>

<template>
  <div class="board">
    <section v-for="group in groups" :key="group.vote" class="group" :data-vote="group.vote">
      <h3 class="group-head">
        <span class="swatch" aria-hidden="true" />
        {{ t(`vote.${group.vote}`) }}
        <span class="count numeric">{{ group.countries.length }}</span>
      </h3>
      <ul class="cells">
        <li v-for="iso3 in group.countries" :key="iso3">
          <RouterLink
            :to="{ name: 'country', params: { iso3 } }"
            class="cell"
            :class="{ highlighted: iso3 === highlight, marked: marked.includes(iso3) }"
            :title="label(iso3, group.vote)"
            :aria-label="label(iso3, group.vote)"
          >
            {{ iso3 }}
          </RouterLink>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.board {
  display: grid;
  gap: var(--space-4);
}

.group {
  display: grid;
  gap: var(--space-2);
}

.group-head {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  font-weight: 500;
  color: var(--ink-2);
}

.count {
  color: var(--ink);
  font-weight: 600;
}

.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  background: var(--fill);
  border: 1px solid var(--edge, transparent);
}

.cells {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(44px, 1fr));
  gap: 2px;
}

.cell {
  display: block;
  padding: 3px 0;
  border-radius: 2px;
  background: var(--fill);
  border: 1px solid var(--edge, transparent);
  color: var(--cell-ink, #fff);
  font-family: var(--font-condensed);
  font-size: 0.75rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  line-height: 1.4;
  text-align: center;
  text-decoration: none;
}

.cell:hover {
  outline: 2px solid var(--ink);
  outline-offset: -1px;
}

.cell.marked {
  position: relative;
}

.cell.marked::after {
  content: '';
  position: absolute;
  inset-block-start: 2px;
  inset-inline-end: 2px;
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--cell-ink, #fff);
}

.cell.highlighted {
  outline: 3px solid var(--focus);
  outline-offset: 1px;
  position: relative;
  z-index: 1;
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
  --fill: transparent;
  --edge: var(--rule-strong);
  --cell-ink: var(--muted);
}
</style>
