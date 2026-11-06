<script setup lang="ts">
/**
 * The most common terms of the listed resolutions as chips: a chip adds its term to the keyword
 * filter, or takes it out again. "More terms" opens the full word cloud in a side drawer, beside
 * the list it narrows.
 */
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  DialogClose,
  DialogContent,
  DialogOverlay,
  DialogPortal,
  DialogRoot,
  DialogTitle,
  DialogTrigger,
} from 'reka-ui'
import { useWordCloud } from '@/api/queries'
import { hasTerm, toggleTerm, useFilters } from '@/composables/useFilters'
import TermCloud from './TermCloud.vue'

const CHIPS = 8

const { t, n } = useI18n()
const { filters, update } = useFilters()
// Same request as the cloud's default mode, so opening the drawer reuses it
const cloud = useWordCloud(filters, 'default')
const open = ref(false)

const chips = computed(() =>
  (cloud.data.value?.terms ?? []).slice(0, CHIPS).map((term) => ({
    term: term.term,
    count: term.count,
    active: hasTerm(filters.value.keyword, term.term),
  })),
)

function toggle(term: string) {
  update({ keyword: toggleTerm(filters.value.keyword, term) })
}
</script>

<template>
  <div v-if="chips.length" class="terms" :class="{ fetching: cloud.isFetching.value }">
    <span class="lead">{{ t('cloud.common') }}</span>
    <button
      v-for="chip in chips"
      :key="chip.term"
      type="button"
      class="chip"
      :aria-pressed="chip.active"
      :title="chip.active ? t('cloud.remove', { term: chip.term }) : t('cloud.termTitle', { term: chip.term, n: chip.count })"
      @click="toggle(chip.term)"
    >
      {{ chip.term }}
      <span v-if="chip.active" aria-hidden="true">×</span>
      <span v-else class="count">{{ n(chip.count) }}</span>
    </button>

    <DialogRoot v-model:open="open">
      <DialogTrigger class="more">{{ t('cloud.more') }}</DialogTrigger>
      <DialogPortal>
        <DialogOverlay class="overlay" />
        <DialogContent class="drawer" :aria-describedby="undefined">
          <div class="drawer-head">
            <DialogTitle class="visually-hidden">{{ t('cloud.more') }}</DialogTitle>
            <DialogClose class="button">{{ t('cloud.close') }}</DialogClose>
          </div>
          <TermCloud />
        </DialogContent>
      </DialogPortal>
    </DialogRoot>
  </div>
</template>

<style scoped>
.terms {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  transition: opacity 0.15s;
}

.terms.fetching {
  opacity: 0.6;
}

.lead {
  color: var(--ink-2);
}

.chip {
  display: inline-flex;
  align-items: baseline;
  gap: var(--space-2);
  padding: 3px 10px;
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  background: var(--surface);
  color: var(--ink);
  font: inherit;
  cursor: pointer;
}

.chip:hover {
  border-color: var(--ink-2);
}

.chip[aria-pressed='true'] {
  background: var(--ink);
  border-color: var(--ink);
  color: var(--surface);
}

.count {
  color: var(--muted);
  font-size: var(--text-xs);
}

.more {
  border: 0;
  background: none;
  padding: 3px 4px;
  font: inherit;
  color: var(--accent);
  text-decoration: underline;
  cursor: pointer;
}
</style>

<!-- Not scoped: the drawer is teleported to <body>, outside this component's scope -->
<style>
.overlay {
  position: fixed;
  inset: 0;
  z-index: 200; /* above the sticky site header */
  background: rgb(0 0 0 / 0.2);
}

.drawer {
  position: fixed;
  z-index: 201;
  inset-block: 0;
  inset-inline-end: 0;
  width: min(100vw, 640px);
  overflow-y: auto;
  background: var(--page);
  border-inline-start: 1px solid var(--rule-strong);
  padding: var(--space-4);
  display: grid;
  align-content: start;
  gap: var(--space-3);
}

.drawer-head {
  display: flex;
  justify-content: flex-end;
}
</style>
