<script setup lang="ts">
/**
 * A small "i" button that explains the thing beside it. A click (or tap, or Enter) opens the
 * text in a popover, so it works on touch screens and from the keyboard, unlike a hover tooltip.
 */
import { PopoverArrow, PopoverContent, PopoverPortal, PopoverRoot, PopoverTrigger } from 'reka-ui'
import { useI18n } from 'vue-i18n'

defineProps<{ text: string; about: string }>()
const { t } = useI18n()
</script>

<template>
  <PopoverRoot>
    <PopoverTrigger class="info" :aria-label="t('info.about', { name: about })">i</PopoverTrigger>
    <PopoverPortal>
      <PopoverContent class="info-text" side="bottom" align="start" :side-offset="6" :collision-padding="12">
        {{ text }}
        <PopoverArrow class="info-arrow" />
      </PopoverContent>
    </PopoverPortal>
  </PopoverRoot>
</template>

<style scoped>
.info {
  display: inline-grid;
  place-items: center;
  width: 16px;
  height: 16px;
  flex: none;
  padding: 0;
  border: 1px solid var(--muted);
  border-radius: 50%;
  background: none;
  color: var(--muted);
  font: italic 600 11px/1 var(--font-serif, Georgia, serif);
  cursor: pointer;
}

.info:hover,
.info[data-state='open'] {
  border-color: var(--ink);
  color: var(--ink);
}
</style>

<style>
/* Portaled outside the component (and reka's popper wrapper drops the scope attribute), so unscoped */
.info-text {
  z-index: 300; /* above the filter drawer */
  max-width: min(300px, 90vw);
  padding: var(--space-3);
  background: var(--surface);
  border: 1px solid var(--rule-strong);
  border-radius: var(--radius-s);
  box-shadow: 0 4px 16px rgb(0 0 0 / 0.08);
  font-size: var(--text-sm);
  font-weight: 400;
  line-height: 1.45;
  color: var(--ink);
}

.info-arrow {
  fill: var(--surface);
  stroke: var(--rule-strong);
}
</style>
