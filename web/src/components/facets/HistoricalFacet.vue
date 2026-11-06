<script setup lang="ts">
/** Whether per-country views include states that no longer exist (USSR, Yugoslavia…). Off by default. */
import { useI18n } from 'vue-i18n'
import { useFilters } from '@/composables/useFilters'
import FacetPanel from './FacetPanel.vue'

const { t } = useI18n()
const { filters, update } = useFilters()
</script>

<template>
  <FacetPanel
    :info="t('facets.info.historical')"
    :title="t('facets.historical')"
    :active="filters.historical"
    @clear="update({ historical: false })"
  >
    <label class="option">
      <input
        type="checkbox"
        :checked="filters.historical"
        @change="update({ historical: ($event.target as HTMLInputElement).checked })"
      />
      {{ t('facets.includeHistorical') }}
    </label>
  </FacetPanel>
</template>

<style scoped>
.option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  cursor: pointer;
}

.option input {
  margin: 0;
  accent-color: var(--ink);
}
</style>
