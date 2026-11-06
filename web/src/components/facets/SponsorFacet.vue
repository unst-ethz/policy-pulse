<script setup lang="ts">
/** Only resolutions a state sponsored, optionally as an original sponsor. */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFilters } from '@/composables/useFilters'
import CountryPicker from '../CountryPicker.vue'
import FacetPanel from './FacetPanel.vue'

const { t } = useI18n()
const { filters, update } = useFilters()

const sponsor = computed({
  get: () => filters.value.sponsor,
  set: (value: string | null) => update({ sponsor: value }),
})
</script>

<template>
  <FacetPanel
    :title="t('filters.sponsor')"
    :active="!!filters.sponsor"
    collapsed
    @clear="update({ sponsor: null, original: false })"
  >
    <CountryPicker v-model="sponsor" :label="t('filters.sponsor')" :placeholder="t('filters.anyCountry')" hide-label />
    <template v-if="filters.sponsor">
      <label class="option">
        <input
          type="checkbox"
          :checked="filters.original"
          @change="update({ original: ($event.target as HTMLInputElement).checked })"
        />
        {{ t('filters.originalOnly') }}
      </label>
      <p class="note">{{ t('resolutions.sponsorCoverage') }}</p>
    </template>
  </FacetPanel>
</template>

<style scoped>
.option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  color: var(--ink-2);
  cursor: pointer;
}

.option input {
  margin: 0;
  accent-color: var(--ink);
}

.note {
  margin: 0;
  font-size: var(--text-xs);
  color: var(--muted);
}
</style>
