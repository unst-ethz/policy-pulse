<script setup lang="ts">
/** A country's point of view: only resolutions it voted on, or those adopted while a member. */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { type CountryMode, useFilters } from '@/composables/useFilters'
import CountryPicker from '../CountryPicker.vue'
import FacetPanel from './FacetPanel.vue'

const { t } = useI18n()
const { filters, update } = useFilters()

const country = computed({
  get: () => filters.value.country,
  set: (value: string | null) =>
    update({ country: value, mode: value ? (filters.value.mode === 'none' ? 'voted' : filters.value.mode) : 'none' }),
})
const modes: CountryMode[] = ['voted', 'member', 'none']
</script>

<template>
  <FacetPanel
    :title="t('filters.country')"
    :active="!!filters.country"
    collapsed
    @clear="update({ country: null, mode: 'none' })"
  >
    <CountryPicker v-model="country" :label="t('filters.country')" :placeholder="t('filters.anyCountry')" hide-label />
    <div v-if="filters.country" class="options" role="radiogroup" :aria-label="t('filters.mode')">
      <label v-for="mode in modes" :key="mode">
        <input
          type="radio"
          name="facet-country-mode"
          :value="mode"
          :checked="filters.mode === mode"
          @change="update({ mode })"
        />
        {{ t(`filters.modes.${mode}`) }}
      </label>
    </div>
  </FacetPanel>
</template>

<style scoped>
.options {
  display: grid;
  gap: var(--space-1);
  font-size: var(--text-sm);
  color: var(--ink-2);
}

.options label {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  cursor: pointer;
}

.options input {
  margin: 0;
  accent-color: var(--ink);
}
</style>
