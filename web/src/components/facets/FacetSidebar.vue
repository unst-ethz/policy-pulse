<script setup lang="ts">
/**
 * The resolution list's filters as a column of facets. Each facet reads and writes the URL
 * filter on its own, so which ones show (and in what order) is decided only here.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { countActive, useFilters } from '@/composables/useFilters'
import CountryFacet from './CountryFacet.vue'
import SponsorFacet from './SponsorFacet.vue'
import SubjectsFacet from './SubjectsFacet.vue'
import TermsFacet from './TermsFacet.vue'
import YearsFacet from './YearsFacet.vue'

const { t } = useI18n()
const { filters, clear } = useFilters()
const active = computed(() => countActive(filters.value))
</script>

<template>
  <div class="facets" role="group" :aria-label="t('filters.label')">
    <div class="head">
      <h2>{{ t('facets.title') }}</h2>
      <button v-if="active" type="button" class="clear" @click="clear">
        {{ t('filters.clear', { n: active }) }}
      </button>
    </div>
    <YearsFacet />
    <SubjectsFacet />
    <TermsFacet />
    <CountryFacet />
    <SponsorFacet />
  </div>
</template>

<style scoped>
.facets {
  display: grid;
  align-content: start;
  min-width: 0;
}

.head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-2);
  padding-bottom: var(--space-3);
}

h2 {
  margin: 0;
  font-size: var(--text-lg);
}

.clear {
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  font-size: var(--text-sm);
  color: var(--accent);
  text-decoration: underline;
  cursor: pointer;
}
</style>
