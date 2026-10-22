<script setup lang="ts">
/**
 * The resolution filter, in one row above everything it applies to. A page shows only the
 * filters that change what it displays, instead of greying out the ones that don't.
 */
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMeta } from '@/api/queries'
import { type CountryMode, countActive, useFilters } from '@/composables/useFilters'
import CountryPicker from './CountryPicker.vue'
import SubjectPicker from './SubjectPicker.vue'

type Field = 'years' | 'subjects' | 'keyword' | 'country'

const props = withDefaults(defineProps<{ fields?: Field[] }>(), {
  fields: () => ['years', 'subjects', 'keyword', 'country'],
})

const { t } = useI18n()
const meta = useMeta()
const { filters, update, clear } = useFilters()

const has = (field: Field) => props.fields.includes(field)

const years = computed(() => {
  const first = Number(meta.data.value?.earliest_date?.slice(0, 4) ?? 1946)
  const last = Number(meta.data.value?.latest_date?.slice(0, 4) ?? new Date().getFullYear())
  return Array.from({ length: last - first + 1 }, (_, i) => first + i)
})

const keyword = ref(filters.value.keyword)
watch(
  () => filters.value.keyword,
  (value) => (keyword.value = value),
)
let timer: ReturnType<typeof setTimeout> | undefined
function onKeyword() {
  clearTimeout(timer)
  timer = setTimeout(() => update({ keyword: keyword.value }), 400)
}

const subjects = computed({
  get: () => filters.value.subjects,
  set: (value: string[]) => update({ subjects: value }),
})
const country = computed({
  get: () => filters.value.country,
  set: (value: string | null) =>
    update({ country: value, mode: value ? (filters.value.mode === 'none' ? 'voted' : filters.value.mode) : 'none' }),
})

const modes: CountryMode[] = ['voted', 'member', 'none']
const active = computed(() => countActive(filters.value))
</script>

<template>
  <form class="filters" :aria-label="t('filters.label')" @submit.prevent>
    <div v-if="has('years')" class="field years">
      <span>{{ t('filters.years') }}</span>
      <div class="pair">
        <select
          class="input"
          :aria-label="t('filters.from')"
          :value="filters.from ?? ''"
          @change="update({ from: Number(($event.target as HTMLSelectElement).value) || null })"
        >
          <option value="">{{ t('filters.earliest') }}</option>
          <option v-for="y in years" :key="y" :value="y">{{ y }}</option>
        </select>
        <span aria-hidden="true">–</span>
        <select
          class="input"
          :aria-label="t('filters.to')"
          :value="filters.to ?? ''"
          @change="update({ to: Number(($event.target as HTMLSelectElement).value) || null })"
        >
          <option value="">{{ t('filters.latest') }}</option>
          <option v-for="y in years" :key="y" :value="y">{{ y }}</option>
        </select>
      </div>
    </div>

    <SubjectPicker v-if="has('subjects')" v-model="subjects" :label="t('filters.subjects')" class="subjects" />

    <label v-if="has('keyword')" class="field keyword">
      <span>{{ t('filters.keyword') }}</span>
      <input
        v-model="keyword"
        class="input"
        type="search"
        :placeholder="t('filters.keywordHint')"
        @input="onKeyword"
        @keydown.enter="update({ keyword })"
      />
    </label>

    <div v-if="has('country')" class="country">
      <CountryPicker v-model="country" :label="t('filters.country')" :placeholder="t('filters.anyCountry')" />
      <div v-if="filters.country" class="modes" role="radiogroup" :aria-label="t('filters.mode')">
        <label v-for="mode in modes" :key="mode">
          <input
            type="radio"
            name="country-mode"
            :value="mode"
            :checked="filters.mode === mode"
            @change="update({ mode })"
          />
          {{ t(`filters.modes.${mode}`) }}
        </label>
      </div>
    </div>

    <button v-if="active" type="button" class="button reset" @click="clear">
      {{ t('filters.clear', { n: active }) }}
    </button>
  </form>
</template>

<style scoped>
.filters {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: var(--space-3) var(--space-4);
  padding: var(--space-4);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius-m);
}

.pair {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.subjects {
  width: 240px;
}

.keyword {
  flex: 1 1 220px;
  max-width: 360px;
}

.keyword .input {
  width: 100%;
}

.country {
  display: grid;
  gap: var(--space-2);
  width: 240px;
}

.modes {
  display: flex;
  flex-wrap: wrap;
  gap: 2px var(--space-3);
  font-size: var(--text-xs);
  color: var(--ink-2);
}

.modes label {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
}

.modes input {
  accent-color: var(--ink);
  margin: 0;
}

.reset {
  align-self: end;
  margin-inline-start: auto;
}

@media (max-width: 640px) {
  .subjects,
  .country,
  .keyword {
    width: 100%;
    max-width: none;
  }
}
</style>
