<script setup lang="ts">
/**
 * An editable list of countries (e.g. the comparison set): chips you can remove plus a picker to
 * add one. Each chip shows the colour its country has in the chart, so the list is the legend.
 */
import { useI18n } from 'vue-i18n'
import { useCountryIndex } from '@/api/queries'
import CountryPicker from './CountryPicker.vue'

const props = withDefaults(
  defineProps<{ label: string; max?: number; colors?: Record<string, string>; exclude?: string[] }>(),
  { max: 5, colors: () => ({}), exclude: () => [] },
)
const model = defineModel<string[]>({ required: true })
const { t } = useI18n()
const { name } = useCountryIndex()

function add(iso3: string | null) {
  if (iso3 && !model.value.includes(iso3) && model.value.length < props.max) {
    model.value = [...model.value, iso3]
  }
}

function remove(iso3: string) {
  model.value = model.value.filter((c) => c !== iso3)
}
</script>

<template>
  <div class="country-list">
    <ul class="chips" :aria-label="label">
      <li v-for="iso3 in model" :key="iso3" class="chip">
        <span v-if="colors[iso3]" class="line" :style="{ background: colors[iso3] }" aria-hidden="true" />
        {{ name(iso3) }}
        <button type="button" :aria-label="t('picker.remove', { name: name(iso3) })" @click="remove(iso3)">
          ×
        </button>
      </li>
    </ul>
    <CountryPicker
      v-if="model.length < max"
      :model-value="null"
      :label="t('picker.add')"
      hide-label
      :placeholder="t('picker.add')"
      :exclude="[...model, ...exclude]"
      class="add"
      @update:model-value="add"
    />
  </div>
</template>

<style scoped>
.country-list {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
}

.chips {
  display: contents;
  list-style: none;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 32px;
  padding-inline: 10px 4px;
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  background: var(--surface);
  font-size: var(--text-sm);
}

.line {
  width: 14px;
  height: 3px;
  border-radius: 2px;
}

.chip button {
  width: 24px;
  height: 24px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
}

.chip button:hover {
  background: var(--surface-sunk);
  color: var(--ink);
}

.add {
  width: 200px;
}
</style>
