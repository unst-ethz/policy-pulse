<script setup lang="ts">
/**
 * An editable list of countries (e.g. the comparison set): chips you can remove plus a picker to
 * add one. Each chip shows the colour its country has in the chart, so the list is the legend.
 * With `groups`, a quick select adds a whole group of States (e.g. the P5) at once.
 */
import { useI18n } from 'vue-i18n'
import { useCountryIndex } from '@/api/queries'
import { GROUPS } from '@/groups'
import CountryPicker from './CountryPicker.vue'

const props = withDefaults(
  defineProps<{ label: string; max?: number; colors?: Record<string, string>; exclude?: string[]; groups?: boolean }>(),
  { max: 5, colors: () => ({}), exclude: () => [], groups: false },
)
const model = defineModel<string[]>({ required: true })
const { t } = useI18n()
const { name } = useCountryIndex()

function add(iso3: string | null) {
  if (iso3 && !model.value.includes(iso3) && model.value.length < props.max) {
    model.value = [...model.value, iso3]
  }
}

// A group is added to the list when it fits, else it replaces the list (e.g. the country page's
// automatic comparisons); greyed out only when all of it is listed already
const additions = (key: keyof typeof GROUPS) =>
  GROUPS[key].filter((c) => !model.value.includes(c) && !props.exclude.includes(c))
function addGroup(select: HTMLSelectElement) {
  const key = select.value as keyof typeof GROUPS
  select.value = '' // back to the prompt
  if (!key) return
  const extra = additions(key)
  const members = GROUPS[key].filter((c) => !props.exclude.includes(c))
  model.value = model.value.length + extra.length <= props.max ? [...model.value, ...extra] : members.slice(0, props.max)
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
    <select
      v-if="groups"
      class="input group"
      :aria-label="t('groups.prompt')"
      @change="addGroup($event.target as HTMLSelectElement)"
    >
      <option value="">{{ t('groups.prompt') }}</option>
      <option v-for="key in Object.keys(GROUPS) as (keyof typeof GROUPS)[]" :key="key" :value="key" :disabled="!additions(key).length">
        {{ t(`groups.${key}`) }}
      </option>
    </select>
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

.add,
.group {
  width: 200px;
}
</style>
