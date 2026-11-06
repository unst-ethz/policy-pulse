<script setup lang="ts">
/**
 * Search-as-you-type country picker. Matches current names in the selected language, English
 * names, historical names and variants ("Burma" finds Myanmar), so people find a country by
 * whatever name they know it under.
 */
import {
  ComboboxAnchor,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxPortal,
  ComboboxRoot,
  ComboboxViewport,
} from 'reka-ui'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useCountries } from '@/api/queries'

const props = withDefaults(
  defineProps<{ label: string; placeholder?: string; exclude?: string[]; hideLabel?: boolean }>(),
  { placeholder: undefined, exclude: () => [], hideLabel: false },
)
const model = defineModel<string | null>({ default: null })

const { t } = useI18n()
const countries = useCountries()
const search = ref('')

const byIso = computed(() => new Map((countries.data.value ?? []).map((c) => [c.iso3, c])))

function normalise(text: string) {
  return text.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase()
}

const options = computed(() => {
  const query = normalise(search.value.trim())
  return (countries.data.value ?? [])
    .filter((c) => !props.exclude.includes(c.iso3))
    .filter(
      (c) =>
        !query ||
        c.iso3.toLowerCase() === query ||
        normalise(`${c.name} ${c.search_terms}`).includes(query),
    )
    .slice(0, 60)
})

const display = (iso3: string | null) => (iso3 ? (byIso.value.get(iso3)?.name ?? iso3) : '')
</script>

<template>
  <ComboboxRoot v-model="model" ignore-filter :reset-search-term-on-blur="true" class="picker">
    <label class="field">
      <span :class="{ 'visually-hidden': hideLabel }">{{ label }}</span>
      <ComboboxAnchor class="anchor">
        <!-- Keyed on the list loading, so a preset value shows its name once names arrive. -->
        <ComboboxInput
          :key="countries.data.value ? 'ready' : 'loading'"
          v-model="search"
          class="input"
          :display-value="display"
          :placeholder="placeholder ?? t('picker.country')"
        />
        <button
          v-if="model"
          type="button"
          class="clear"
          :aria-label="t('picker.clear', { name: display(model) })"
          @click="model = null"
        >
          ×
        </button>
      </ComboboxAnchor>
    </label>
    <ComboboxPortal>
      <ComboboxContent position="popper" :side-offset="4" class="popover">
        <ComboboxViewport class="viewport">
          <ComboboxEmpty class="empty">{{ t('picker.noMatch') }}</ComboboxEmpty>
          <ComboboxItem
            v-for="country in options"
            :key="country.iso3"
            :value="country.iso3"
            :text-value="country.name"
            class="option"
          >
            <span>{{ country.name }}</span>
            <span v-if="country.former_state" class="former">{{ t('picker.former') }}</span>
            <span class="code">{{ country.iso3 }}</span>
          </ComboboxItem>
        </ComboboxViewport>
      </ComboboxContent>
    </ComboboxPortal>
  </ComboboxRoot>
</template>

<style scoped>
.picker {
  min-width: 0;
}

.anchor {
  position: relative;
  display: flex;
}

.anchor .input {
  width: 100%;
  padding-inline-end: 28px;
}

.clear {
  position: absolute;
  inset-inline-end: 4px;
  top: 50%;
  translate: 0 -50%;
  width: 24px;
  height: 24px;
  border: 0;
  background: transparent;
  color: var(--muted);
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
}

.clear:hover {
  color: var(--ink);
}
</style>

<style>
/* Portaled outside the component, so unscoped. Shared by the pickers. */
.popover {
  z-index: 300; /* above dialogs and drawers (201) */
  min-width: var(--reka-combobox-trigger-width, 240px);
  max-height: min(360px, var(--reka-combobox-content-available-height, 360px));
  background: var(--surface);
  border: 1px solid var(--rule-strong);
  border-radius: var(--radius-m);
  box-shadow: 0 8px 24px rgb(15 20 30 / 0.12);
  overflow: hidden;
  display: grid;
}

.popover .viewport {
  overflow-y: auto;
  padding: 4px;
}

.popover .option {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  padding: 6px 8px;
  border-radius: var(--radius-s);
  font-size: var(--text-sm);
  cursor: pointer;
}

.popover .option[data-highlighted] {
  background: var(--accent-wash);
}

.popover .option[data-state='checked'] {
  font-weight: 600;
}

.popover .option .code {
  margin-inline-start: auto;
  font-family: var(--font-condensed);
  font-size: var(--text-xs);
  color: var(--muted);
}

.popover .option .former {
  font-size: var(--text-xs);
  color: var(--muted);
}

.popover .empty {
  padding: 8px;
  color: var(--muted);
  font-size: var(--text-sm);
}
</style>
