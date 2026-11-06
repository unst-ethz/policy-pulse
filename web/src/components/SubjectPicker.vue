<script setup lang="ts">
/**
 * Pick thesaurus subjects. A subject includes everything below it, so picking a domain selects
 * the whole branch. Typing searches every level and shows each hit with its domain.
 * The `trigger` slot swaps the select-like button for another PopoverTrigger.
 */
import { PopoverContent, PopoverPortal, PopoverRoot, PopoverTrigger } from 'reka-ui'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useSubjectTree } from '@/api/queries'
import { flatSubjects, sentenceCase } from '@/format'

const NO_SUBJECT = '__no_subject__'

defineProps<{ label: string }>()
const model = defineModel<string[]>({ required: true })

const { t } = useI18n()
const tree = useSubjectTree()
const search = ref('')
const expanded = ref(new Set<string>())

const flat = computed(() => flatSubjects(tree.data.value ?? []))

const labels = computed(() => new Map(flat.value.map((f) => [f.id, f.label])))
// Not `label`: that name is the prop the template shows as the field label.
const labelOf = (id: string) =>
  id === NO_SUBJECT ? t('subjects.none') : (labels.value.get(id) ?? id)

const matches = computed(() => {
  const q = search.value.trim().toLowerCase()
  if (!q) return []
  return flat.value.filter((f) => f.label.toLowerCase().includes(q)).slice(0, 80)
})

function toggle(id: string) {
  model.value = model.value.includes(id) ? model.value.filter((s) => s !== id) : [...model.value, id]
}

function toggleOpen(id: string) {
  const next = new Set(expanded.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expanded.value = next
}

const summary = computed(() => {
  if (!model.value.length) return t('subjects.all')
  if (model.value.length === 1) return labelOf(model.value[0]!)
  return t('subjects.count', { n: model.value.length })
})
</script>

<template>
  <div class="field subject-picker">
    <span v-if="!$slots.trigger">{{ label }}</span>
    <PopoverRoot>
      <!-- A custom trigger (a PopoverTrigger) replaces the select-like button and its label -->
      <slot name="trigger">
        <PopoverTrigger class="input trigger" :aria-label="`${label}: ${summary}`">
          <span class="summary">{{ summary }}</span>
          <span aria-hidden="true" class="caret">▾</span>
        </PopoverTrigger>
      </slot>
      <PopoverPortal>
        <PopoverContent class="popover subjects" :side-offset="4" align="start">
          <div class="search">
            <input
              v-model="search"
              class="input"
              type="search"
              :placeholder="t('subjects.search')"
              :aria-label="t('subjects.search')"
            />
            <button v-if="model.length" type="button" class="button" @click="model = []">
              {{ t('subjects.clear') }}
            </button>
          </div>
          <div class="viewport">
            <template v-if="search.trim()">
              <p v-if="!matches.length" class="empty">{{ t('picker.noMatch') }}</p>
              <label v-for="m in matches" :key="m.id" class="option">
                <input type="checkbox" :checked="model.includes(m.id)" @change="toggle(m.id)" />
                <span class="text">
                  {{ m.label }}
                  <small v-if="m.domain">{{ m.domain }}</small>
                </span>
              </label>
            </template>
            <template v-else>
              <label class="option">
                <input type="checkbox" :checked="model.includes(NO_SUBJECT)" @change="toggle(NO_SUBJECT)" />
                <span class="text">{{ t('subjects.none') }}</span>
              </label>
              <ul class="tree">
                <li v-for="domain in tree.data.value ?? []" :key="domain.id">
                  <div class="row">
                    <button
                      v-if="domain.children?.length"
                      type="button"
                      class="disclosure"
                      :aria-expanded="expanded.has(domain.id)"
                      :aria-label="t('subjects.expand', { name: domain.label })"
                      @click="toggleOpen(domain.id)"
                    >
                      {{ expanded.has(domain.id) ? '−' : '+' }}
                    </button>
                    <label class="option">
                      <input type="checkbox" :checked="model.includes(domain.id)" @change="toggle(domain.id)" />
                      <span class="text">{{ sentenceCase(domain.label) }}</span>
                    </label>
                  </div>
                  <ul v-if="expanded.has(domain.id)" class="tree nested">
                    <li v-for="child in domain.children" :key="child.id">
                      <label class="option">
                        <input type="checkbox" :checked="model.includes(child.id)" @change="toggle(child.id)" />
                        <span class="text">{{ sentenceCase(child.label) }}</span>
                      </label>
                    </li>
                  </ul>
                </li>
              </ul>
            </template>
          </div>
        </PopoverContent>
      </PopoverPortal>
    </PopoverRoot>
  </div>
</template>

<style scoped>
.trigger {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  width: 100%;
  cursor: pointer;
  text-align: start;
}

.summary {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.caret {
  color: var(--muted);
}
</style>

<style>
.popover.subjects {
  width: min(420px, 92vw);
  grid-template-rows: auto 1fr;
}

.popover.subjects .search {
  display: flex;
  gap: var(--space-2);
  padding: 8px;
  border-bottom: 1px solid var(--rule);
}

.popover.subjects .search .input {
  flex: 1;
}

.popover.subjects .tree {
  list-style: none;
  margin: 0;
  padding: 0;
}

.popover.subjects .tree.nested {
  padding-inline-start: 28px;
}

.popover.subjects .row {
  display: flex;
  align-items: center;
}

.popover.subjects .disclosure {
  width: 24px;
  height: 24px;
  flex: none;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
}

.popover.subjects .row .option {
  flex: 1;
}

.popover.subjects .tree:not(.nested) > li > .row .option {
  padding-inline-start: 0;
}

.popover.subjects .option {
  align-items: flex-start;
}

.popover.subjects .option input {
  margin-top: 3px;
  accent-color: var(--ink);
}

.popover.subjects .text {
  display: grid;
}

.popover.subjects .text small {
  color: var(--muted);
  font-size: var(--text-xs);
}
</style>
