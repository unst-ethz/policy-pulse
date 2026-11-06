<script setup lang="ts">
/**
 * The frame every facet shares: a title that collapses it, an "i" that explains it, and "Clear"
 * while its filter is set.
 * A facet is open by default; `collapsed` starts it closed unless its filter is active.
 */
import { ref, useId } from 'vue'
import { useI18n } from 'vue-i18n'
import InfoTip from '../InfoTip.vue'

const props = defineProps<{ title: string; info: string; active?: boolean; collapsed?: boolean }>()
defineEmits<{ clear: [] }>()

const { t } = useI18n()
const open = ref(!props.collapsed || props.active)
const body = useId()
</script>

<template>
  <section class="facet">
    <header class="head">
      <h2>
        <button type="button" class="toggle" :aria-expanded="open" :aria-controls="body" @click="open = !open">
          <span aria-hidden="true" class="caret" :class="{ open }">▸</span>
          {{ title }}
          <span v-if="active" class="dot" :title="t('facets.active')" />
        </button>
      </h2>
      <InfoTip :text="info" :about="title" />
      <button v-if="active" type="button" class="clear" @click="$emit('clear')">
        {{ t('facets.clear') }}<span class="visually-hidden">: {{ title }}</span>
      </button>
    </header>
    <div v-show="open" :id="body" class="body">
      <slot />
    </div>
  </section>
</template>

<style scoped>
.facet {
  display: grid;
  gap: var(--space-3);
  padding-block: var(--space-4);
  border-top: 1px solid var(--rule);
}

.head {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}

.clear {
  margin-inline-start: auto;
}

h2 {
  margin: 0;
  font-size: var(--text-sm);
  font-weight: 600;
}

.toggle {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  color: var(--ink);
  cursor: pointer;
}

.caret {
  display: inline-block;
  width: 1em;
  color: var(--muted);
  transition: transform 0.15s;
}

.caret.open {
  transform: rotate(90deg);
}

[dir='rtl'] .caret:not(.open) {
  transform: scaleX(-1);
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
}

.clear {
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  font-size: var(--text-xs);
  color: var(--accent);
  text-decoration: underline;
  cursor: pointer;
}

.body {
  display: grid;
  gap: var(--space-3);
  min-width: 0;
}
</style>
