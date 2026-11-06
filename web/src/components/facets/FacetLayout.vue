<script setup lang="ts">
/**
 * A page with its facets: in a sticky column beside the content on wide screens, behind a
 * "Filters (n)" button in a drawer otherwise. The sidebar is rendered in one place only, so each
 * facet's queries and state exist once. `applied` is what the page actually filters on (its own
 * context included), handed to the facets so their counts match the page.
 */
import { DialogClose, DialogContent, DialogOverlay, DialogPortal, DialogRoot, DialogTitle, DialogTrigger } from 'reka-ui'
import { computed, onMounted, onUnmounted, provide, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { type Filters, PAGE_FILTERS, useFilters } from '@/composables/useFilters'
import FacetSidebar from './FacetSidebar.vue'
import { activeCount, type FacetKey } from './registry'

const props = defineProps<{
  allowed: FacetKey[]
  defaults: FacetKey[]
  storageKey: string
  applied?: Filters
  /** The drawer's close button, e.g. "Show 120 resolutions" (default "Done"). */
  doneLabel?: string
  /** How many filters are set, when the page has some outside the facets (default: the facets'). */
  count?: number
}>()

const { t } = useI18n()
const { filters } = useFilters()
provide(
  PAGE_FILTERS,
  computed(() => props.applied ?? filters.value),
)

const wideQuery = window.matchMedia('(min-width: 960px)')
const wide = ref(wideQuery.matches)
const onWide = (event: MediaQueryListEvent) => (wide.value = event.matches)
onMounted(() => wideQuery.addEventListener('change', onWide))
onUnmounted(() => wideQuery.removeEventListener('change', onWide))
const drawer = ref(false)
const active = computed(() => props.count ?? activeCount(props.allowed, filters.value))
</script>

<template>
  <div class="layout">
    <aside v-if="wide" class="side">
      <FacetSidebar :allowed="allowed" :defaults="defaults" :storage-key="storageKey" :count="count" />
    </aside>

    <div class="main">
      <div v-if="$slots.toolbar || !wide" class="bar">
        <div v-if="$slots.toolbar" class="tools"><slot name="toolbar" /></div>
        <DialogRoot v-if="!wide" v-model:open="drawer">
          <DialogTrigger class="button">
            {{ active ? t('facets.openActive', { n: active }) : t('facets.open') }}
          </DialogTrigger>
          <DialogPortal>
            <DialogOverlay class="overlay" />
            <DialogContent class="drawer" :aria-describedby="undefined">
              <DialogTitle class="visually-hidden">{{ t('facets.title') }}</DialogTitle>
              <FacetSidebar :allowed="allowed" :defaults="defaults" :storage-key="storageKey" :count="count" />
              <div class="drawer-foot">
                <DialogClose class="button primary">{{ doneLabel ?? t('facets.done') }}</DialogClose>
              </div>
            </DialogContent>
          </DialogPortal>
        </DialogRoot>
      </div>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: var(--space-6);
  align-items: start;
}

@media (max-width: 959px) {
  .layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

/* Sticks below the site header and scrolls on its own when taller than the screen */
.side {
  position: sticky;
  top: 76px;
  max-height: calc(100vh - 92px);
  overflow-y: auto;
  padding-inline-end: var(--space-2);
}

.main {
  display: grid;
  gap: var(--space-6);
  min-width: 0;
}

.bar {
  display: flex;
  align-items: end;
  gap: var(--space-3);
}

.tools {
  flex: 1;
  min-width: 0;
}

.overlay {
  position: fixed;
  inset: 0;
  z-index: 200; /* above the sticky site header */
  background: rgb(0 0 0 / 0.2);
}

.drawer {
  position: fixed;
  z-index: 201;
  inset-block: 0;
  inset-inline-start: 0;
  width: min(92vw, 360px);
  overflow-y: auto;
  background: var(--page);
  border-inline-end: 1px solid var(--rule-strong);
  padding: var(--space-4) var(--space-4) 0;
  display: grid;
  align-content: start;
}

.drawer-foot {
  position: sticky;
  bottom: 0;
  padding-block: var(--space-3);
  background: var(--page);
  border-top: 1px solid var(--rule);
}

.drawer-foot .button {
  width: 100%;
  justify-content: center;
}
</style>
