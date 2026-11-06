<script setup lang="ts">
/**
 * The most common terms of the listed resolutions as chips: a chip adds its term to the keyword
 * filter, or takes it out again. "More terms" opens the full word cloud in a side drawer, beside
 * the list it narrows. Unlike the other facets its counts include its own filter: each is what
 * adding that term would give.
 */
import { computed, ref, watch } from "vue";
import { useI18n } from "vue-i18n";
import {
  DialogClose,
  DialogContent,
  DialogOverlay,
  DialogPortal,
  DialogRoot,
  DialogTitle,
  DialogTrigger,
} from "reka-ui";
import { useWordCloud } from "@/api/queries";
import { useRoute, useRouter } from "vue-router";
import {
  exactTerms,
  hasTerm,
  toggleTerm,
  useFilters,
} from "@/composables/useFilters";
import TermCloud from "../TermCloud.vue";
import FacetPanel from "./FacetPanel.vue";

const CHIPS = 8;

const { t, n } = useI18n();
const { filters, update } = useFilters();
// Same request as the cloud's default mode, so opening the drawer reuses it
const cloud = useWordCloud(filters, "default");
const open = ref(false);

// The terms in the filter first, pressed, so each can be taken out again even when it is not
// among the most common ones; then the most common others, up to CHIPS
const chips = computed(() => {
  const counts = new Map(
    (cloud.data.value?.terms ?? []).map((term) => [
      term.term.toLocaleLowerCase(),
      term.count,
    ]),
  );
  const active = exactTerms(filters.value.keyword).map((term) => ({
    term,
    count: counts.get(term.toLocaleLowerCase()) ?? null,
    active: true,
  }));
  const others = (cloud.data.value?.terms ?? [])
    .filter((term) => !hasTerm(filters.value.keyword, term.term))
    .slice(0, Math.max(0, CHIPS - active.length))
    .map((term) => ({ term: term.term, count: term.count, active: false }));
  return [...active, ...others];
});

// The drawer's mode lives in the URL while it is open; closing it drops it again
const route = useRoute();
const router = useRouter();
watch(open, (isOpen) => {
  if (!isOpen && route.query.cloud)
    router.replace({ query: { ...route.query, cloud: undefined } });
});

function toggle(term: string) {
  update({ keyword: toggleTerm(filters.value.keyword, term) });
}

// Takes out the chips' terms only; the rest of a typed search stays
const active = computed(() => exactTerms(filters.value.keyword).length > 0);
function clear() {
  update({ keyword: exactTerms(filters.value.keyword).reduce(toggleTerm, filters.value.keyword) });
}
</script>

<template>
  <FacetPanel :title="t('facets.terms')" :active="active" @clear="clear">
  <div class="terms" :class="{ fetching: cloud.isFetching.value }">
    <template v-if="chips.length">
      <button
        v-for="chip in chips"
        :key="chip.term"
        type="button"
        class="chip"
        :aria-pressed="chip.active"
        :aria-label="chip.term"
        :title="
          chip.active
            ? t('cloud.remove', { term: chip.term })
            : t('cloud.termTitle', { term: chip.term, n: chip.count ?? 0 })
        "
        @click="toggle(chip.term)"
      >
        {{ chip.term }}
        <span v-if="chip.active" aria-hidden="true">×</span>
        <span v-else class="count" aria-hidden="true">{{
          n(chip.count ?? 0)
        }}</span>
      </button>
    </template>

    <DialogRoot v-model:open="open">
      <DialogTrigger class="more">{{ t("cloud.more") }}</DialogTrigger>
      <DialogPortal>
        <DialogOverlay class="overlay" />
        <DialogContent class="drawer" :aria-describedby="undefined">
          <div class="drawer-head">
            <DialogTitle class="visually-hidden">{{
              t("cloud.more")
            }}</DialogTitle>
            <DialogClose class="button">{{ t("cloud.close") }}</DialogClose>
          </div>
          <TermCloud />
        </DialogContent>
      </DialogPortal>
    </DialogRoot>
  </div>
  </FacetPanel>
</template>

<style scoped>
.terms {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
  font-size: var(--text-sm);
  transition: opacity 0.15s;
}

.terms.fetching {
  opacity: 0.6;
}

.chip {
  display: inline-flex;
  align-items: baseline;
  gap: var(--space-2);
  padding: 3px 10px;
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  background: var(--surface);
  color: var(--ink);
  font: inherit;
  cursor: pointer;
}

.chip:hover {
  border-color: var(--ink-2);
}

.chip[aria-pressed="true"] {
  background: var(--ink);
  border-color: var(--ink);
  color: var(--surface);
}

.count {
  color: var(--muted);
  font-size: var(--text-xs);
}

.more {
  border: 0;
  background: none;
  padding: 3px 4px;
  font: inherit;
  color: var(--accent);
  text-decoration: underline;
  cursor: pointer;
}

/* The drawer is teleported to <body>; scoped styles still reach it (the scope attribute is on
   the elements, not their position in the DOM). */
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
  inset-inline-end: 0;
  width: min(100vw, 640px);
  overflow-y: auto;
  background: var(--page);
  border-inline-start: 1px solid var(--rule-strong);
  padding: var(--space-4);
  display: grid;
  align-content: start;
  gap: var(--space-3);
}

.drawer-head {
  display: flex;
  justify-content: flex-end;
}
</style>
