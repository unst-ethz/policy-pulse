<script setup lang="ts">
import { DialogClose, DialogContent, DialogOverlay, DialogPortal, DialogRoot, DialogTitle, DialogTrigger } from 'reka-ui'
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { csvUrl, type Lang, type ListOptions, useCountryIndex, useResolutions } from '@/api/queries'
import CountryList from '@/components/CountryList.vue'
import QueryState from '@/components/QueryState.vue'
import ResolutionTable from '@/components/ResolutionTable.vue'
import FacetSidebar from '@/components/facets/FacetSidebar.vue'
import KeywordField from '@/components/KeywordField.vue'
import { countActive, useFilters } from '@/composables/useFilters'

const PAGE_SIZE = 50
const SORTS: ListOptions['sort'][] = ['date_desc', 'date_asc', 'consensus_asc', 'consensus_desc']

const { t, n, locale } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()
const { name } = useCountryIndex()

function setQuery(patch: Record<string, string | string[] | undefined>) {
  const query = { ...route.query, ...patch }
  for (const key of Object.keys(patch)) if (patch[key] === undefined) delete query[key]
  router.replace({ query })
}

const page = computed(() => Math.max(1, Number(route.query.page) || 1))
const sort = computed<ListOptions['sort']>(() => {
  const s = route.query.sort as ListOptions['sort']
  return SORTS.includes(s) ? s : 'date_desc'
})
const votes = computed<string[]>({
  get: () => {
    const v = route.query.votes
    return (Array.isArray(v) ? v : v ? [v] : []).filter((x): x is string => !!x)
  },
  set: (value) => setQuery({ votes: value.length ? value : undefined }),
})

const options = computed<ListOptions>(() => ({
  sort: sort.value,
  offset: (page.value - 1) * PAGE_SIZE,
  limit: PAGE_SIZE,
  votes: votes.value,
}))

const result = useResolutions(filters, options)
const pages = computed(() => Math.max(1, Math.ceil((result.data.value?.total ?? 0) / PAGE_SIZE)))
const voteColumns = computed(() => votes.value.map((iso3) => ({ iso3, label: name(iso3) })))
const download = computed(() => csvUrl(filters.value, votes.value, locale.value as Lang))

// Facets beside the list on wide screens; behind a "Filters" button in a drawer otherwise.
// Rendered in one place only, so each facet's queries and state exist once.
const wideQuery = window.matchMedia('(min-width: 960px)')
const wide = ref(wideQuery.matches)
const onWide = (event: MediaQueryListEvent) => (wide.value = event.matches)
onMounted(() => wideQuery.addEventListener('change', onWide))
onUnmounted(() => wideQuery.removeEventListener('change', onWide))
const drawer = ref(false)
const active = computed(() => countActive(filters.value))
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('resolutions.title') }}</h1>
      <p>{{ t('resolutions.lead') }}</p>
    </div>

    <div class="layout">
      <aside v-if="wide" class="side">
        <FacetSidebar />
      </aside>

    <section class="section">
      <div class="search">
        <KeywordField />
        <DialogRoot v-if="!wide" v-model:open="drawer">
          <DialogTrigger class="button">
            {{ active ? t('facets.openActive', { n: active }) : t('facets.open') }}
          </DialogTrigger>
          <DialogPortal>
            <DialogOverlay class="overlay" />
            <DialogContent class="drawer" :aria-describedby="undefined">
              <DialogTitle class="visually-hidden">{{ t('facets.title') }}</DialogTitle>
              <FacetSidebar />
              <div class="drawer-foot">
                <DialogClose class="button primary">
                  {{ t('facets.show', { n: n(result.data.value?.total ?? 0) }) }}
                </DialogClose>
              </div>
            </DialogContent>
          </DialogPortal>
        </DialogRoot>
      </div>
      <div class="toolbar">
        <p class="summary" aria-live="polite">
          <template v-if="result.data.value">
            {{
              t('resolutions.summary', {
                total: n(result.data.value.total),
                recorded: n(result.data.value.recorded_votes),
              })
            }}
          </template>
        </p>
        <label class="field inline">
          <span>{{ t('resolutions.sort') }}</span>
          <select
            class="input"
            :value="sort"
            @change="setQuery({ sort: ($event.target as HTMLSelectElement).value, page: undefined })"
          >
            <option v-for="s in SORTS" :key="s" :value="s">{{ t(`resolutions.sorts.${s}`) }}</option>
          </select>
        </label>
        <a class="button" :href="download" download>{{ t('resolutions.download') }}</a>
      </div>
      <div class="votes-of">
        <span>{{ t('resolutions.showVotes') }}</span>
        <CountryList v-model="votes" :label="t('resolutions.showVotes')" :max="4" />
      </div>

      <div class="panel flush">
        <QueryState
          :loading="result.isLoading.value"
          :fetching="result.isFetching.value"
          :error="result.error.value"
          :empty="result.data.value?.total === 0"
          :empty-text="t('resolutions.empty')"
          min-height="240px"
          @retry="result.refetch()"
        >
          <ResolutionTable :items="result.data.value?.items ?? []" :vote-columns="voteColumns" sponsors />
        </QueryState>
      </div>

      <nav v-if="pages > 1" class="pager" :aria-label="t('resolutions.pages')">
        <button class="button" type="button" :disabled="page <= 1" @click="setQuery({ page: String(page - 1) })">
          {{ t('resolutions.previous') }}
        </button>
        <span>{{ t('resolutions.page', { page, pages }) }}</span>
        <button
          class="button"
          type="button"
          :disabled="page >= pages"
          @click="setQuery({ page: String(page + 1) })"
        >
          {{ t('resolutions.next') }}
        </button>
      </nav>
    </section>
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

.search {
  display: flex;
  align-items: end;
  gap: var(--space-3);
}

.search .keyword {
  flex: 1;
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

.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: var(--space-3) var(--space-4);
}

.summary {
  margin-inline-end: auto;
  font-weight: 500;
}

.inline {
  grid-auto-flow: column;
  align-items: center;
  gap: var(--space-2);
}

.votes-of {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2) var(--space-3);
  font-size: var(--text-sm);
  color: var(--ink-2);
}

.flush {
  padding: 0;
  overflow: hidden;
}

.pager {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-4);
  font-size: var(--text-sm);
}

.pager .button:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
