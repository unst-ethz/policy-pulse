<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { csvUrl, type Lang, type ListOptions, useCountryIndex, useResolutions } from '@/api/queries'
import CountryList from '@/components/CountryList.vue'
import FilterBar from '@/components/FilterBar.vue'
import QueryState from '@/components/QueryState.vue'
import ResolutionTable from '@/components/ResolutionTable.vue'
import { useFilters } from '@/composables/useFilters'

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
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('resolutions.title') }}</h1>
      <p>{{ t('resolutions.lead') }}</p>
    </div>

    <FilterBar />

    <section class="section">
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
          <ResolutionTable :items="result.data.value?.items ?? []" :vote-columns="voteColumns" />
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
</template>

<style scoped>
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
