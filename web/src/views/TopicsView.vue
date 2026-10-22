<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { type CloudMode, useWordCloud } from '@/api/queries'
import ChartFrame from '@/components/ChartFrame.vue'
import FilterBar from '@/components/FilterBar.vue'
import QueryState from '@/components/QueryState.vue'
import WordCloud from '@/components/WordCloud.vue'
import type { CloudWord } from '@/charts/types'
import { serializeFilters, useFilters } from '@/composables/useFilters'

const MODES: CloudMode[] = ['default', 'thematic', 'geopolitical', 'action', 'category']

const { t, n } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()

const mode = computed<CloudMode>({
  get: () => (MODES.includes(route.query.mode as CloudMode) ? (route.query.mode as CloudMode) : 'default'),
  set: (value) => router.replace({ query: { ...route.query, mode: value === 'default' ? undefined : value } }),
})

const cloud = useWordCloud(filters, mode)

/** Where following a term goes: the resolution list, narrowed by the term on top of the filters. */
function target(term: string, subjectIds: string[]) {
  const base = serializeFilters(filters.value)
  if (mode.value === 'category' && subjectIds.length) {
    return { name: 'resolutions', query: { ...base, subject: subjectIds } }
  }
  const exact = `"${term}"`
  const q = filters.value.keyword.trim() ? `${filters.value.keyword.trim()} & ${exact}` : exact
  return { name: 'resolutions', query: { ...base, q } }
}

const words = computed<CloudWord[]>(() =>
  (cloud.data.value?.terms ?? []).map((term) => ({
    term: term.term,
    count: term.count,
    to: target(term.term, term.subject_ids ?? []),
    title: t('topics.termTitle', { term: term.term, n: term.count }),
  })),
)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('topics.title') }}</h1>
      <p>{{ t('topics.lead') }}</p>
    </div>

    <FilterBar :fields="['years', 'subjects', 'keyword', 'country']" />

    <ChartFrame
      :title="t('topics.cloudTitle')"
      :description="
        cloud.data.value ? t('topics.cloudText', { n: n(cloud.data.value.resolution_count) }) : undefined
      "
    >
      <template #controls>
        <div class="modes" role="tablist" :aria-label="t('topics.mode')">
          <button
            v-for="m in MODES"
            :key="m"
            type="button"
            role="tab"
            :aria-selected="mode === m"
            @click="mode = m"
          >
            {{ t(`topics.modes.${m}`) }}
          </button>
        </div>
      </template>
      <QueryState
        :loading="cloud.isLoading.value"
        :fetching="cloud.isFetching.value"
        :error="cloud.error.value"
        :empty="cloud.data.value?.terms.length === 0"
        :empty-text="t('topics.empty')"
        min-height="380px"
        @retry="cloud.refetch()"
      >
        <WordCloud :words="words" />
      </QueryState>
      <template #table>
        <table class="data-table">
          <thead>
            <tr>
              <th scope="col">{{ t('topics.term') }}</th>
              <th scope="col" class="numeric">{{ t('topics.found') }}</th>
              <th scope="col" class="numeric">{{ t('topics.tagged') }}</th>
              <th scope="col" class="numeric">{{ t('topics.consensus') }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="term in cloud.data.value?.terms ?? []" :key="term.term">
              <td>
                <RouterLink :to="target(term.term, term.subject_ids ?? [])">{{ term.term }}</RouterLink>
              </td>
              <td class="numeric">{{ n(term.count) }}</td>
              <td class="numeric">{{ n(term.frequency) }}</td>
              <td class="numeric">{{ term.mean_consensus == null ? '' : n(term.mean_consensus, 'score') }}</td>
            </tr>
          </tbody>
        </table>
      </template>
      <template #note>
        <p>{{ t('topics.note') }}</p>
      </template>
    </ChartFrame>
  </div>
</template>

<style scoped>
.modes {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 2px;
  padding: 2px;
  background: var(--surface-sunk);
  border: 1px solid var(--rule);
  border-radius: var(--radius-s);
}

.modes button {
  border: 0;
  background: transparent;
  padding: 4px 10px;
  border-radius: 2px;
  font-size: var(--text-xs);
  color: var(--ink-2);
  cursor: pointer;
}

.modes button[aria-selected='true'] {
  background: var(--surface);
  color: var(--ink);
  box-shadow: 0 0 0 1px var(--rule-strong);
  font-weight: 500;
}
</style>
