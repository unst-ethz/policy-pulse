<script setup lang="ts">
/**
 * The most frequent terms of the filtered resolutions. Following a term narrows the resolution
 * list by it, on top of the filters; the other list options (sort, votes) are kept.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { type CloudMode, useWordCloud } from '@/api/queries'
import ChartFrame from './ChartFrame.vue'
import QueryState from './QueryState.vue'
import WordCloud from './WordCloud.vue'
import type { CloudWord } from '@/charts/types'
import { hasTerm, toggleTerm, useFilters } from '@/composables/useFilters'

const MODES: CloudMode[] = ['default', 'thematic', 'geopolitical', 'action', 'category']

const { t, n } = useI18n()
const route = useRoute()
const router = useRouter()
const { filters } = useFilters()

// `cloud`, not `mode`: the filters already use `mode` for the country participation mode.
const mode = computed<CloudMode>({
  get: () => (MODES.includes(route.query.cloud as CloudMode) ? (route.query.cloud as CloudMode) : 'default'),
  set: (value) => router.replace({ query: { ...route.query, cloud: value === 'default' ? undefined : value } }),
})

const cloud = useWordCloud(filters, mode)

/** Where following a term goes: this list, narrowed by the term on top of the filters. */
function target(term: string, subjectIds: string[]) {
  const query = { ...route.query, page: undefined }
  if (mode.value === 'category' && subjectIds.length) return { query: { ...query, subject: subjectIds } }
  return { query: { ...query, q: toggleTerm(filters.value.keyword, term) } }
}

// Terms already in the keyword filter match every listed resolution: they say nothing here
const terms = computed(() =>
  (cloud.data.value?.terms ?? []).filter((term) => !hasTerm(filters.value.keyword, term.term)),
)

const words = computed<CloudWord[]>(() =>
  terms.value.map((term) => ({
    term: term.term,
    count: term.count,
    to: target(term.term, term.subject_ids ?? []),
    title: t('cloud.termTitle', { term: term.term, n: term.count }),
  })),
)
</script>

<template>
  <ChartFrame
    :title="t('cloud.cloudTitle')"
    :description="
      cloud.data.value ? t('cloud.cloudText', { n: n(cloud.data.value.resolution_count) }) : undefined
    "
  >
    <template #controls>
      <div class="modes" role="tablist" :aria-label="t('cloud.mode')">
        <button
          v-for="m in MODES"
          :key="m"
          type="button"
          role="tab"
          :aria-selected="mode === m"
          @click="mode = m"
        >
          {{ t(`cloud.modes.${m}`) }}
        </button>
      </div>
    </template>
    <QueryState
      :loading="cloud.isLoading.value"
      :fetching="cloud.isFetching.value"
      :error="cloud.error.value"
      :empty="cloud.data.value != null && terms.length === 0"
      :empty-text="t('cloud.empty')"
      min-height="380px"
      @retry="cloud.refetch()"
    >
      <WordCloud :words="words" />
    </QueryState>
    <template #table>
      <table class="data-table">
        <thead>
          <tr>
            <th scope="col">{{ t('cloud.term') }}</th>
            <th scope="col" class="numeric">{{ t('cloud.found') }}</th>
            <th scope="col" class="numeric">{{ t('cloud.tagged') }}</th>
            <th scope="col" class="numeric">{{ t('cloud.consensus') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="term in terms" :key="term.term">
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
      <p>{{ t('cloud.note') }}</p>
    </template>
  </ChartFrame>
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
