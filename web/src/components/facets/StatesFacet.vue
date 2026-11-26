<script setup lang="ts">
/**
 * The States the list is about. Picking them only adds their votes as columns; the radio below is
 * the separate, optional filter: for one State its point of view (voted on, while a member, how
 * it voted), for
 * several how they voted together (alike, split, or from three States a majority one way).
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { TreeNode } from '@/api/client'
import { useCountryIndex, useFacets, useRegionTree } from '@/api/queries'
import {
  type CountryMode,
  GROUP_VOTES,
  type GroupVote,
  MAX_VOTE_COLUMNS,
  statesOf,
  useFilters,
  usePageFilters,
  withStates,
} from '@/composables/useFilters'
import { GROUPS } from '@/groups'
import CountryList from '../CountryList.vue'
import FacetPanel from './FacetPanel.vue'

const MODES: CountryMode[] = ['none', 'voted', 'member', 'yes', 'no', 'abstain']
// A majority of two is just both of them: offered from three States up
const MAJORITIES: GroupVote[] = ['yes', 'no', 'abstain']

const { t, n } = useI18n()
const { filters, update } = useFilters()
const facets = useFacets(usePageFilters())
const tree = useRegionTree()
const { index } = useCountryIndex()

const states = computed({
  get: () => statesOf(filters.value),
  set: (value: string[]) => update(withStates(filters.value, value)),
})

// Continents, each with its sub-regions (the M49 tree below "World")
const continents = computed(() => tree.data.value?.find((node) => node.id === '001')?.children ?? [])
const regionNodes = computed(
  () => new Map(continents.value.flatMap((c) => [c, ...(c.children ?? [])]).map((r) => [r.id, r])),
)
// A region's States that have votes in the data (the tree also lists territories)
const leaves = (node: TreeNode): string[] => (node.children?.length ? node.children.flatMap(leaves) : [node.id])
// Options are `group:<key>` or a region code
const members = (option: string) => {
  const key = option.startsWith('group:') ? (option.slice(6) as keyof typeof GROUPS) : null
  const node = key ? null : regionNodes.value.get(option)
  const codes = key ? [...GROUPS[key]] : node ? leaves(node) : []
  return codes.filter((iso3) => index.value.has(iso3))
}
// A region or group is a shortcut: it adds each of its States, which can then be removed one by one
function addMembers(select: HTMLSelectElement) {
  const option = select.value
  select.value = '' // back to the prompt
  if (option) states.value = [...new Set([...states.value, ...members(option)])]
}
const complete = (option: string) => members(option).every((iso3) => states.value.includes(iso3))

// "Opposed" (Y/N or N/Y) is offered for a pair only, majorities from three States up. The chosen
// way stays listed even when the count changes, so it can be changed.
const offered = (v: GroupVote) =>
  v === 'opposed' ? states.value.length === 2 : !MAJORITIES.includes(v) || states.value.length >= 3
const groupVotes = computed(() => GROUP_VOTES.filter((v) => offered(v) || filters.value.together === v))
const counts = computed(
  () => new Map((facets.data.value?.group_votes ?? []).map((g) => [g.group_vote, g.resolutions])),
)
const modeCounts = computed(
  () => new Map((facets.data.value?.country_modes ?? []).map((m) => [m.country_mode, m.resolutions])),
)
</script>

<template>
  <FacetPanel
    :info="t('facets.info.states')"
    :title="t('facets.states')"
    :active="states.length > 0"
    @clear="states = []"
  >
    <CountryList v-model="states" :label="t('facets.states')" :max="Infinity" />
    <select class="input" :aria-label="t('facets.addMany')" @change="addMembers($event.target as HTMLSelectElement)">
      <option value="">{{ t('facets.addMany') }}</option>
      <optgroup :label="t('groups.title')">
        <option v-for="key in Object.keys(GROUPS)" :key="key" :value="`group:${key}`" :disabled="complete(`group:${key}`)">
          {{ t(`groups.${key}`) }}
        </option>
      </optgroup>
      <optgroup v-for="c in continents" :key="c.id" :label="c.label">
        <option :value="c.id" :disabled="complete(c.id)">{{ t('facets.wholeRegion', { name: c.label }) }}</option>
        <option v-for="s in c.children ?? []" :key="s.id" :value="s.id" :disabled="complete(s.id)">
          {{ s.label }}
        </option>
      </optgroup>
    </select>

    <p v-if="states.length > MAX_VOTE_COLUMNS" class="hint">{{ t('facets.tooManyColumns', { n: MAX_VOTE_COLUMNS }) }}</p>

    <div
      v-if="states.length === 1"
      class="options"
      :class="{ fetching: facets.isFetching.value }"
      role="radiogroup"
      :aria-label="t('filters.mode')"
    >
      <label v-for="mode in MODES" :key="mode" class="row">
        <input type="radio" name="facet-states-mode" :checked="filters.mode === mode" @change="update({ mode })" />
        <span>{{ mode === 'none' ? t('facets.anyResolution') : t(`filters.modes.${mode}`) }}</span>
        <span v-if="mode !== 'none'" class="count">{{ n(modeCounts.get(mode) ?? 0) }}</span>
      </label>
    </div>
    <template v-else-if="states.length > 1">
      <div class="options" :class="{ fetching: facets.isFetching.value }" role="radiogroup" :aria-label="t('facets.group')">
        <label class="row">
          <input type="radio" name="facet-states-vote" :checked="!filters.together" @change="update({ together: null })" />
          <span>{{ t('facets.anyResolution') }}</span>
        </label>
        <label v-for="vote in groupVotes" :key="vote" class="row">
          <input type="radio" name="facet-states-vote" :checked="filters.together === vote" @change="update({ together: vote })" />
          <span>{{ t(`facets.groupVotes.${vote}`) }}</span>
          <span class="count">{{ n(counts.get(vote) ?? 0) }}</span>
        </label>
      </div>
      <label v-if="filters.together" class="row all">
        <input
          type="checkbox"
          :checked="filters.allVoted"
          @change="update({ allVoted: ($event.target as HTMLInputElement).checked })"
        />
        <span>{{ t('facets.allVoted') }}</span>
      </label>
    </template>
  </FacetPanel>
</template>

<style scoped>
.hint {
  margin: 0;
  font-size: var(--text-xs);
  color: var(--muted);
}

.options {
  display: grid;
  gap: 2px;
  transition: opacity 0.15s;
}

.options.fetching {
  opacity: 0.6;
}

.row {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  column-gap: var(--space-2);
  padding-block: 3px;
  font-size: var(--text-sm);
  cursor: pointer;
}

.row input {
  margin: 0;
  accent-color: var(--ink);
}

.row.all {
  grid-template-columns: auto 1fr;
  color: var(--ink-2);
}

.count {
  color: var(--ink-2);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}
</style>
