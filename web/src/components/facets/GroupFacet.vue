<script setup lang="ts">
/**
 * How a group of states voted together: pick countries (a region adds all of its states), then
 * whether they all voted alike, split, or (from three states) mostly voted one way. With two
 * countries this is the bilateral case. States that did not vote are left out, unless all of them must have voted.
 * Up to four picked countries also become the list's vote columns, unless the viewer chose others.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import type { TreeNode } from '@/api/client'
import { useCountryIndex, useFacets, useRegionTree } from '@/api/queries'
import { GROUP_VOTES, type GroupVote, useFilters, usePageFilters } from '@/composables/useFilters'
import CountryList from '../CountryList.vue'
import FacetPanel from './FacetPanel.vue'

const VOTE_COLUMNS = 4
// A majority of two is just both of them: offered from three states up
const MAJORITIES: GroupVote[] = ['yes', 'no', 'abstain']

const { t, n } = useI18n()
const route = useRoute()
const { filters, update } = useFilters()
const facets = useFacets(usePageFilters())
const tree = useRegionTree()
const { index } = useCountryIndex()

// Continents, each with its sub-regions (the M49 tree below "World")
const continents = computed(() => tree.data.value?.find((node) => node.id === '001')?.children ?? [])
const regionNodes = computed(
  () => new Map(continents.value.flatMap((c) => [c, ...(c.children ?? [])]).map((r) => [r.id, r])),
)
// A region's states that have votes in the data (the tree also lists territories)
const leaves = (node: TreeNode): string[] => (node.children?.length ? node.children.flatMap(leaves) : [node.id])
const members = (code: string) => {
  const node = regionNodes.value.get(code)
  return node ? leaves(node).filter((iso3) => index.value.has(iso3)) : []
}

const countries = computed({
  get: () => filters.value.group,
  set: (value: string[]) => {
    // The vote columns follow the group's countries until the viewer adds others of their own;
    // a group larger than the columns gets none, rather than a few of its states picked at random
    const v = route.query.votes
    const columns = (Array.isArray(v) ? v : v ? [v] : []).filter((c): c is string => !!c)
    const ours = columns.every((c) => countries.value.includes(c))
    const votes = value.length && value.length <= VOTE_COLUMNS ? value : undefined
    update({ group: value }, ours ? { votes } : {})
  },
})

// A region is a shortcut: it adds each of its states to the list, where they can be removed one by one
function addRegion(select: HTMLSelectElement) {
  const code = select.value
  select.value = '' // back to the "Add a region" prompt
  if (code) countries.value = [...new Set([...countries.value, ...members(code)])]
}
const complete = (code: string) => members(code).every((iso3) => countries.value.includes(iso3))
// The chosen way stays listed even if the group shrank below three, so it can be changed
const votes = computed(() =>
  GROUP_VOTES.filter((v) => !MAJORITIES.includes(v) || countries.value.length >= 3 || filters.value.together === v),
)
const counts = computed(
  () => new Map((facets.data.value?.group_votes ?? []).map((g) => [g.group_vote, g.resolutions])),
)
</script>

<template>
  <FacetPanel
    :info="t('facets.info.group')"
    :title="t('facets.group')"
    :active="filters.group.length > 0"
    @clear="update({ group: [], together: null, allVoted: false })"
  >
    <CountryList v-model="countries" :label="t('facets.groupCountries')" :max="Infinity" />
    <select
      class="input"
      :aria-label="t('facets.addRegion')"
      @change="addRegion($event.target as HTMLSelectElement)"
    >
      <option value="">{{ t('facets.addRegion') }}</option>
      <optgroup v-for="c in continents" :key="c.id" :label="c.label">
        <option :value="c.id" :disabled="complete(c.id)">{{ t('facets.wholeRegion', { name: c.label }) }}</option>
        <option v-for="s in c.children ?? []" :key="s.id" :value="s.id" :disabled="complete(s.id)">
          {{ s.label }}
        </option>
      </optgroup>
    </select>

    <p v-if="countries.length < 2" class="hint">{{ t('facets.groupHint') }}</p>
    <template v-else>
      <div class="options" :class="{ fetching: facets.isFetching.value }" role="radiogroup" :aria-label="t('facets.group')">
        <label v-for="vote in votes" :key="vote" class="row">
          <input
            type="radio"
            name="facet-group-vote"
            :value="vote"
            :checked="filters.together === vote"
            @change="update({ together: vote as GroupVote })"
          />
          <span>{{ t(`facets.groupVotes.${vote}`) }}</span>
          <span class="count">{{ n(counts.get(vote) ?? 0) }}</span>
        </label>
      </div>
      <label class="row all">
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
