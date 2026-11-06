<script setup lang="ts">
/**
 * How a group of states voted together: pick countries and/or regions, then whether they all
 * voted alike, split, or (from three states) mostly voted one way. With two countries this is
 * the bilateral case. States that did not vote are left out, unless all of them must have voted.
 * Picked countries also become the list's vote columns, unless the viewer chose other ones.
 */
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import type { TreeNode } from '@/api/client'
import { useFacets, useRegionTree } from '@/api/queries'
import { GROUP_VOTES, type GroupVote, groupActive, useFilters } from '@/composables/useFilters'
import CountryList from '../CountryList.vue'
import FacetPanel from './FacetPanel.vue'

const MAX_COUNTRIES = 10
const VOTE_COLUMNS = 4
// A majority of two is just both of them: offered from three states up
const MAJORITIES: GroupVote[] = ['yes', 'no', 'abstain']

const { t, n } = useI18n()
const route = useRoute()
const { filters, update } = useFilters()
const facets = useFacets(filters)
const tree = useRegionTree()

const isCountry = (code: string) => /^[A-Z]{3}$/.test(code)

// Continents, each with its sub-regions (the M49 tree below "World")
const continents = computed(() => tree.data.value?.find((node) => node.id === '001')?.children ?? [])
const regionNodes = computed(
  () => new Map(continents.value.flatMap((c) => [c, ...(c.children ?? [])]).map((r) => [r.id, r])),
)
const regionLabel = (code: string) => regionNodes.value.get(code)?.label ?? code
const regions = computed(() => filters.value.group.filter((code) => !isCountry(code)))

const countries = computed({
  get: () => filters.value.group.filter(isCountry),
  set: (value: string[]) => {
    // The vote columns follow the group's countries until the viewer adds others of their own
    const v = route.query.votes
    const columns = (Array.isArray(v) ? v : v ? [v] : []).filter((c): c is string => !!c)
    const ours = columns.every((c) => countries.value.includes(c))
    const votes = value.length ? value.slice(0, VOTE_COLUMNS) : undefined
    update({ group: [...value, ...regions.value] }, ours ? { votes } : {})
  },
})

function addRegion(select: HTMLSelectElement) {
  const code = select.value
  select.value = '' // back to the "Add a region" prompt
  if (code && !filters.value.group.includes(code)) update({ group: [...filters.value.group, code] })
}
function removeRegion(code: string) {
  update({ group: filters.value.group.filter((c) => c !== code) })
}

// How many states the group stands for, regions counted by their members
const leaves = (node: TreeNode): string[] => (node.children?.length ? node.children.flatMap(leaves) : [node.id])
const states = computed(
  () =>
    new Set([
      ...countries.value,
      ...regions.value.flatMap((code) => {
        const node = regionNodes.value.get(code)
        return node ? leaves(node) : []
      }),
    ]).size,
)
// The chosen way stays listed even if the group shrank below three, so it can be changed
const votes = computed(() =>
  GROUP_VOTES.filter((v) => !MAJORITIES.includes(v) || states.value >= 3 || filters.value.together === v),
)
const counts = computed(
  () => new Map((facets.data.value?.group_votes ?? []).map((g) => [g.group_vote, g.resolutions])),
)
</script>

<template>
  <FacetPanel
    :info="t('facets.info.group')"
    :title="t('facets.group')"
    :active="groupActive(filters)"
    @clear="update({ group: [], together: null, allVoted: false })"
  >
    <CountryList v-model="countries" :label="t('facets.groupCountries')" :max="MAX_COUNTRIES" />
    <ul v-if="regions.length" class="chips" :aria-label="t('facets.groupRegions')">
      <li v-for="code in regions" :key="code" class="chip">
        {{ regionLabel(code) }}
        <button type="button" :aria-label="t('picker.remove', { name: regionLabel(code) })" @click="removeRegion(code)">
          ×
        </button>
      </li>
    </ul>
    <select
      class="input"
      :aria-label="t('facets.addRegion')"
      @change="addRegion($event.target as HTMLSelectElement)"
    >
      <option value="">{{ t('facets.addRegion') }}</option>
      <optgroup v-for="c in continents" :key="c.id" :label="c.label">
        <option :value="c.id" :disabled="regions.includes(c.id)">{{ t('facets.wholeRegion', { name: c.label }) }}</option>
        <option v-for="s in c.children ?? []" :key="s.id" :value="s.id" :disabled="regions.includes(s.id)">
          {{ s.label }}
        </option>
      </optgroup>
    </select>

    <p v-if="states < 2" class="hint">{{ t('facets.groupHint') }}</p>
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
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
  margin: 0;
  padding: 0;
  list-style: none;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 32px;
  padding-inline: 10px 4px;
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  background: var(--surface);
  font-size: var(--text-sm);
}

.chip button {
  width: 24px;
  height: 24px;
  border: 0;
  border-radius: 50%;
  background: none;
  color: var(--muted);
  cursor: pointer;
}

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
