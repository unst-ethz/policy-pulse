<script setup lang="ts">
/** Resolutions as a table: when, which, what, how it was adopted, and chosen countries' votes. */
import { useI18n } from 'vue-i18n'
import type { Resolution } from '@/api/client'
import { cleanTitle } from '@/format'
import VoteChip from './VoteChip.vue'

defineProps<{ items: Resolution[]; voteColumns?: { iso3: string; label: string }[]; sponsors?: boolean }>()
const { t, d, n } = useI18n()

function outcome(r: Resolution): string {
  if (r.yes == null) return r.modality ?? t('resolution.noRecordedVote')
  return t('resolution.tally', { yes: r.yes, no: r.no ?? 0, abstain: r.abstentions ?? 0 })
}
</script>

<template>
  <div class="table-wrap">
    <table class="data-table resolutions">
      <thead>
        <tr>
          <th scope="col">{{ t('resolution.date') }}</th>
          <th scope="col">{{ t('resolution.title') }}</th>
          <th scope="col">{{ t('resolution.outcome') }}</th>
          <th scope="col" class="numeric">
            <abbr :title="t('resolution.consensusHelp')">{{ t('resolution.consensus') }}</abbr>
          </th>
          <th v-if="sponsors" scope="col" class="numeric">{{ t('resolution.sponsorCount') }}</th>
          <th v-for="col in voteColumns ?? []" :key="col.iso3" scope="col">{{ col.label }}</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in items" :key="r.undl_id">
          <td class="date">{{ r.date ? d(new Date(r.date), 'short') : '' }}</td>
          <td class="what">
            <RouterLink :to="{ name: 'resolution', params: { id: r.undl_id } }" class="symbol">
              {{ r.symbol ?? r.undl_id }}
            </RouterLink>
            <span class="title">{{ cleanTitle(r.title) }}</span>
          </td>
          <td class="outcome" :class="{ unrecorded: r.yes == null }">{{ outcome(r) }}</td>
          <td class="numeric">{{ r.consensus_score == null ? '' : n(r.consensus_score, 'score') }}</td>
          <!-- null is "not known", never zero sponsors -->
          <td v-if="sponsors" class="numeric" :class="{ unrecorded: r.sponsor_count == null }">
            {{ r.sponsor_count == null ? t('resolution.unknown') : n(r.sponsor_count) }}
          </td>
          <td v-for="col in voteColumns ?? []" :key="col.iso3">
            <!-- No chip without a recorded vote: nobody cast an individual vote to show. -->
            <VoteChip v-if="r.yes != null" :vote="r.votes?.[col.iso3]" />
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.table-wrap {
  overflow-x: auto;
}

.resolutions td {
  vertical-align: top;
}

.date {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  color: var(--ink-2);
}

.what {
  display: grid;
  gap: 2px;
  min-width: 18rem;
}

.symbol {
  font-family: var(--font-condensed);
  font-weight: 500;
  white-space: nowrap;
}

.title {
  color: var(--ink-2);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  max-width: 70ch;
}

.outcome {
  white-space: nowrap;
}

.outcome.unrecorded,
.numeric.unrecorded {
  color: var(--muted);
}
</style>
