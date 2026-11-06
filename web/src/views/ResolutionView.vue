<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import type { Vote } from '@/api/client'
import { useCountryIndex, useMembersOn, useResolution } from '@/api/queries'
import ChartFrame from '@/components/ChartFrame.vue'
import QueryState from '@/components/QueryState.vue'
import VoteBar from '@/components/VoteBar.vue'
import VoteBoard from '@/components/VoteBoard.vue'
import WorldMap from '@/components/WorldMap.vue'
import { useThemeColors } from '@/composables/useThemeColors'
import { cleanTitle, sentenceCase } from '@/format'

const props = defineProps<{ id: string }>()
const { t, d } = useI18n()
const router = useRouter()
const resolution = useResolution(() => props.id)
const { name } = useCountryIndex()
const colors = useThemeColors()

const r = computed(() => resolution.data.value)
const membersOn = useMembersOn()
// Only states that were members when it was adopted (see useMembersOn).
const votes = computed(() => membersOn(r.value?.all_votes ?? {}, r.value?.date))
const recorded = computed(() => r.value?.yes != null)
const absent = computed(() => Object.values(votes.value).filter((v) => v === 'X').length)

const fillByVote = computed<Record<Vote, string | null>>(() => ({
  Y: colors.yes,
  N: colors.no,
  A: colors.abstain,
  X: null,
}))
const fill = (iso3: string) => {
  const vote = votes.value[iso3]
  return vote ? fillByVote.value[vote] : null
}
const tooltip = (iso3: string) => {
  const vote = votes.value[iso3]
  return `<b>${name(iso3)}</b><br>${vote ? t(`vote.${vote}`) : t('map.noData')}`
}

/** A keyword search for exactly this term (the API's quoted-term syntax). */
const exactTerm = (term: string) => `"${term}"`

const keywordGroups = computed(() =>
  Object.entries(r.value?.keywords ?? {}).filter(([, terms]) => terms.length),
)
const sponsors = computed(() => r.value?.sponsors ?? [])
const originals = computed(() => sponsors.value.filter((s) => s.kind === 'original'))
const additionals = computed(() => sponsors.value.filter((s) => s.kind === 'additional'))
const sponsorKind = computed(() => new Map(sponsors.value.filter((s) => s.country).map((s) => [s.country!, s.kind])))
const sponsorStates = computed(() => [...sponsorKind.value.keys()])
// The ingest's placeholder notes, in plain words; anything new falls back to the generic line.
const noteText = computed(() => {
  switch (r.value?.sponsorship_note) {
    case 'no draft backlinked, report to library':
      return t('resolution.noteNoDraft')
    case 'draft symbol not in catalogue, report to library':
      return t('resolution.noteDraftMissing')
    case 'draft has no authors catalogued':
      return t('resolution.noteNoAuthors')
    case 'linked record is not a draft, report to library':
      return t('resolution.noteNotDraft')
    default:
      return t('resolution.sponsorsUnknown')
  }
})

const tableRows = computed(() =>
  Object.entries(votes.value)
    .filter(([, v]) => v !== 'X')
    .map(([iso3, vote]) => ({ iso3, name: name(iso3), vote, sponsor: sponsorKind.value.get(iso3) }))
    .sort((a, b) => a.name.localeCompare(b.name)),
)
</script>

<template>
  <div class="page">
    <QueryState
      :loading="resolution.isLoading.value"
      :error="resolution.error.value"
      min-height="300px"
      @retry="resolution.refetch()"
    >
      <div v-if="r" class="stack">
        <div class="page-head">
          <p class="symbol">{{ r.symbol }}</p>
          <h1>{{ cleanTitle(r.title) }}</h1>
          <dl class="facts">
            <div v-if="r.date">
              <dt>{{ t('resolution.adopted') }}</dt>
              <dd>{{ d(new Date(r.date), 'long') }}</dd>
            </div>
            <div v-if="r.session">
              <dt>{{ t('resolution.session') }}</dt>
              <dd>{{ r.session }}</dd>
            </div>
            <div v-if="r.modality">
              <dt>{{ t('resolution.how') }}</dt>
              <dd>{{ r.modality }}</dd>
            </div>
            <div v-if="r.agenda_title">
              <dt>{{ t('resolution.agenda') }}</dt>
              <dd>{{ r.agenda_title }}</dd>
            </div>
            <div v-if="r.draft">
              <dt>{{ t('resolution.draft') }}</dt>
              <dd>{{ r.draft }}</dd>
            </div>
          </dl>
          <p v-if="r.undl_link">
            <a :href="r.undl_link" target="_blank" rel="noopener">{{ t('resolution.library') }}</a>
          </p>
        </div>

        <template v-if="recorded">
          <section class="section">
            <VoteBar
              :yes="r.yes ?? 0"
              :no="r.no ?? 0"
              :abstain="r.abstentions ?? 0"
              :absent="absent"
            />
          </section>

          <ChartFrame :title="t('resolution.mapTitle')" :description="t('resolution.mapText')">
            <WorldMap
              :fill="fill"
              :tooltip="tooltip"
              @select="(iso3) => router.push({ name: 'country', params: { iso3 } })"
            />
            <template #table>
              <table class="data-table">
                <thead>
                  <tr>
                    <th scope="col">{{ t('country.country') }}</th>
                    <th scope="col">{{ t('resolution.vote') }}</th>
                    <th v-if="sponsors.length" scope="col">{{ t('resolution.sponsor') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in tableRows" :key="row.iso3">
                    <td>
                      <RouterLink :to="{ name: 'country', params: { iso3: row.iso3 } }">{{ row.name }}</RouterLink>
                    </td>
                    <td>{{ t(`vote.${row.vote}`) }}</td>
                    <td v-if="sponsors.length">{{ row.sponsor ? t(`resolution.kinds.${row.sponsor}`) : '' }}</td>
                  </tr>
                </tbody>
              </table>
            </template>
            <template #note>
              <p>{{ t('map.disclaimer') }}</p>
            </template>
          </ChartFrame>

          <section class="section panel">
            <h2>{{ t('resolution.boardTitle') }}</h2>
            <p v-if="sponsorStates.length" class="hint">
              <span class="dot" aria-hidden="true" /> {{ t('resolution.boardSponsors') }}
            </p>
            <VoteBoard :votes="votes" :name="name" :marked="sponsorStates" :marked-label="t('resolution.sponsor')" />
          </section>
        </template>
        <p v-else class="panel unrecorded">{{ t('resolution.noVotes') }}</p>

        <section class="section panel">
          <h2>{{ t('resolution.sponsors') }}</h2>
          <template v-if="sponsors.length">
            <div v-for="group in [{ key: 'original', list: originals }, { key: 'additional', list: additionals }]" :key="group.key">
              <template v-if="group.list.length">
                <h3>{{ t(`resolution.sponsorGroups.${group.key}`, { n: group.list.length }) }}</h3>
                <ul class="tags sponsor-list">
                  <li v-for="s in group.list" :key="s.name">
                    <RouterLink v-if="s.country" class="tag" :to="{ name: 'country', params: { iso3: s.country } }">
                      {{ s.name }}
                    </RouterLink>
                    <span v-else class="tag plain">
                      {{ s.name }} <small>{{ t(`resolution.sponsorTypes.${s.type}`) }}</small>
                    </span>
                  </li>
                </ul>
              </template>
            </div>
            <p class="hint">{{ t('resolution.sponsorsText') }}</p>
          </template>
          <template v-else>
            <p>{{ noteText }}</p>
            <p class="hint">{{ t('resolution.sponsorsCoverage') }}</p>
          </template>
        </section>

        <section v-if="r.subjects.length || keywordGroups.length" class="section">
          <h2>{{ t('resolution.about') }}</h2>
          <div v-if="r.subjects.length" class="tags">
            <span class="tag-label">{{ t('resolution.subjects') }}</span>
            <RouterLink
              v-for="s in r.subjects"
              :key="s.id"
              class="tag"
              :to="{ name: 'resolutions', query: { subject: s.id } }"
            >
              {{ sentenceCase(s.label) }}
            </RouterLink>
          </div>
          <div v-for="[dimension, terms] in keywordGroups" :key="dimension" class="tags">
            <span class="tag-label">{{ t(`cloud.modes.${dimension === 'general' ? 'default' : dimension}`) }}</span>
            <RouterLink
              v-for="term in terms"
              :key="term"
              class="tag"
              :to="{ name: 'resolutions', query: { q: exactTerm(term) } }"
            >
              {{ term }}
            </RouterLink>
          </div>
        </section>
      </div>
    </QueryState>
  </div>
</template>

<style scoped>
.stack {
  display: grid;
  gap: var(--space-6);
}

.symbol {
  font-family: var(--font-condensed);
  font-weight: 500;
  font-size: var(--text-lg);
  color: var(--accent);
}

h1 {
  max-width: 40ch;
}

.facts {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3) var(--space-6);
  margin: var(--space-2) 0 0;
  font-size: var(--text-sm);
}

.facts dt {
  color: var(--muted);
}

.facts dd {
  margin: 0;
  max-width: 40ch;
}

.unrecorded {
  color: var(--ink-2);
}

.tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
}

.tag-label {
  min-width: 7rem;
  color: var(--muted);
  font-size: var(--text-sm);
}

.tag {
  padding: 2px 10px;
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  background: var(--surface);
  font-size: var(--text-sm);
  text-decoration: none;
}

.sponsor-list {
  list-style: none;
  margin: var(--space-2) 0 var(--space-4);
  padding: 0;
}

.tag.plain small {
  color: var(--muted);
}

.hint {
  color: var(--muted);
  font-size: var(--text-xs);
}

.dot {
  display: inline-block;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--ink-2);
  vertical-align: middle;
}

.tag:hover {
  border-color: var(--ink-2);
}
</style>
