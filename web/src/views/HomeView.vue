<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useCountryIndex, useMembersOn, useRecentResolutions, useResolution, useStats } from '@/api/queries'
import CountryPicker from '@/components/CountryPicker.vue'
import QueryState from '@/components/QueryState.vue'
import ResolutionTable from '@/components/ResolutionTable.vue'
import VoteBar from '@/components/VoteBar.vue'
import VoteBoard from '@/components/VoteBoard.vue'
import { cleanTitle } from '@/format'

const { t, d, n } = useI18n()
const router = useRouter()

const recent = useRecentResolutions(40)
const stats = useStats()
const { name } = useCountryIndex()

// The hero shows the latest resolution adopted by recorded vote: most are adopted without one,
// and those have no per-country votes to show.
const latestVotedId = computed(
  () => recent.data.value?.items.find((r) => r.yes != null)?.undl_id ?? null,
)
const latest = useResolution(latestVotedId)
const membersOn = useMembersOn()
const absentCount = computed(
  () =>
    Object.values(membersOn(latest.data.value?.all_votes ?? {}, latest.data.value?.date)).filter(
      (v) => v === 'X',
    ).length,
)

function openCountry(iso3: string | null) {
  if (iso3) router.push({ name: 'country', params: { iso3 } })
}
</script>

<template>
  <div class="page home">
    <div class="page-head intro">
      <h1>{{ t('home.title') }}</h1>
      <p v-if="stats.data.value">
        {{
          t('home.lead', {
            resolutions: n(stats.data.value.total_resolutions),
            from: stats.data.value.year_min,
            to: stats.data.value.year_max,
          })
        }}
      </p>
    </div>

    <section class="latest panel" :aria-labelledby="'latest-title'">
      <QueryState
        :loading="recent.isLoading.value || latest.isLoading.value"
        :error="recent.error.value ?? latest.error.value"
        :empty="!latestVotedId"
        min-height="420px"
        @retry="recent.refetch()"
      >
        <div v-if="latest.data.value" class="latest-grid">
          <div class="latest-text">
            <p class="kicker">{{ t('home.latestVote') }}</p>
            <h2 id="latest-title">{{ cleanTitle(latest.data.value.title) }}</h2>
            <p class="meta">
              <span class="symbol">{{ latest.data.value.symbol }}</span>
              <span v-if="latest.data.value.date">{{ d(new Date(latest.data.value.date), 'long') }}</span>
            </p>
            <VoteBar
              :yes="latest.data.value.yes ?? 0"
              :no="latest.data.value.no ?? 0"
              :abstain="latest.data.value.abstentions ?? 0"
              :absent="absentCount"
            />
            <RouterLink
              class="button"
              :to="{ name: 'resolution', params: { id: latest.data.value.undl_id } }"
            >
              {{ t('home.openResolution') }}
            </RouterLink>
          </div>
          <VoteBoard :votes="latest.data.value.all_votes" :name="name" :show-absent="false" />
        </div>
      </QueryState>
    </section>

    <section class="start" :aria-label="t('home.startLabel')">
      <div class="entry">
        <h2>{{ t('home.byCountry') }}</h2>
        <p>{{ t('home.byCountryText') }}</p>
        <CountryPicker
          :model-value="null"
          :label="t('home.byCountry')"
          hide-label
          @update:model-value="openCountry"
        />
      </div>
      <div class="entry">
        <h2>{{ t('home.byResolution') }}</h2>
        <p>{{ t('home.byResolutionText') }}</p>
        <RouterLink class="button" to="/resolutions">{{ t('home.browse') }}</RouterLink>
      </div>
      <div class="entry">
        <h2>{{ t('home.byTopic') }}</h2>
        <p>{{ t('home.byTopicText') }}</p>
        <RouterLink class="button" to="/topics">{{ t('home.topics') }}</RouterLink>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>{{ t('home.recent') }}</h2>
        <RouterLink to="/resolutions">{{ t('home.allResolutions') }}</RouterLink>
      </div>
      <div class="panel flush">
        <QueryState :loading="recent.isLoading.value" :error="recent.error.value" @retry="recent.refetch()">
          <ResolutionTable :items="recent.data.value?.items.slice(0, 8) ?? []" />
        </QueryState>
      </div>
    </section>
  </div>
</template>

<style scoped>
.intro h1 {
  font-size: var(--text-3xl);
  font-weight: 500;
  max-width: 30ch;
}

.intro p {
  font-size: var(--text-lg);
}

.latest {
  padding: var(--space-6);
}

.latest-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
  gap: var(--space-7);
  align-items: start;
}

.latest-text {
  display: grid;
  gap: var(--space-4);
  justify-items: start;
  position: sticky;
  top: 84px;
}

.kicker {
  color: var(--accent);
  font-weight: 600;
  font-size: var(--text-sm);
}

.latest-text h2 {
  font-size: var(--text-2xl);
}

.meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1) var(--space-3);
  color: var(--ink-2);
  font-size: var(--text-sm);
}

.symbol {
  font-family: var(--font-condensed);
  font-weight: 500;
  color: var(--ink);
}

.start {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  border-top: 1px solid var(--rule-strong);
  border-bottom: 1px solid var(--rule);
}

.entry {
  display: grid;
  gap: var(--space-3);
  align-content: start;
  justify-items: start;
  padding: var(--space-5) var(--space-5) var(--space-5) 0;
}

.entry + .entry {
  padding-inline-start: var(--space-5);
  border-inline-start: 1px solid var(--rule);
}

.entry h2 {
  font-size: var(--text-lg);
}

.entry p {
  color: var(--ink-2);
  font-size: var(--text-sm);
}

.entry :deep(.picker) {
  width: 100%;
  max-width: 280px;
}

.flush {
  padding: 0;
  overflow: hidden;
}

@media (max-width: 640px) {
  .intro h1 {
    font-size: var(--text-2xl);
  }
}

@media (max-width: 900px) {
  .latest {
    padding: var(--space-4);
  }

  .latest-grid,
  .start {
    grid-template-columns: 1fr;
  }

  .latest-text {
    position: static;
  }

  .entry,
  .entry + .entry {
    padding: var(--space-4) 0;
    border-inline-start: 0;
  }

  .entry + .entry {
    border-top: 1px solid var(--rule);
  }
}
</style>
