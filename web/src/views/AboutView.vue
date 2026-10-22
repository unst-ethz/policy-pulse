<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useMeta } from '@/api/queries'

const { t, d, n } = useI18n()
const meta = useMeta()
</script>

<template>
  <div class="page about">
    <div class="page-head">
      <h1>{{ t('about.title') }}</h1>
      <p>{{ t('about.lead') }}</p>
    </div>

    <section class="prose">
      <h2>{{ t('about.dataTitle') }}</h2>
      <p>{{ t('about.data') }}</p>
      <p v-if="meta.data.value">
        {{
          t('about.coverage', {
            count: n(meta.data.value.resolution_count),
            from: meta.data.value.earliest_date ? d(new Date(meta.data.value.earliest_date), 'long') : '',
            to: meta.data.value.latest_date ? d(new Date(meta.data.value.latest_date), 'long') : '',
            countries: meta.data.value.country_count,
          })
        }}
        <template v-if="meta.data.value.last_ingestion">
          {{ t('about.updated', { date: d(new Date(meta.data.value.last_ingestion), 'long') }) }}
        </template>
      </p>
      <p>{{ t('about.recorded') }}</p>

      <h2>{{ t('about.agreementTitle') }}</h2>
      <p>{{ t('about.agreement') }}</p>
      <table class="data-table scores">
        <thead>
          <tr>
            <th scope="col">{{ t('about.pair') }}</th>
            <th scope="col" class="numeric">{{ t('about.score') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>{{ t('about.pairSame') }}</td><td class="numeric">1</td></tr>
          <tr><td>{{ t('about.pairHalf') }}</td><td class="numeric">0.5</td></tr>
          <tr><td>{{ t('about.pairOpposite') }}</td><td class="numeric">0</td></tr>
          <tr><td>{{ t('about.pairAbsent') }}</td><td class="numeric">–</td></tr>
        </tbody>
      </table>
      <p>{{ t('about.agreementAverage') }}</p>

      <h2>{{ t('about.consensusTitle') }}</h2>
      <p>{{ t('about.consensus') }}</p>

      <h2>{{ t('about.alignmentTitle') }}</h2>
      <p>{{ t('about.alignment') }}</p>

      <h2>{{ t('about.keywordsTitle') }}</h2>
      <p>{{ t('about.keywords') }}</p>

      <h2>{{ t('about.mapsTitle') }}</h2>
      <p>{{ t('map.disclaimer') }}</p>

      <h2>{{ t('about.teamTitle') }}</h2>
      <p>{{ t('about.team') }}</p>
    </section>
  </div>
</template>

<style scoped>
.prose {
  display: grid;
  gap: var(--space-3);
  max-width: 72ch;
}

.prose h2 {
  margin-top: var(--space-5);
  font-size: var(--text-xl);
}

.prose p {
  line-height: 1.6;
}

.scores {
  max-width: 28rem;
}
</style>
