<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useCountries } from '@/api/queries'
import QueryState from '@/components/QueryState.vue'

const { t } = useI18n()
const countries = useCountries()
const search = ref('')

const normalise = (text: string) => text.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase()

const REGION_ORDER = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']

const groups = computed(() => {
  const q = normalise(search.value.trim())
  const matching = (countries.data.value ?? []).filter(
    (c) => !q || normalise(`${c.name} ${c.search_terms} ${c.iso3}`).includes(q),
  )
  const byRegion = new Map<string, typeof matching>()
  for (const c of matching) {
    const key = c.former_state ? 'former' : REGION_ORDER.includes(c.region) ? c.region : 'other'
    byRegion.set(key, [...(byRegion.get(key) ?? []), c])
  }
  return [...REGION_ORDER, 'other', 'former']
    .filter((key) => byRegion.has(key))
    .map((key) => ({ key, countries: byRegion.get(key)! }))
})

function membership(c: { membership: { start: string; end: string | null }[] }): string {
  const first = c.membership[0]
  const last = c.membership[c.membership.length - 1]
  if (!first || !last) return ''
  return last.end
    ? `${first.start.slice(0, 4)}–${last.end.slice(0, 4)}`
    : t('countries.since', { year: first.start.slice(0, 4) })
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('countries.title') }}</h1>
      <p>{{ t('countries.lead') }}</p>
    </div>
    <label class="field search">
      <span>{{ t('countries.search') }}</span>
      <input v-model="search" class="input" type="search" :placeholder="t('countries.searchHint')" />
    </label>
    <QueryState
      :loading="countries.isLoading.value"
      :error="countries.error.value"
      :empty="!groups.length"
      :empty-text="t('countries.empty')"
      @retry="countries.refetch()"
    >
      <div class="regions">
        <section v-for="group in groups" :key="group.key" class="region">
          <h2>
            {{ t(`regions.${group.key}`) }}
            <span class="count">{{ group.countries.length }}</span>
          </h2>
          <ul>
            <li v-for="c in group.countries" :key="c.iso3">
              <RouterLink :to="{ name: 'country', params: { iso3: c.iso3 } }">{{ c.name }}</RouterLink>
              <small>{{ membership(c) }}</small>
              <small v-if="c.historical_names.length">
                {{ t('countries.formerly', { names: c.historical_names.join(', ') }) }}
              </small>
            </li>
          </ul>
        </section>
      </div>
    </QueryState>
  </div>
</template>

<style scoped>
.search {
  max-width: 360px;
}

.regions {
  columns: 3 280px;
  column-gap: var(--space-6);
}

.region {
  break-inside: avoid;
  margin-bottom: var(--space-6);
}

.region h2 {
  font-size: var(--text-lg);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--rule-strong);
  margin-bottom: var(--space-2);
  display: flex;
  justify-content: space-between;
  align-items: baseline;
}

.count {
  font-size: var(--text-sm);
  color: var(--muted);
  font-weight: 400;
}

ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

li {
  display: grid;
  padding: 3px 0;
  font-size: var(--text-sm);
  break-inside: avoid;
}

li a {
  text-decoration: none;
}

li a:hover {
  text-decoration: underline;
}

small {
  color: var(--muted);
  font-size: var(--text-xs);
}
</style>
