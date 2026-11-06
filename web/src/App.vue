<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useMeta } from './api/queries'
import { LANGUAGES, type LanguageCode, setLanguage } from './i18n'

const { t, locale, d } = useI18n()
const meta = useMeta()

const nav = computed(() => [
  { to: '/resolutions', label: t('nav.resolutions') },
  { to: '/countries', label: t('nav.countries') },
  { to: '/subjects', label: t('nav.subjects') },
  { to: '/alignment', label: t('nav.alignment') },
  { to: '/about', label: t('nav.about') },
])
</script>

<template>
  <a class="skip" href="#main">{{ t('nav.skip') }}</a>
  <header class="top">
    <div class="bar">
      <RouterLink to="/" class="brand">
        <span class="mark" aria-hidden="true"><i /><i /><i /><i /></span>
        <span class="name">Policy Pulse</span>
        <span class="tagline">{{ t('brand.tagline') }}</span>
      </RouterLink>
      <nav :aria-label="t('nav.label')">
        <RouterLink v-for="item in nav" :key="item.to" :to="item.to" class="nav-link">
          {{ item.label }}
        </RouterLink>
      </nav>
      <label class="lang">
        <span class="visually-hidden">{{ t('nav.language') }}</span>
        <select
          class="input"
          :value="locale"
          @change="setLanguage(($event.target as HTMLSelectElement).value as LanguageCode)"
        >
          <option v-for="lang in LANGUAGES" :key="lang.code" :value="lang.code" :lang="lang.code">
            {{ lang.label }}
          </option>
        </select>
      </label>
    </div>
  </header>

  <main id="main">
    <RouterView />
  </main>

  <footer class="foot">
    <div class="inner">
      <p v-if="meta.data.value">
        {{
          t('footer.coverage', {
            count: meta.data.value.resolution_count.toLocaleString(locale),
            latest: meta.data.value.latest_date ? d(new Date(meta.data.value.latest_date), 'long') : '',
          })
        }}
      </p>
      <p>
        {{ t('footer.source') }}
        <RouterLink to="/about">{{ t('footer.method') }}</RouterLink>
      </p>
      <p>{{ t('footer.team') }}</p>
    </div>
  </footer>
</template>

<style scoped>
.skip {
  position: absolute;
  inset-inline-start: var(--space-4);
  top: -48px;
  padding: var(--space-2) var(--space-3);
  background: var(--ink);
  color: var(--surface);
  z-index: 100;
}

.skip:focus {
  top: var(--space-2);
}

.top {
  background: var(--surface);
  border-bottom: 1px solid var(--rule);
  position: sticky;
  top: 0;
  z-index: 20;
}

.bar {
  max-width: var(--content-width);
  margin-inline: auto;
  padding: 0 var(--gutter);
  min-height: 60px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2) var(--space-6);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: var(--space-3);
  text-decoration: none;
}

.mark {
  display: grid;
  grid-template-columns: repeat(2, 7px);
  gap: 2px;
}

.mark i {
  width: 7px;
  height: 7px;
  border-radius: 1px;
  background: var(--yes-cell);
}

.mark i:nth-child(3) {
  background: var(--no-cell);
}

.mark i:nth-child(4) {
  background: var(--abstain-cell);
}

.name {
  font-weight: 600;
  font-size: var(--text-lg);
  letter-spacing: -0.01em;
}

.tagline {
  color: var(--muted);
  font-size: var(--text-sm);
  border-inline-start: 1px solid var(--rule-strong);
  padding-inline-start: var(--space-3);
}

nav {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
  margin-inline-start: auto;
}

.nav-link {
  white-space: nowrap;
  padding: 6px 10px;
  border-radius: var(--radius-s);
  text-decoration: none;
  color: var(--ink-2);
  font-size: var(--text-sm);
  font-weight: 500;
}

.nav-link:hover {
  color: var(--ink);
  background: var(--surface-sunk);
}

.nav-link.router-link-active {
  color: var(--ink);
  box-shadow: inset 0 -2px 0 var(--accent);
  border-radius: 0;
}

.lang .input {
  min-height: 32px;
}

.foot {
  border-top: 1px solid var(--rule);
  background: var(--surface);
  color: var(--muted);
  font-size: var(--text-xs);
}

.foot .inner {
  max-width: var(--content-width);
  margin-inline: auto;
  padding: var(--space-5) var(--gutter);
  display: grid;
  gap: var(--space-1);
}

@media (max-width: 900px) {
  .tagline {
    display: none;
  }

  nav {
    order: 3;
    width: 100%;
    margin-inline-start: 0;
    overflow-x: auto;
    flex-wrap: nowrap;
    padding-bottom: var(--space-2);
  }

  .lang {
    margin-inline-start: auto;
  }
}
</style>
