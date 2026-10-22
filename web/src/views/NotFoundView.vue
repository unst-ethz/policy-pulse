<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>{{ t('notFound.title') }}</h1>
      <p>{{ t('notFound.text') }}</p>
    </div>
    <p><RouterLink class="button" to="/">{{ t('notFound.home') }}</RouterLink></p>
  </div>
</template>
