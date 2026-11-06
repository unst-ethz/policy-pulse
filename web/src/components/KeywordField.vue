<script setup lang="ts">
/** The title/keyword search, written to the URL once typing pauses (or on Enter). */
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useFilters } from '@/composables/useFilters'

const { t } = useI18n()
const { filters, update } = useFilters()

const keyword = ref(filters.value.keyword)
watch(
  () => filters.value.keyword,
  (value) => (keyword.value = value),
)
let timer: ReturnType<typeof setTimeout> | undefined
function onKeyword() {
  clearTimeout(timer)
  timer = setTimeout(() => update({ keyword: keyword.value }), 400)
}
</script>

<template>
  <label class="field keyword">
    <span>{{ t('filters.keyword') }}</span>
    <input
      v-model="keyword"
      class="input"
      type="search"
      :placeholder="t('filters.keywordHint')"
      @input="onKeyword"
      @keydown.enter="update({ keyword })"
    />
  </label>
</template>

<style scoped>
.input {
  width: 100%;
}
</style>
