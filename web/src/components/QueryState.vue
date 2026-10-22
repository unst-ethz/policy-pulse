<script setup lang="ts">
/**
 * Loading, error and empty states around a query's content. While a new result loads the
 * previous one stays visible, dimmed, so the layout doesn't jump.
 */
import { useI18n } from 'vue-i18n'

defineProps<{
  loading: boolean
  fetching?: boolean
  error?: Error | null
  empty?: boolean
  emptyText?: string
  minHeight?: string
}>()
const emit = defineEmits<{ retry: [] }>()
const { t } = useI18n()
</script>

<template>
  <div class="state" :style="{ minHeight }" :aria-busy="loading || fetching">
    <p v-if="error" class="message error" role="alert">
      {{ t('state.error', { detail: error.message }) }}
      <button class="button" type="button" @click="emit('retry')">{{ t('state.retry') }}</button>
    </p>
    <p v-else-if="loading" class="message">{{ t('state.loading') }}</p>
    <p v-else-if="empty" class="message">{{ emptyText ?? t('state.empty') }}</p>
    <div v-else class="content" :class="{ stale: fetching }">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.state {
  display: grid;
  min-width: 0;
}

.message {
  align-self: center;
  justify-self: start;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-3);
  color: var(--muted);
  font-size: var(--text-sm);
  padding: var(--space-4) 0;
}

.message.error {
  color: var(--danger-ink);
}

.content {
  min-width: 0;
  transition: opacity 150ms ease;
}

.content.stale {
  opacity: 0.55;
}
</style>
