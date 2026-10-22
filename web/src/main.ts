import '@fontsource/ibm-plex-sans/400.css'
import '@fontsource/ibm-plex-sans/500.css'
import '@fontsource/ibm-plex-sans/600.css'
import '@fontsource/ibm-plex-sans-condensed/400.css'
import '@fontsource/ibm-plex-sans-condensed/500.css'
import './styles/tokens.css'
import './styles/base.css'

import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query'
import { createApp } from 'vue'
import App from './App.vue'
import { i18n } from './i18n'
import { router } from './router'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // The data changes at most twice a day (ingestion runs), and the API answers repeat
      // requests with 304s, so revalidating on every focus would only add noise.
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
      retry: (count, error) => count < 2 && !('status' in error && Number(error.status) < 500),
    },
  },
})

createApp(App).use(router).use(i18n).use(VueQueryPlugin, { queryClient }).mount('#app')
