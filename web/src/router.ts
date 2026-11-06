import { createRouter, createWebHistory } from 'vue-router'
import HomeView from './views/HomeView.vue'

export const router = createRouter({
  // BASE_URL is Vite's `base`: '/' by default, '/experimental/' for the VM build.
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', name: 'home', component: HomeView },
    {
      path: '/resolutions',
      name: 'resolutions',
      component: () => import('./views/ResolutionsView.vue'),
    },
    {
      path: '/resolutions/:id',
      name: 'resolution',
      component: () => import('./views/ResolutionView.vue'),
      props: true,
    },
    { path: '/countries', name: 'countries', component: () => import('./views/CountriesView.vue') },
    {
      path: '/countries/:iso3',
      name: 'country',
      component: () => import('./views/CountryView.vue'),
      props: (route) => ({ iso3: String(route.params.iso3).toUpperCase() }),
    },
    { path: '/subjects', name: 'subjects', component: () => import('./views/SubjectsView.vue') },
    {
      path: '/subjects/:id',
      name: 'subject',
      component: () => import('./views/SubjectView.vue'),
      props: true,
    },
    // The old word-cloud page: the cloud now sits above the resolution list
    { path: '/topics', redirect: (to) => ({ name: 'resolutions', query: to.query }) },
    { path: '/alignment', name: 'alignment', component: () => import('./views/AlignmentView.vue') },
    { path: '/about', name: 'about', component: () => import('./views/AboutView.vue') },
    { path: '/:pathMatch(.*)*', name: 'not-found', component: () => import('./views/NotFoundView.vue') },
  ],
  scrollBehavior(to, from, saved) {
    if (saved) return saved
    // Filter changes rewrite the query string only; keep the reader where they are.
    if (to.path === from.path) return false
    return { top: 0 }
  },
})
