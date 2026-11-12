/**
 * The design tokens as values, for chart code that can't use CSS variables directly (ECharts
 * draws on its own surface). Re-read whenever the colour scheme changes, so charts follow dark
 * mode without a reload, and when the page forces a theme (`data-theme`, e.g. light for printing).
 */
import { onBeforeUnmount, onMounted, reactive } from 'vue'

const NAMES = [
  'surface',
  'ink',
  'ink-2',
  'muted',
  'rule',
  'rule-strong',
  'accent',
  'yes',
  'no',
  'abstain',
  'absent',
  'scale-mid',
  'font-sans',
] as const

export type ThemeColors = Record<(typeof NAMES)[number], string>

function read(): ThemeColors {
  const style = getComputedStyle(document.documentElement)
  return Object.fromEntries(
    NAMES.map((name) => [name, style.getPropertyValue(`--${name}`).trim()]),
  ) as ThemeColors
}

export function useThemeColors() {
  const colors = reactive(read())
  const media = window.matchMedia('(prefers-color-scheme: dark)')
  const refresh = () => Object.assign(colors, read())
  const observer = new MutationObserver(refresh)
  onMounted(() => {
    media.addEventListener('change', refresh)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
  })
  onBeforeUnmount(() => {
    media.removeEventListener('change', refresh)
    observer.disconnect()
  })
  return colors
}
