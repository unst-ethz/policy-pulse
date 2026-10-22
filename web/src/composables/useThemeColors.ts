/**
 * The design tokens as values, for chart code that can't use CSS variables directly (ECharts
 * draws on its own surface). Re-read whenever the colour scheme changes, so charts follow dark
 * mode without a reload.
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
  onMounted(() => media.addEventListener('change', refresh))
  onBeforeUnmount(() => media.removeEventListener('change', refresh))
  return colors
}
