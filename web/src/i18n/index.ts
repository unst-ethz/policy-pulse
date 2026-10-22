/**
 * UI strings. English and French are complete; the other UN languages fall back to English for
 * interface text while country and subject names still come from the API in that language.
 */
import { createI18n } from 'vue-i18n'
import en from './en'
import fr from './fr'

export const LANGUAGES = [
  { code: 'en', label: 'English', dir: 'ltr' },
  { code: 'fr', label: 'Français', dir: 'ltr' },
  { code: 'es', label: 'Español', dir: 'ltr' },
  { code: 'ar', label: 'العربية', dir: 'rtl' },
  { code: 'zh', label: '中文', dir: 'ltr' },
  { code: 'ru', label: 'Русский', dir: 'ltr' },
] as const

export type LanguageCode = (typeof LANGUAGES)[number]['code']

const STORAGE_KEY = 'policy-pulse.lang'

function initialLanguage(): LanguageCode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (LANGUAGES.some((l) => l.code === saved)) return saved as LanguageCode
  } catch {
    // Storage can be unavailable (private mode); the browser language is a fine fallback.
  }
  const browser = navigator.language.slice(0, 2)
  return (LANGUAGES.find((l) => l.code === browser)?.code ?? 'en') as LanguageCode
}

const numberFormats = {
  score: { minimumFractionDigits: 2, maximumFractionDigits: 2 },
  percent: { style: 'percent', maximumFractionDigits: 1 },
  percentShort: { style: 'percent', maximumFractionDigits: 0 },
} as const

const dateFormats = {
  short: { year: 'numeric', month: 'short', day: 'numeric' },
  long: { year: 'numeric', month: 'long', day: 'numeric' },
} as const

const codes = LANGUAGES.map((l) => l.code)

export const i18n = createI18n({
  legacy: false,
  locale: initialLanguage(),
  fallbackLocale: 'en',
  messages: { en, fr },
  missingWarn: false,
  fallbackWarn: false,
  numberFormats: Object.fromEntries(codes.map((c) => [c, numberFormats])),
  datetimeFormats: Object.fromEntries(codes.map((c) => [c, dateFormats])),
})

export function setLanguage(code: LanguageCode) {
  i18n.global.locale.value = code
  const lang = LANGUAGES.find((l) => l.code === code)!
  document.documentElement.lang = code
  document.documentElement.dir = lang.dir
  try {
    localStorage.setItem(STORAGE_KEY, code)
  } catch {
    // Not persisted; the choice still applies to this visit.
  }
}

setLanguage(i18n.global.locale.value as LanguageCode)
