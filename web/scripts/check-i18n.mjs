// Every `t('key')` used in src/ must exist in en; fr must have exactly en's keys.
// Run: node scripts/check-i18n.mjs  (uses the TypeScript sources through Vite's loader-free path:
// the message files are plain object literals, so they are evaluated after stripping types.)
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

function load(file) {
  const src = readFileSync(file, 'utf8')
    .replace(/^import type .*$/m, '')
    .replace(/const (\w+): typeof \w+ =/, 'const $1 =')
    .replace(/export default (\w+)\s*$/m, 'return $1')
    .replace(/^export default /m, 'return ')
  return new Function(src)()
}

function keys(obj, prefix = '') {
  return Object.entries(obj).flatMap(([k, v]) =>
    typeof v === 'object' ? keys(v, `${prefix}${k}.`) : [`${prefix}${k}`],
  )
}

function files(dir) {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f)
    return statSync(p).isDirectory() ? files(p) : /\.(vue|ts)$/.test(p) ? [p] : []
  })
}

const en = new Set(keys(load('src/i18n/en.ts')))
const fr = new Set(keys(load('src/i18n/fr.ts')))
let problems = 0
for (const k of en) if (!fr.has(k)) (problems++, console.log('fr missing', k))
for (const k of fr) if (!en.has(k)) (problems++, console.log('fr extra', k))

const used = new Set()
const dynamic = []
for (const file of files('src')) {
  const text = readFileSync(file, 'utf8')
  for (const m of text.matchAll(/\bt\(\s*'([\w.]+)'/g)) used.add(m[1])
  for (const m of text.matchAll(/\bt\(\s*`([^`]+)`/g)) dynamic.push(`${file}: ${m[1]}`)
}
for (const k of used) if (!en.has(k)) (problems++, console.log('en missing', k))
console.log(`${used.size} static keys used, ${en.size} defined; dynamic keys to eyeball:`)
for (const d of dynamic) console.log('  ', d)
process.exit(problems ? 1 : 0)
