/** Files the browser builds and hands over: chart images, table CSVs and the report ZIP. */

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})

export function crc32(data: Uint8Array): number {
  let c = 0xffffffff
  for (const byte of data) c = CRC_TABLE[(c ^ byte) & 0xff]! ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

export interface ZipEntry {
  name: string
  data: Uint8Array | string
}

/**
 * A ZIP without compression ("stored"): a report is a few SVGs, PNGs and CSVs, small enough that
 * a compression library isn't worth its weight. File names are UTF-8 (flag bit 11).
 */
export function zip(entries: ZipEntry[]): Blob {
  const encoder = new TextEncoder()
  const parts: Uint8Array[] = []
  const directory: Uint8Array[] = []
  let offset = 0
  for (const entry of entries) {
    const name = encoder.encode(entry.name)
    const data = typeof entry.data === 'string' ? encoder.encode(entry.data) : entry.data
    const crc = crc32(data)

    const local = new DataView(new ArrayBuffer(30))
    local.setUint32(0, 0x04034b50, true)
    local.setUint16(4, 20, true) // version needed
    local.setUint16(6, 0x0800, true) // UTF-8 names
    local.setUint16(12, 0x21, true) // 1980-01-01: a valid DOS date, as some unzip tools insist
    local.setUint32(14, crc, true)
    local.setUint32(18, data.length, true)
    local.setUint32(22, data.length, true)
    local.setUint16(26, name.length, true)
    parts.push(new Uint8Array(local.buffer), name, data)

    const central = new DataView(new ArrayBuffer(46))
    central.setUint32(0, 0x02014b50, true)
    central.setUint16(4, 20, true)
    central.setUint16(6, 20, true)
    central.setUint16(8, 0x0800, true)
    central.setUint16(14, 0x21, true)
    central.setUint32(16, crc, true)
    central.setUint32(20, data.length, true)
    central.setUint32(24, data.length, true)
    central.setUint16(28, name.length, true)
    central.setUint32(42, offset, true)
    directory.push(new Uint8Array(central.buffer), name)

    offset += 30 + name.length + data.length
  }
  const size = directory.reduce((sum, part) => sum + part.length, 0)
  const end = new DataView(new ArrayBuffer(22))
  end.setUint32(0, 0x06054b50, true)
  end.setUint16(8, entries.length, true)
  end.setUint16(10, entries.length, true)
  end.setUint32(12, size, true)
  end.setUint32(16, offset, true)
  return new Blob([...parts, ...directory, new Uint8Array(end.buffer)] as BlobPart[], {
    type: 'application/zip',
  })
}

/** RFC 4180 CSV, every field quoted. */
export function toCsv(rows: string[][]): string {
  return rows.map((row) => row.map((cell) => `"${cell.replaceAll('"', '""')}"`).join(',')).join('\r\n') + '\r\n'
}

/**
 * A rendered table as CSV, header included. A cell's `data-value`, when set, replaces its text,
 * so numbers go out raw ("0.8512", "1017") rather than as shown ("0.85", "1,017", "0,85").
 */
export function tableCsv(table: HTMLTableElement): string {
  return toCsv(
    [...table.rows].map((row) =>
      [...row.cells].map((cell) => cell.dataset.value ?? (cell.textContent ?? '').trim()),
    ),
  )
}

/** An SVG drawn onto a canvas at `scale` times its size, as PNG. */
export function svgToPng(svg: string, width: number, height: number, scale = 2): Promise<Blob> {
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }))
  return new Promise<Blob>((resolve, reject) => {
    const image = new Image()
    image.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = width * scale
      canvas.height = height * scale
      canvas.getContext('2d')!.drawImage(image, 0, 0, canvas.width, canvas.height)
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG export failed'))), 'image/png')
    }
    image.onerror = () => reject(new Error('SVG could not be drawn'))
    image.src = url
  }).finally(() => URL.revokeObjectURL(url))
}

/** Hand a file to the browser's download. */
export function download(data: Blob | string, filename: string, type = 'text/csv;charset=utf-8') {
  const blob = typeof data === 'string' ? new Blob([data], { type }) : data
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

/** A file-name-safe slug: "Agreement with France, 1990–2020" → "agreement-with-france-1990-2020". */
export function slug(text: string): string {
  return (
    text
      .normalize('NFKD')
      .replace(/[̀-ͯ]/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '') || 'chart'
  )
}

/** A legend entry: a coloured line (a series) or box (a bar, a highlighted area). */
export interface LegendItem {
  label: string
  color: string
  shape?: 'line' | 'box'
}

/** A continuous scale: gradient stops (evenly spaced) with labelled ends, a marked value, no-data. */
export interface LegendScale {
  colors: string[]
  low: string
  high: string
  /** A marked point on the scale, `at` in 0..1. */
  mark?: { at: number; label: string }
  noData?: { color: string; label: string }
}

export interface ChartLegend {
  items?: LegendItem[]
  scale?: LegendScale
}

export interface ComposeOptions {
  title: string
  legend?: ChartLegend
  footer?: string
  colors: { background: string; ink: string; muted: string; rule: string }
  font: string
}

const escapeXml = (text: string) =>
  text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)

let measureContext: CanvasRenderingContext2D | null | undefined
function textWidth(text: string, size: number, font: string): number {
  // jsdom has no canvas: fall back to an average glyph width
  measureContext ??= typeof document === 'undefined' ? null : document.createElement('canvas').getContext('2d')
  if (!measureContext) return text.length * size * 0.55
  measureContext.font = `${size}px ${font}`
  return measureContext.measureText(text).width
}

/**
 * The chart's SVG with its title above, its legend below the title and a source line at the
 * bottom, as one self-contained SVG: the legends on the page are HTML, which ECharts can't export.
 */
export function composeChart(chart: { svg: string; width: number; height: number }, options: ComposeOptions) {
  const { colors, font, legend } = options
  const pad = 16
  const width = Math.max(chart.width, 360)
  const parts: string[] = []
  const text = (x: number, y: number, content: string, size: number, fill: string, extra = '') =>
    parts.push(
      `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}"${extra}>${escapeXml(content)}</text>`,
    )

  let y = pad + 18
  text(pad, y, options.title, 18, colors.ink, ' font-weight="600"')
  y += 14

  // Items flow left to right and wrap, like the chips and swatches on the page
  if (legend?.items?.length) {
    let x = pad
    y += 14
    for (const item of legend.items) {
      const w = 22 + textWidth(item.label, 13, font) + 18
      if (x > pad && x + w > width - pad) {
        x = pad
        y += 22
      }
      parts.push(
        item.shape === 'line'
          ? `<line x1="${x}" y1="${y - 4}" x2="${x + 16}" y2="${y - 4}" stroke="${item.color}" stroke-width="3" stroke-linecap="round"/>`
          : `<rect x="${x}" y="${y - 10}" width="12" height="12" rx="2" fill="${item.color}" stroke="${colors.rule}" stroke-width="0.5"/>`,
      )
      text(x + 22, y, item.label, 13, colors.ink)
      x += w
    }
    y += 8
  }

  const scale = legend?.scale
  if (scale) {
    const barWidth = Math.min(260, width - 2 * pad)
    const stops = scale.colors
      .map((c, i) => `<stop offset="${(i / Math.max(1, scale.colors.length - 1)) * 100}%" stop-color="${c}"/>`)
      .join('')
    y += 14
    parts.push(
      `<defs><linearGradient id="legend-scale">${stops}</linearGradient></defs>`,
      `<rect x="${pad}" y="${y}" width="${barWidth}" height="10" fill="url(#legend-scale)"/>`,
    )
    if (scale.noData) {
      const x = pad + barWidth + 24
      parts.push(
        `<rect x="${x}" y="${y - 1}" width="12" height="12" rx="2" fill="${scale.noData.color}" stroke="${colors.rule}"/>`,
      )
      text(x + 18, y + 10, scale.noData.label, 13, colors.ink)
    }
    y += 26
    text(pad, y, scale.low, 12, colors.muted)
    text(pad + barWidth, y, scale.high, 12, colors.muted, ' text-anchor="end"')
    if (scale.mark) {
      const x = pad + Math.min(1, Math.max(0, scale.mark.at)) * barWidth
      parts.push(`<line x1="${x}" y1="${y - 30}" x2="${x}" y2="${y - 12}" stroke="${colors.ink}" stroke-width="2"/>`)
      y += 18
      parts.push(`<line x1="${pad + 1}" y1="${y - 10}" x2="${pad + 1}" y2="${y + 2}" stroke="${colors.ink}" stroke-width="2"/>`)
      text(pad + 8, y, scale.mark.label, 12, colors.muted)
    }
    y += 4
  }

  // The source line wraps at its " · " separators when it is wider than the chart.
  // shortcut: a single part wider than the chart (a very long URL) still overflows; wrap within parts if that bites
  const footer: string[] = []
  for (const part of options.footer?.split(' · ') ?? []) {
    const last = footer.length - 1
    if (last >= 0 && textWidth(`${footer[last]} · ${part}`, 11, font) <= width - 2 * pad) footer[last] += ` · ${part}`
    else footer.push(part)
  }

  const top = y + 8
  const height = top + chart.height + (footer.length ? 12 + footer.length * 16 : 8)
  // The chart keeps its own coordinates, nested as an <svg> placed below the legend
  const nested = chart.svg.replace(/<svg\b/, `<svg x="${(width - chart.width) / 2}" y="${top}"`)
  footer.forEach((line, i) => text(pad, height - 10 - (footer.length - 1 - i) * 16, line, 11, colors.muted))

  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" font-family="${escapeXml(font)}">` +
    `<rect width="100%" height="100%" fill="${colors.background}"/>` +
    parts.join('') +
    nested +
    '</svg>'
  return { svg, width, height }
}
