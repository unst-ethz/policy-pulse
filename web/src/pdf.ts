/**
 * A report as an A4 PDF: a cover block, then sections of headings, text, key figures, chart
 * images and tables, with a running header and page numbers. jsPDF loads only when a report is
 * made, so the page itself doesn't carry it.
 */
import type { jsPDF } from 'jspdf'

// The light theme's tokens (styles/tokens.css): a PDF is for paper, whatever the screen's theme
export const PDF_COLORS = {
  ink: '#172033',
  ink2: '#485165',
  muted: '#646d7e',
  rule: '#dde1e8',
  sunk: '#f7f8fa',
  accent: '#8a650c',
  yes: '#2a78d6',
  no: '#e34948',
  abstain: '#8f8d86',
  absent: '#c3c2b7',
}

// mm
const PAGE = { width: 210, height: 297, margin: 16, top: 22, bottom: 22 }
const CONTENT = PAGE.width - 2 * PAGE.margin
const PT = 0.3528 // mm per point

export interface PdfTable {
  head: string[]
  rows: string[][]
  /** Columns aligned right (numbers). */
  numeric?: number[]
  /** A column drawn as a bar instead of text: one value per row, 0..1 (null: no bar). */
  bar?: { column: number; values: (number | null)[]; color?: string }
}

/**
 * The PDF's built-in Helvetica covers Latin-1 only; characters outside it (typographic quotes,
 * dashes) would print as garbage, so they become their plain equivalents.
 */
// shortcut: names outside Latin-1 (e.g. Vietnamese, non-Latin scripts) lose their marks; embed a
// TTF font with doc.addFont if reports ever need them
export function pdfText(text: string): string {
  return text
    .replace(/[‘’ʼ]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[–—]/g, '-')
    .replace(/…/g, '...')
    .replace(/[  ]/g, ' ')
    .normalize('NFC')
    .replace(/[^\u0000-ÿ]/g, (c) => c.normalize('NFKD').replace(/[^\u0000-ÿ]/g, '') || '?')
}

export async function createPdf(running: { header: string; footer: string }) {
  const [{ jsPDF }, { autoTable }] = await Promise.all([import('jspdf'), import('jspdf-autotable')])
  const doc: jsPDF = new jsPDF({ unit: 'mm', format: 'a4' })
  let y = PAGE.top

  const font = (size: number, color = PDF_COLORS.ink, style: 'normal' | 'bold' = 'normal') => {
    doc.setFont('helvetica', style)
    doc.setFontSize(size)
    doc.setTextColor(color)
  }
  const lineHeight = (size: number) => size * PT * 1.35
  /** Start a new page unless `height` mm still fit on this one. */
  const room = (height: number) => {
    if (y + height > PAGE.height - PAGE.bottom) {
      doc.addPage()
      y = PAGE.top
    }
  }
  const lines = (text: string, size: number, width = CONTENT): string[] => {
    doc.setFontSize(size)
    return doc.splitTextToSize(pdfText(text), width) as string[]
  }
  const write = (text: string, size: number, color: string, style: 'normal' | 'bold' = 'normal', x = PAGE.margin, width = CONTENT) => {
    font(size, color, style)
    for (const line of lines(text, size, width)) {
      room(lineHeight(size))
      y += lineHeight(size)
      doc.text(line, x, y - lineHeight(size) * 0.28)
    }
  }

  return {
    cover(title: string, subtitle: string, rows: [string, string][]) {
      y = PAGE.top + 6
      write(subtitle, 10, PDF_COLORS.accent, 'bold')
      y += 2
      write(title, 26, PDF_COLORS.ink, 'bold')
      y += 4
      doc.setDrawColor(PDF_COLORS.rule)
      doc.line(PAGE.margin, y, PAGE.width - PAGE.margin, y)
      y += 3
      for (const [label, value] of rows) {
        const start = y
        write(label, 9, PDF_COLORS.muted, 'normal', PAGE.margin, 38)
        const end = y
        y = start
        write(value, 9, PDF_COLORS.ink, 'normal', PAGE.margin + 40, CONTENT - 40)
        y = Math.max(y, end) + 0.8
      }
      y += 4
    },

    heading(text: string) {
      room(60) // never a heading alone at the foot of a page: room for its text and first rows
      y += 5
      write(text, 15, PDF_COLORS.ink, 'bold')
      y += 1.5
    },

    subheading(text: string) {
      room(36) // with the text and first rows that follow
      y += 2
      write(text, 11, PDF_COLORS.ink, 'bold')
      y += 0.5
    },

    paragraph(text: string, size = 9, color = PDF_COLORS.ink2) {
      write(text, size, color)
      y += 1.5
    },

    /** Key figures side by side: a large value over a short caption. */
    figures(items: { value: string; label: string }[]) {
      room(24)
      const width = CONTENT / items.length
      const top = y
      let bottom = y
      items.forEach((item, i) => {
        const x = PAGE.margin + i * width
        y = top
        write(item.value, 20, PDF_COLORS.ink, 'bold', x, width - 4)
        write(item.label, 8.5, PDF_COLORS.muted, 'normal', x, width - 4)
        bottom = Math.max(bottom, y)
      })
      y = bottom + 3
    },

    /** One bar split into coloured parts, with a legend line under it. */
    stackedBar(parts: { value: number; color: string; label: string }[]) {
      room(16)
      const total = parts.reduce((sum, p) => sum + p.value, 0) || 1
      let x = PAGE.margin
      for (const part of parts) {
        const w = (part.value / total) * CONTENT
        doc.setFillColor(part.color)
        doc.rect(x, y, w, 4, 'F')
        x += w
      }
      y += 8
      x = PAGE.margin
      font(8.5, PDF_COLORS.ink2)
      for (const part of parts) {
        const label = pdfText(part.label)
        const w = 5 + doc.getTextWidth(label) + 6
        if (x + w > PAGE.width - PAGE.margin) {
          x = PAGE.margin
          y += 5
        }
        doc.setFillColor(part.color)
        doc.rect(x, y - 2.6, 2.8, 2.8, 'F')
        doc.text(label, x + 4.2, y)
        x += w
      }
      y += 5
    },

    /**
     * A chart (PNG) under its heading and description, all on one page: as wide as the text
     * unless that makes it taller than the page leaves.
     */
    chart(title: string, description: string | undefined, png: string, widthPx: number, heightPx: number) {
      const textHeight = 13 + (description ? lines(description, 9).length * lineHeight(9) + 1.5 : 0)
      const maxHeight = PAGE.height - PAGE.top - PAGE.bottom - textHeight - 6
      let w = CONTENT
      let h = (heightPx / widthPx) * w
      if (h > maxHeight) {
        h = maxHeight
        w = (widthPx / heightPx) * h
      }
      room(textHeight + h + 8)
      this.heading(title)
      if (description) this.paragraph(description)
      doc.addImage(png, 'PNG', PAGE.margin + (CONTENT - w) / 2, y, w, h)
      y += h + 3
    },

    table({ head, rows, numeric = [], bar }: PdfTable) {
      // The head and a few rows, or the whole of a short table, before the page ends
      room(10 + Math.min(rows.length, 5) * 6)
      autoTable(doc, {
        startY: y,
        rowPageBreak: 'avoid',
        head: [head.map(pdfText)],
        body: rows.map((row) => row.map(pdfText)),
        margin: { left: PAGE.margin, right: PAGE.margin, top: PAGE.top, bottom: PAGE.bottom },
        theme: 'plain',
        styles: { font: 'helvetica', fontSize: 8.5, cellPadding: { top: 1.4, bottom: 1.4, left: 1.5, right: 1.5 }, textColor: PDF_COLORS.ink },
        headStyles: { textColor: PDF_COLORS.muted, fontStyle: 'bold', fillColor: PDF_COLORS.sunk },
        alternateRowStyles: { fillColor: '#fbfbfc' },
        columnStyles: Object.fromEntries([
          ...numeric.map((c) => [c, { halign: 'right' as const }]),
          ...(bar ? [[bar.column, { cellWidth: 48 }]] : []),
        ]),
        // Column styles reach only the body: numeric headers line up with their numbers here
        didParseCell: (data) => {
          if (data.section === 'head' && numeric.includes(data.column.index)) data.cell.styles.halign = 'right'
        },
        didDrawCell: (data) => {
          if (!bar || data.section !== 'body' || data.column.index !== bar.column) return
          const value = bar.values[data.row.index]
          const x = data.cell.x + 1.5
          const width = data.cell.width - 3
          const top = data.cell.y + data.cell.height / 2 - 1.3
          doc.setFillColor(PDF_COLORS.rule)
          doc.rect(x, top, width, 2.6, 'F')
          if (value != null) {
            doc.setFillColor(bar.color ?? PDF_COLORS.yes)
            doc.rect(x, top, width * Math.min(1, Math.max(0, value)), 2.6, 'F')
          }
        },
      })
      y = (doc as unknown as { lastAutoTable: { finalY: number } }).lastAutoTable.finalY + 4
    },

    /** Header and page numbers on every page but the first, then hand the file over. */
    blob(): Blob {
      const pages = doc.getNumberOfPages()
      for (let page = 1; page <= pages; page++) {
        doc.setPage(page)
        doc.setDrawColor(PDF_COLORS.rule)
        if (page > 1) {
          font(8, PDF_COLORS.muted)
          doc.text(pdfText(running.header), PAGE.margin, 13)
          doc.line(PAGE.margin, 15, PAGE.width - PAGE.margin, 15)
        }
        font(7.5, PDF_COLORS.muted)
        const footY = PAGE.height - 12
        doc.line(PAGE.margin, footY - 4, PAGE.width - PAGE.margin, footY - 4)
        doc.text(lines(running.footer, 7.5, CONTENT - 20).slice(0, 2), PAGE.margin, footY)
        doc.text(`${page} / ${pages}`, PAGE.width - PAGE.margin, footY, { align: 'right' })
      }
      return doc.output('blob')
    },
  }
}
