/**
 * The charts a page can put into a report. Each ChartFrame on the page registers itself while it
 * is mounted, so the report holds exactly what is on screen: hiding a section leaves it out.
 */
import { type InjectionKey, inject, onBeforeUnmount, onMounted, provide } from 'vue'

export interface ChartImage {
  svg: string
  width: number
  height: number
}

export interface ReportChart {
  title: () => string
  description: () => string | undefined
  /**
   * The chart as SVG with title, legend and source line (`bare`: the legend only, for a PDF that
   * sets its own), or null when it has none (an HTML chart) or nothing drawn yet.
   */
  image: (options?: { bare?: boolean }) => Promise<ChartImage | null>
  /** The table twin as CSV, or null when there is none. */
  csv: () => string | null
}

export interface Report {
  /** File-name prefix for single downloads, e.g. "CHE_1990-2020". */
  prefix: () => string
  charts: Map<string, ReportChart>
}

const REPORT: InjectionKey<Report> = Symbol('report')

export function provideReport(prefix: () => string): Report {
  const report: Report = { prefix, charts: new Map() }
  provide(REPORT, report)
  return report
}

/** Register a chart under `id` with the page's report, if the page has one. */
export function useReportChart(id: string | undefined, chart: ReportChart): Report | null {
  const report = inject(REPORT, null)
  if (report && id) {
    onMounted(() => report.charts.set(id, chart))
    // Only its own entry: a frame with the same id may already have replaced it
    onBeforeUnmount(() => report.charts.get(id) === chart && report.charts.delete(id))
  }
  return report
}
