/** Data shapes the chart components take. */
import type { RouteLocationRaw } from 'vue-router'

export interface TimelineSeries {
  country: string
  label: string
  color: string
  /** `x` is the fractional year of the session's median date; `special` marks (emergency) special sessions. */
  points: { x: number; year: number; session: string; special: boolean; agreement: number | null; votes: number }[]
}

/** The five UN geoscheme regions, in legend order; each has a `--region-*` colour token. */
export const REGIONS = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'] as const

export interface ScatterPoint {
  country: string
  name: string
  region: string
  x: number
  y: number
  votes: number
}

export interface CloudWord {
  term: string
  count: number
  to: RouteLocationRaw
  title: string
}
