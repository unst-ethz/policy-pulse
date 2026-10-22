/** Data shapes the chart components take. */
import type { RouteLocationRaw } from 'vue-router'

export interface TimelineSeries {
  country: string
  label: string
  color: string
  points: { year: number; session: string; agreement: number | null; votes: number }[]
}

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
