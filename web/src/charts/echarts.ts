/** ECharts, registered piecewise so only what the app draws ends up in the bundle. */
import { BarChart, LineChart, MapChart, ScatterChart } from 'echarts/charts'
import {
  GeoComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from 'echarts/components'
import * as echarts from 'echarts/core'
import { SVGRenderer } from 'echarts/renderers'
import { geoEqualEarth } from 'd3-geo'
import type { FeatureCollection } from 'geojson'

echarts.use([
  BarChart,
  LineChart,
  MapChart,
  ScatterChart,
  GeoComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
  SVGRenderer,
])

export { echarts }

/** Equal Earth: an equal-area projection, so no country looks bigger than it is. */
export const equalEarth = geoEqualEarth()
export const mapProjection = {
  project: (point: number[]) => equalEarth(point as [number, number]) as number[],
  unproject: (point: number[]) => equalEarth.invert!(point as [number, number]) as number[],
}

let worldLoading: Promise<void> | null = null

/** Register the world map once (public/geo/world.json, built by web/scripts/build_world_geo.py). */
export function loadWorld(): Promise<void> {
  worldLoading ??= fetch('/geo/world.json')
    .then((response) => response.json() as Promise<FeatureCollection>)
    .then((world) => {
      // Antarctica has no vote and takes a fifth of the canvas.
      world.features = world.features.filter((f) => f.properties?.iso3 !== 'ATA')
      echarts.registerMap('world', world as never)
    })
  return worldLoading
}
