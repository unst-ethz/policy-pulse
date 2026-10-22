/**
 * Colour scales shared by the charts. Interpolation runs in OKLab so the diverging ramp is
 * perceptually even and its midpoint really reads as neutral.
 */

type Rgb = [number, number, number]
type Lab = [number, number, number]

function hexToRgb(hex: string): Rgb {
  const h = hex.replace('#', '')
  const full = h.length === 3 ? [...h].map((c) => c + c).join('') : h
  return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255) as Rgb
}

const toLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4)
const toGamma = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055)

function rgbToOklab([r, g, b]: Rgb): Lab {
  const [lr, lg, lb] = [toLinear(r), toLinear(g), toLinear(b)]
  const l = Math.cbrt(0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb)
  const m = Math.cbrt(0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb)
  const s = Math.cbrt(0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb)
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ]
}

function oklabToHex([L, a, b]: Lab): string {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
  const s = (L - 0.0894841775 * a - 1.291485548 * b) ** 3
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s,
  ]
  return (
    '#' +
    rgb
      .map((c) => Math.round(Math.min(1, Math.max(0, toGamma(c))) * 255))
      .map((c) => c.toString(16).padStart(2, '0'))
      .join('')
  )
}

function mix(a: string, b: string, t: number): string {
  const [la, lb] = [rgbToOklab(hexToRgb(a)), rgbToOklab(hexToRgb(b))]
  return oklabToHex([0, 1, 2].map((i) => la[i]! + (lb[i]! - la[i]!) * t) as Lab)
}

/**
 * A diverging scale from `low` through `mid` to `high`. `center` is the value that maps to the
 * neutral midpoint (0.5 by default; the mean consensus when centring on "the average country").
 */
export function divergingScale(
  low: string,
  mid: string,
  high: string,
  domain: [number, number] = [0, 1],
  center = (domain[0] + domain[1]) / 2,
) {
  const [d0, d1] = domain
  return (value: number): string => {
    const v = Math.min(d1, Math.max(d0, value))
    if (v <= center) return mix(low, mid, center === d0 ? 1 : (v - d0) / (center - d0))
    return mix(mid, high, d1 === center ? 0 : (v - center) / (d1 - center))
  }
}
