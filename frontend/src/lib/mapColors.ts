// The original Plotly RdYlBu palette, from opposite votes to identical votes.
export const agreementPalette = [
  '#a50026',
  '#d73027',
  '#f46d43',
  '#fdae61',
  '#fee090',
  '#ffffbf',
  '#e0f3f8',
  '#abd9e9',
  '#74add1',
  '#4575b4',
  '#313695',
];

export function agreementColor(value: number | null | undefined, midpoint = 0.5) {
  if (value == null) return '#e1e1e1';
  const center = midpoint > 0 && midpoint < 1 ? midpoint : 0.5;
  const t = value <= center ? value / center / 2 : 0.5 + (value - center) / (1 - center) / 2;
  const position = Math.max(0, Math.min(1, t)) * (agreementPalette.length - 1);
  const index = Math.min(Math.floor(position), agreementPalette.length - 2);
  const fraction = position - index;
  const a = parseInt(agreementPalette[index].slice(1), 16);
  const b = parseInt(agreementPalette[index + 1].slice(1), 16);
  return `rgb(${[16, 8, 0].map((shift) => Math.round(((a >> shift) & 255) * (1 - fraction) + ((b >> shift) & 255) * fraction)).join(', ')})`;
}

export function agreementGradient(midpoint = 0.5) {
  const center = midpoint > 0 && midpoint < 1 ? midpoint : 0.5;
  return `linear-gradient(90deg, ${agreementPalette
    .map((color, i) => {
      const t = i / (agreementPalette.length - 1);
      const position = t <= 0.5 ? t * 2 * center : center + (t - 0.5) * 2 * (1 - center);
      return `${color} ${position * 100}%`;
    })
    .join(', ')})`;
}
