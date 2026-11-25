// Quick-select groups of States, labelled under `groups.<key>` in the i18n files.
// shortcut: the regional groups are five representative States each, placeholders until real
// groupings (full regional groups, G77, ...) are modelled.
export const GROUPS = {
  p5: ['USA', 'GBR', 'FRA', 'RUS', 'CHN'],
  africa: ['NGA', 'ZAF', 'EGY', 'KEN', 'ETH'],
  asiaPacific: ['IND', 'JPN', 'IDN', 'BGD', 'PAK'],
  grulac: ['BRA', 'MEX', 'ARG', 'COL', 'CHL'],
  sids: ['MDV', 'FJI', 'JAM', 'TTO', 'VUT'],
} as const satisfies Record<string, readonly string[]>
