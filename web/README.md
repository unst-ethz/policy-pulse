# Policy Pulse web (Vue 3 prototype)

The Vue frontend that will replace the Dash app. It talks only to the HTTP API in `app/api/`.

```bash
# terminal 1, repo root: the API on :8000 (needs the local database, see the root README)
uv run start-api

# terminal 2, web/
npm install
npm run dev          # http://localhost:5173, /api is proxied to :8000
```

| Command | What it does |
|---|---|
| `npm run dev` | Dev server with hot reload |
| `npm run build` | Type-check (`vue-tsc`) and build to `dist/` |
| `npm test` | Unit tests (Vitest) |
| `npm run check:i18n` | Every UI string used exists in English, and French has the same keys |
| `npm run gen:api` | Regenerate `src/api/schema.d.ts` from the running API's OpenAPI document |

Run `gen:api` whenever the API changes; the type check then shows every place the frontend has to
follow.

## Layout

```
src/api/            typed client (openapi-fetch on the generated schema) + one TanStack Query
                    composable per endpoint
src/composables/    the URL-backed filter model, theme colours for charts
src/components/     VoteBoard, VoteBar, WorldMap, TimelineChart, AlignmentScatter, WordCloud,
                    pickers, FilterBar, ChartFrame (chart + table twin), QueryState
src/views/          one per route: home, resolutions (+ detail), countries (+ country), topics,
                    alignment, about
src/i18n/           UI strings (en, fr); the other UN languages fall back to English for UI text,
                    while country and subject names come from the API in that language
src/styles/         design tokens (light and dark) and base styles
public/geo/         country shapes, built by scripts/build_world_geo.py from Natural Earth
```

## Decisions

- **State lives in the URL.** Filters, sort, page, comparison countries and chart options are
  query parameters, so every view can be bookmarked and shared, and the back button works.
- **Data, not figures.** The API returns numbers; charts are drawn here (ECharts for the map, the
  timeline and the scatter; HTML for bars, the vote board and the legends; d3-cloud for the word
  cloud layout).
- **Colour follows the data's own scale.** Votes are Yes = +1, Abstain = 0, No = −1, so they use a
  diverging blue ↔ grey ↔ red, and so does every agreement scale: blue is "with", red "against",
  everywhere. The palette was checked for colour-vision deficiency and contrast in light and dark
  mode. Every chart has a table view, and the vote board groups by vote, so colour is never the
  only cue.
- **Headless components (Reka UI)** for comboboxes and popovers: accessible behaviour without a
  visual style to fight. If the UN partners mandate a component library, it slots in here.
- **Prototype scope.** No end-to-end tests yet, UI strings only in English and French, and no
  production deployment (the build is static files; nginx would serve `dist/` and proxy `/api`).
- **AWS deploy.** On `feature/api-mongodb`, the `web-to-s3` job in `.github/workflows/deploy.yml` builds `dist/` for `/`, syncs it to S3 and invalidates `index.html` on CloudFront; it runs only once the repo variables `WEB_BUCKET` and `WEB_DISTRIBUTION_ID` are set.
