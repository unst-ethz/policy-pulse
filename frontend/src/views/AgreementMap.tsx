import { useState } from 'react';
import { geoPath } from 'd3-geo';
import { geoRobinson } from 'd3-geo-projection';
import { feature } from 'topojson-client';
import type { GeometryCollection, Topology } from 'topojson-specification';
import world from 'world-atlas/countries-110m.json';
import { Link } from 'react-router-dom';
import { useApi, type Model } from '../api/client';
import { Empty, ErrorPanel, Loading, number, Panel, score } from '../components/ui';
import { commonParams } from '../lib/filters';
import { agreementColor, agreementGradient } from '../lib/mapColors';
import type { ViewProps } from './types';

const geography = feature<{ name?: string }>(
  world as unknown as Topology,
  world.objects.countries as GeometryCollection,
);
export default function AgreementMap({ filters, metadata }: ViewProps) {
  const query = useApi<Model<'AgreementResult'>>(
    `/analysis/agreement/${filters.country}`,
    commonParams(filters, 'map'),
    Boolean(filters.country),
  );
  const [selected, setSelected] = useState<string | null>(null);
  const [adaptive, setAdaptive] = useState(false);
  if (!filters.country)
    return (
      <Panel title="Agreement map">
        <Empty title="Select a reference country">
          Choose a reference country and apply filters to compare its recorded votes with other
          member states.
        </Empty>
      </Panel>
    );
  if (query.isPending) return <Loading />;
  if (query.error) return <ErrorPanel error={query.error} retry={() => query.refetch()} />;
  const data = query.data!;
  const projection = geoRobinson()
    .rotate([-data.reference_longitude, 0])
    .fitExtent(
      [
        [12, 12],
        [888, 445],
      ],
      { type: 'Sphere' },
    );
  const path = geoPath(projection);
  const canAdapt =
    data.consensus_midpoint != null && data.consensus_midpoint > 0 && data.consensus_midpoint < 1;
  const midpoint = adaptive && canAdapt ? data.consensus_midpoint! : 0.5;
  const byCode = new Map(data.items.map((item) => [item.country, item]));
  const byM49 = new Map(metadata.countries.filter((c) => c.m49).map((c) => [Number(c.m49), c]));
  const names = new Map(metadata.countries.map((c) => [c.code, c.name]));
  const detail = data.items.find((c) => c.country === selected);
  return (
    <Panel
      title={`Agreement with ${names.get(filters.country) ?? filters.country}`}
      description={`${number(data.resolution_count)} selected resolutions. Each comparison includes only votes cast by both countries.`}
    >
      {!data.resolution_count ? (
        <Empty />
      ) : (
        <>
          <label className="checkbox-label map-color-control">
            <input
              type="checkbox"
              checked={adaptive}
              disabled={!canAdapt}
              onChange={(event) => setAdaptive(event.target.checked)}
            />
            Centre colour scale on average consensus score
          </label>
          <div className="map-wrapper">
            <svg
              className="world-map"
              viewBox="0 0 900 465"
              role="img"
              aria-label="World map of bilateral voting agreement"
            >
              {geography.features.map((shape, shapeIndex) => {
                const country = byM49.get(Number(shape.id));
                const entry = country ? byCode.get(country.code) : undefined;
                return (
                  <path
                    key={`${shape.id ?? 'unmapped'}-${shapeIndex}`}
                    d={path(shape) ?? undefined}
                    fill={
                      country?.code === filters.country
                        ? '#a078d3'
                        : agreementColor(entry?.score, midpoint)
                    }
                    stroke={country?.code === filters.country ? '#212529' : '#999'}
                    strokeWidth={country?.code === filters.country ? 1.5 : 0.5}
                    tabIndex={country ? 0 : undefined}
                    role={country ? 'button' : undefined}
                    aria-label={
                      country
                        ? `${country.name}: ${country.code === filters.country ? 'reference country' : score(entry?.score)}`
                        : undefined
                    }
                    onClick={() => country && setSelected(country.code)}
                    onKeyDown={(e) => {
                      if (country && (e.key === 'Enter' || e.key === ' ')) {
                        e.preventDefault();
                        setSelected(country.code);
                      }
                    }}
                  >
                    <title>
                      {country?.name ?? String(shape.properties?.name ?? '')}:{' '}
                      {country?.code === filters.country
                        ? 'Reference country'
                        : entry?.score == null
                          ? 'No shared votes / no data'
                          : `Agreement ${score(entry.score)} · ${entry.shared_votes} shared votes`}
                    </title>
                  </path>
                );
              })}
            </svg>
            <div className="map-legend">
              <div className="agreement-scale">
                <i style={{ background: agreementGradient(midpoint) }} />
                <div>
                  <span>0 · Always opposed</span>
                  <span>1 · Always agreeing</span>
                </div>
                {adaptive && canAdapt && (
                  <p>Yellow midpoint: {score(midpoint)} average consensus</p>
                )}
              </div>
              <span className="reference-key">Reference country</span>
              <span className="no-data-key">No shared votes / no mapped data</span>
            </div>
          </div>
          <p className="map-source-note">
            <strong>About this map:</strong> Natural Earth boundaries, rendered with the original
            Robinson projection; this is not an official UN map. A simplified, static geography may
            omit small states and does not show historical border changes. All comparisons remain
            available in the table. The boundaries and names shown and the designations used do not
            imply official endorsement or acceptance by the United Nations.{' '}
            <a href="https://github.com/topojson/world-atlas" target="_blank" rel="noreferrer">
              Map data source
            </a>{' '}
            ·{' '}
            <a
              href="https://www.un.org/geospatial/sites/www.un.org.geospatial/files/files/documents/2020/Nov/world_map_4170_r19_oct20.pdf"
              target="_blank"
              rel="noreferrer"
            >
              UN-published world map (PDF)
            </a>
          </p>
          <div className="map-detail" aria-live="polite">
            {detail ? (
              <>
                <strong>{names.get(detail.country)}</strong>
                <span>
                  Agreement {score(detail.score)} · {number(detail.shared_votes)} shared votes
                </span>
                <Link to={`/profile?country=${detail.country}`}>Open profile →</Link>
              </>
            ) : (
              <span>
                Select a country on the map, or use the table below. Historical entities may appear
                only in the table.
              </span>
            )}
          </div>
          <details className="data-table-details">
            <summary>View all country comparisons ({data.items.length})</summary>
            <table>
              <thead>
                <tr>
                  <th>Country</th>
                  <th>Agreement</th>
                  <th>Shared votes</th>
                </tr>
              </thead>
              <tbody>
                {[...data.items]
                  .sort((a, b) =>
                    (names.get(a.country) ?? a.country).localeCompare(
                      names.get(b.country) ?? b.country,
                    ),
                  )
                  .map((row) => (
                    <tr key={row.country}>
                      <td>
                        <Link to={`/profile?country=${row.country}`}>
                          {names.get(row.country) ?? row.country}
                        </Link>
                      </td>
                      <td>{score(row.score)}</td>
                      <td>{number(row.shared_votes)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </details>
        </>
      )}
    </Panel>
  );
}
