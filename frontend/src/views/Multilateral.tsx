import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useApi, type Model } from '../api/client';
import { Empty, ErrorPanel, Loading, Panel, percent, score } from '../components/ui';
import { commonParams } from '../lib/filters';
import type { ViewProps } from './types';

const regions: Record<string, string> = {
  Africa: '#be7839',
  Asia: '#a55d91',
  Europe: '#397faa',
  Americas: '#549071',
  Oceania: '#7b72ac',
  Other: '#8b939d',
};
export default function Multilateral({ filters, metadata }: ViewProps) {
  const [metric, setMetric] = useState<'abstention_rate' | 'yes_rate' | 'no_rate'>(
    'abstention_rate',
  );
  const query = useApi<Model<'MultilateralResult'>>(
    '/analysis/multilateral',
    commonParams(filters, 'multilateral'),
  );
  if (query.isPending) return <Loading />;
  if (query.error) return <ErrorPanel error={query.error} retry={() => query.refetch()} />;
  const data = query.data!;
  const countries = new Map(metadata.countries.map((c) => [c.code, c]));
  const points = data.items.map((row) => ({
    ...row,
    name: countries.get(row.country)?.name ?? row.country,
    region: countries.get(row.country)?.region ?? 'Other',
  }));
  const selected = points.find((p) => p.country === filters.country);
  return (
    <Panel
      title="Multilateral alignment"
      description={`Agreement with other voting countries, averaged across resolutions. Countries need at least ${data.minimum_votes} votes in this selection.`}
    >
      <div className="chart-controls">
        <label>
          Vertical axis
          <select value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)}>
            <option value="abstention_rate">Abstention rate</option>
            <option value="yes_rate">Yes rate</option>
            <option value="no_rate">No rate</option>
          </select>
        </label>
        <span>
          Mean alignment of displayed countries: <b>{score(data.mean_alignment)}</b>
        </span>
      </div>
      {points.length ? (
        <>
          <div className="chart">
            <ResponsiveContainer width="100%" height={380}>
              <ScatterChart margin={{ top: 20, right: 30, bottom: 35, left: 10 }}>
                <CartesianGrid stroke="#e6ebf1" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="multilateral_alignment"
                  domain={[0, 1]}
                  name="Alignment"
                  label={{ value: 'Multilateral alignment', position: 'bottom', offset: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey={metric}
                  domain={[0, 1]}
                  name={metric.replaceAll('_', ' ')}
                  tickFormatter={percent}
                />
                <Tooltip
                  content={({ active, payload }) =>
                    active && payload?.[0] ? (
                      <div className="chart-tooltip">
                        <strong>{payload[0].payload.name}</strong>
                        <p>Alignment: {score(payload[0].payload.multilateral_alignment)}</p>
                        <p>
                          {metric.replaceAll('_', ' ')}: {percent(payload[0].payload[metric])}
                        </p>
                        <p>{payload[0].payload.participation_count} votes cast</p>
                      </div>
                    ) : null
                  }
                />
                <Legend verticalAlign="top" />
                {Object.entries(regions).map(([region, color]) => (
                  <Scatter
                    key={region}
                    name={region}
                    data={points.filter((p) => p.region === region)}
                    fill={color}
                    fillOpacity={0.8}
                    isAnimationActive={false}
                  />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          {selected && (
            <div className="map-detail">
              <strong>{selected.name}</strong>
              <span>
                Alignment {score(selected.multilateral_alignment)} · {selected.participation_count}{' '}
                votes cast
              </span>
            </div>
          )}
          <details className="data-table-details">
            <summary>View country statistics</summary>
            <table>
              <thead>
                <tr>
                  <th>Country</th>
                  <th>Alignment</th>
                  <th>Yes</th>
                  <th>No</th>
                  <th>Abstain</th>
                  <th>Votes cast</th>
                </tr>
              </thead>
              <tbody>
                {points.map((row) => (
                  <tr key={row.country}>
                    <td>
                      <Link to={`/profile?country=${row.country}`}>{row.name}</Link>
                    </td>
                    <td>{score(row.multilateral_alignment)}</td>
                    <td>{percent(row.yes_rate)}</td>
                    <td>{percent(row.no_rate)}</td>
                    <td>{percent(row.abstention_rate)}</td>
                    <td>{row.participation_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      ) : (
        <Empty title="No countries meet the sample threshold" />
      )}
      <p className="chart-note">
        Vote rates use Y + N + A as their denominator. Non-votes and missing observations are
        excluded. A higher alignment score indicates more similar recorded votes.
      </p>
    </Panel>
  );
}
