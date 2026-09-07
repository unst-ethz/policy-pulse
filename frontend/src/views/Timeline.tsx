import { useState } from 'react';
import { useApi, type Model } from '../api/client';
import { AgreementChart } from '../components/AgreementChart';
import { Empty, ErrorPanel, Loading, number, Panel, score } from '../components/ui';
import type { ViewProps } from './types';

export default function Timeline({ filters, metadata }: ViewProps) {
  const [special, setSpecial] = useState(false);
  const enabled = Boolean(filters.country && filters.compare.length);
  const query = useApi<Model<'TimelineResult'>>(
    `/analysis/timeline/${filters.country}`,
    { compare: filters.compare, include_special: special },
    enabled,
  );
  const names = Object.fromEntries(metadata.countries.map((c) => [c.code, c.name]));
  if (!enabled)
    return (
      <Panel title="Agreement timeline">
        <Empty title="Choose countries to compare">
          Select a reference country and at least one comparison country, then apply filters.
        </Empty>
      </Panel>
    );
  if (query.isPending) return <Loading />;
  if (query.error) return <ErrorPanel error={query.error} retry={() => query.refetch()} />;
  const data = query.data!;
  const valid = data.items.some((point) => Object.values(point.scores).some((v) => v !== null));
  return (
    <Panel
      title="Agreement over time"
      description="Full-history session averages. Date, subject, and keyword filters do not apply to this view."
    >
      <label className="checkbox-label">
        <input type="checkbox" checked={special} onChange={(e) => setSpecial(e.target.checked)} />
        Include special and emergency sessions
      </label>
      {valid ? (
        <AgreementChart
          data={data.items.map((point) => ({
            year: point.year,
            session: point.session,
            special: point.special,
            ...point.scores,
          }))}
          countries={data.comparisons}
          names={names}
        />
      ) : (
        <Empty title="No sessions meet the sample threshold">
          At least {data.minimum_shared_votes} shared votes are required per session.
        </Empty>
      )}
      <p className="chart-note">
        Each point is a mean across resolutions where both countries voted. At least 3 shared votes
        are required. The year comes from the session's median resolution date. Black dots mark
        special or emergency sessions.
      </p>
      <details className="data-table-details">
        <summary>View session values and sample sizes</summary>
        <table>
          <thead>
            <tr>
              <th>Session</th>
              <th>Year</th>
              {data.comparisons.map((c) => (
                <th key={c}>{names[c] ?? c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.items.map((point) => (
              <tr key={point.session}>
                <td>{point.session}</td>
                <td>{point.year}</td>
                {data.comparisons.map((c) => (
                  <td key={c}>
                    {score(point.scores[c])} <small>({number(point.shared_votes[c])} votes)</small>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </Panel>
  );
}
