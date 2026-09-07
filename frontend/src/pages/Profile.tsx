import { Link, useSearchParams } from 'react-router-dom';
import { Printer } from 'lucide-react';
import { useApi, type Model } from '../api/client';
import { AgreementChart } from '../components/AgreementChart';
import {
  Empty,
  ErrorPanel,
  Loading,
  number,
  Panel,
  ResolutionCard,
  score,
  VoteComposition,
} from '../components/ui';
import { parseFilters, serializeFilters } from '../lib/filters';

export default function Profile() {
  const [params] = useSearchParams();
  const filters = parseFilters(params);
  const query = useApi<Model<'Profile'>>(
    `/countries/${filters.country}/profile`,
    { start_date: filters.start_date, end_date: filters.end_date, compare: filters.compare },
    Boolean(filters.country),
  );
  const metadata = useApi<Model<'Metadata'>>('/metadata', { language: 'en' });
  if (!filters.country)
    return (
      <div className="page">
        <Empty title="Select a country first">
          <Link to="/trends">Return to the explorer and choose a reference country.</Link>
        </Empty>
      </div>
    );
  if (query.isPending || metadata.isPending) return <Loading />;
  if (query.error || metadata.error)
    return (
      <ErrorPanel
        error={query.error ?? metadata.error!}
        retry={() => {
          query.refetch();
          metadata.refetch();
        }}
      />
    );
  const data = query.data!;
  const names = Object.fromEntries(metadata.data!.countries.map((c) => [c.code, c.name]));
  const comparisons = data.yearly.length ? Object.keys(data.yearly[0].scores) : [];
  const ranking = (title: string, rows: Model<'AgreementRow'>[]) => (
    <Panel
      title={title}
      description={`At least ${data.minimum_shared_votes} shared votes are required.`}
    >
      {rows.length ? (
        <table>
          <thead>
            <tr>
              <th>Country</th>
              <th>Agreement</th>
              <th>Shared votes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.country}>
                <td>
                  <Link to={`/profile?country=${row.country}`}>
                    {names[row.country] ?? row.country}
                  </Link>
                </td>
                <td>{score(row.score)}</td>
                <td>{number(row.shared_votes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Empty title="Insufficient shared votes" />
      )}
    </Panel>
  );
  return (
    <article className="page profile">
      <div className="page-title">
        <div>
          <Link className="back-link no-print" to={`/trends?${serializeFilters(filters)}`}>
            ← Back to explorer
          </Link>
          <span className="eyebrow">COUNTRY VOTING PROFILE</span>
          <h1>{data.country.display_name}</h1>
          <p>
            {data.country.subregion ?? data.country.region} · {data.start_date ?? 'First record'} to{' '}
            {data.end_date ?? 'latest record'}
          </p>
        </div>
        <button className="primary no-print" onClick={() => window.print()}>
          <Printer size={15} /> Print / Save as PDF
        </button>
      </div>
      <section className="stats-grid">
        <div className="stat-card">
          <strong>{number(data.resolution_count)}</strong>
          <span>Resolutions in period</span>
        </div>
        <div className="stat-card">
          <strong>{score(data.alignment)}</strong>
          <span>Multilateral alignment</span>
        </div>
        <div className="stat-card">
          <strong>{data.rank ? `${data.rank} / ${data.ranked_countries}` : '—'}</strong>
          <span>Alignment rank in period</span>
        </div>
        <div className="stat-card">
          <strong>{data.country.membership_start?.slice(0, 4) ?? '—'}</strong>
          <span>First membership year</span>
        </div>
      </section>
      <Panel
        title="Recorded votes"
        description="Composition across all resolutions in the profile period, including non-votes."
      >
        <VoteComposition counts={data.votes} />
      </Panel>
      <Panel
        title="Bilateral agreement by year"
        description="Annual means with selected comparison countries, or the P5 when no comparison is selected."
      >
        {data.yearly.length ? (
          <AgreementChart
            data={data.yearly.map((p) => ({ year: p.year, ...p.scores }))}
            countries={comparisons}
            names={names}
          />
        ) : (
          <Empty title="No overlapping votes" />
        )}
      </Panel>
      <div className="two-column">
        {ranking('Most aligned voting records', data.most_aligned)}
        {ranking('Least aligned voting records', data.least_aligned)}
      </div>
      <div className="two-column">
        <Panel
          title="No or abstention on high-consensus resolutions"
          description="Up to five records, ordered by resolution consensus."
        >
          {data.opposed_high_consensus.length ? (
            data.opposed_high_consensus.map((r) => (
              <ResolutionCard key={r.id} resolution={r} countryNames={names} />
            ))
          ) : (
            <Empty title="No matching records" />
          )}
        </Panel>
        <Panel
          title="Yes on low-consensus resolutions"
          description="Up to five records, ordered by resolution consensus."
        >
          {data.supported_low_consensus.length ? (
            data.supported_low_consensus.map((r) => (
              <ResolutionCard key={r.id} resolution={r} countryNames={names} />
            ))
          ) : (
            <Empty title="No matching records" />
          )}
        </Panel>
      </div>
      <p className="scope-note">
        Dates are clamped to the existing membership year range. Rankings describe voting
        similarity; they do not assess positions.{' '}
        <Link to="/methodology">Read all definitions.</Link>
      </p>
    </article>
  );
}
