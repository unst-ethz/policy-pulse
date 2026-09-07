import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useApi, type Model } from '../api/client';
import {
  ErrorPanel,
  Loading,
  Panel,
  safeSourceUrl,
  score,
  VoteBadge,
  VoteComposition,
} from '../components/ui';

export default function Resolution() {
  const { id } = useParams();
  const [search, setSearch] = useState('');
  const query = useApi<Model<'ResolutionDetail'>>(`/resolutions/${encodeURIComponent(id ?? '')}`);
  const metadata = useApi<Model<'Metadata'>>('/metadata', { language: 'en' });
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
  const source = safeSourceUrl(data.source_url);
  const names = Object.fromEntries(metadata.data!.countries.map((c) => [c.code, c.display_name]));
  const votes = Object.entries(data.votes)
    .filter(([code]) => `${code} ${names[code]}`.toLowerCase().includes(search.toLowerCase()))
    .sort(([a], [b]) => (names[a] ?? a).localeCompare(names[b] ?? b));
  return (
    <article className="page resolution-detail">
      <Link className="back-link" to="/trends">
        ← Browse resolutions
      </Link>
      <span className="eyebrow">
        {data.symbol ?? data.id} · {data.date} · SESSION {data.session}
      </span>
      <h1>{data.title}</h1>
      <div className="resolution-detail-links">
        <span>
          Consensus <strong>{score(data.consensus)}</strong>
        </span>
        {source && (
          <a className="button secondary" href={source} target="_blank" rel="noreferrer">
            Read the original UN record ↗
          </a>
        )}
      </div>
      <Panel title="Recorded vote totals">
        <VoteComposition counts={data.counts} />
      </Panel>
      <Panel title="UN subjects">
        {data.subjects.length ? (
          <div className="subject-tags">
            {data.subjects.map((s) => (
              <Link key={s.id} to={`/trends?subject=${encodeURIComponent(s.id)}`}>
                {s.label}
              </Link>
            ))}
          </div>
        ) : (
          <p>No subject assigned in the dataset.</p>
        )}
      </Panel>
      <Panel
        title="Country votes"
        description="X means did not vote. Missing observations may represent non-membership or unavailable data."
      >
        <input
          className="country-search"
          placeholder="Search countries…"
          aria-label="Search country votes"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="votes-grid">
          {votes.map(([code, vote]) => (
            <div key={code}>
              <Link to={`/profile?country=${code}`}>{names[code] ?? code}</Link>
              <VoteBadge vote={vote} />
            </div>
          ))}
        </div>
        {!votes.length && <p>No countries match your search.</p>}
      </Panel>
    </article>
  );
}
