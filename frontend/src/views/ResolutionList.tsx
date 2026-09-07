import { Download } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { apiUrl, useApi, type Model } from '../api/client';
import { Empty, ErrorPanel, Loading, number, ResolutionCard } from '../components/ui';
import { commonParams } from '../lib/filters';
import type { ViewProps } from './types';

export function ResolutionList({ filters, metadata }: ViewProps) {
  const [search, setSearch] = useSearchParams();
  const offset = Math.max(0, Number(search.get('offset')) || 0);
  const sort = search.get('sort') ?? 'date_desc';
  const vote = filters.country && !filters.compare.length ? (search.get('vote') ?? '') : '';
  const agreement =
    filters.country && filters.compare.length === 1 ? (search.get('agreement') ?? '') : '';
  const params = {
    ...commonParams(filters, 'resolutions'),
    compare: filters.compare,
    sort,
    offset,
    limit: 20,
    vote,
    agreement,
  };
  const query = useApi<Model<'ResolutionPage'>>('/resolutions', params);
  const names = Object.fromEntries(metadata.countries.map((c) => [c.code, c.name]));
  const change = (key: string, value: string) => {
    const next = new URLSearchParams(search);
    next.set(key, value);
    if (key !== 'offset') next.delete('offset');
    setSearch(next);
  };
  return (
    <section className="panel resolution-list">
      <div className="list-toolbar">
        <div>
          <h2>Resolutions</h2>
          <p aria-live="polite">
            {query.data ? `${number(query.data.total)} matching records` : 'Retrieving records…'}
          </p>
        </div>
        <a className="button secondary small" href={apiUrl('/resolutions/export.csv', params)}>
          <Download size={14} />
          Download CSV
        </a>
      </div>
      <div className="list-controls">
        <label>
          Sort by
          <select value={sort} onChange={(e) => change('sort', e.target.value)}>
            <option value="date_desc">Newest first</option>
            <option value="date_asc">Oldest first</option>
            <option value="consensus_desc">Highest consensus</option>
            <option value="consensus_asc">Lowest consensus</option>
          </select>
        </label>
        {filters.country && !filters.compare.length && (
          <label>
            Recorded vote
            <select value={vote} onChange={(e) => change('vote', e.target.value)}>
              <option value="">All votes</option>
              <option value="Y">Yes</option>
              <option value="N">No</option>
              <option value="A">Abstain</option>
              <option value="X">Did not vote</option>
            </select>
          </label>
        )}
        {filters.country && filters.compare.length === 1 && (
          <label>
            Recorded vote comparison
            <select value={agreement} onChange={(e) => change('agreement', e.target.value)}>
              <option value="">All comparisons</option>
              <option value="AGREED">Same recorded vote</option>
              <option value="DISAGREED">Different recorded vote</option>
              <option value="STRONGLY_DISAGREED">Yes versus No</option>
            </select>
          </label>
        )}
      </div>
      {query.isPending ? (
        <Loading />
      ) : query.error ? (
        <ErrorPanel error={query.error} retry={() => query.refetch()} />
      ) : (
        query.data && (
          <>
            {query.data.items.length ? (
              query.data.items.map((r) => (
                <ResolutionCard key={r.id} resolution={r} countryNames={names} />
              ))
            ) : (
              <Empty />
            )}
            <div className="pagination">
              <span>
                {query.data.total
                  ? `${offset + 1}–${Math.min(offset + query.data.limit, query.data.total)} of ${number(query.data.total)}`
                  : '0 records'}
              </span>
              <div>
                <button
                  disabled={offset === 0}
                  onClick={() => change('offset', String(Math.max(0, offset - 20)))}
                >
                  Previous
                </button>
                <button
                  disabled={offset + 20 >= query.data.total}
                  onClick={() => change('offset', String(offset + 20))}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )
      )}
    </section>
  );
}
