import { AlertCircle, ArrowUpRight, LoaderCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { ReactNode } from 'react';
import type { ApiError, Model } from '../api/client';

export const number = (value: number | null | undefined) =>
  value == null ? '—' : value.toLocaleString('en-US');
export const score = (value: number | null | undefined) => (value == null ? '—' : value.toFixed(3));
export const percent = (value: number | null | undefined) =>
  value == null ? '—' : `${(value * 100).toFixed(1)}%`;
export function safeSourceUrl(value: string | null | undefined) {
  if (!value) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : undefined;
  } catch {
    return undefined;
  }
}
export function Loading({ text = 'Loading records…' }: { text?: string }) {
  return (
    <div className="state" role="status">
      <LoaderCircle className="spin" size={22} />
      <p>{text}</p>
    </div>
  );
}
export function ErrorPanel({ error, retry }: { error: ApiError | Error; retry?: () => void }) {
  return (
    <div className="state error" role="alert">
      <AlertCircle size={25} />
      <h3>
        {'status' in error && error.status === 503
          ? 'Dataset temporarily unavailable'
          : 'Unable to load this view'}
      </h3>
      <p>{error.message}</p>
      {retry && <button onClick={retry}>Try again</button>}
    </div>
  );
}
export function Empty({
  title = 'No matching resolutions',
  children,
}: {
  title?: string;
  children?: ReactNode;
}) {
  return (
    <div className="state">
      <h3>{title}</h3>
      <p>{children ?? 'Try a wider date range or adjust the filters.'}</p>
    </div>
  );
}
export function Panel({
  title,
  description,
  children,
  action,
}: {
  title: string;
  description?: ReactNode;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
const voteLabels = { Y: 'Yes', N: 'No', A: 'Abstain', X: 'Did not vote' } as const;
export function VoteBadge({ vote }: { vote: string | null | undefined }) {
  return (
    <span className={`vote vote-${vote ?? 'missing'}`}>
      {vote && vote in voteLabels
        ? voteLabels[vote as keyof typeof voteLabels]
        : 'Non-member / no data'}
    </span>
  );
}
export function VoteSummary({ counts }: { counts: Model<'VoteCounts'> }) {
  return (
    <div className="vote-summary">
      <span>
        Yes <b>{number(counts.yes)}</b>
      </span>
      <span>
        No <b>{number(counts.no)}</b>
      </span>
      <span>
        Abstain <b>{number(counts.abstain)}</b>
      </span>
    </div>
  );
}
export function ResolutionCard({
  resolution: r,
  countryNames = {},
}: {
  resolution: Model<'Resolution'>;
  countryNames?: Record<string, string>;
}) {
  const source = safeSourceUrl(r.source_url);
  return (
    <article className="resolution-card">
      <div className="resolution-meta">
        <span>{r.symbol ?? r.id}</span>
        <time>{r.date ?? 'Date unavailable'}</time>
        <span>Session {r.session ?? '—'}</span>
      </div>
      <h3>
        <Link to={`/resolutions/${encodeURIComponent(r.id)}`}>{r.title}</Link>
      </h3>
      <div className="resolution-bottom">
        <VoteSummary counts={r.counts} />
        <span className="consensus">
          Consensus <b>{score(r.consensus)}</b>
        </span>
        {source && (
          <a href={source} target="_blank" rel="noreferrer" className="source-link">
            UN record <ArrowUpRight size={13} />
          </a>
        )}
      </div>
      {Object.keys(r.votes).length > 0 && (
        <div className="country-votes">
          {Object.entries(r.votes).map(([code, vote]) => (
            <span key={code}>
              <Link to={`/profile?country=${code}`}>{countryNames[code] ?? code}</Link>
              <VoteBadge vote={vote} />
            </span>
          ))}
        </div>
      )}
    </article>
  );
}
export function VoteComposition({ counts }: { counts: Model<'VoteCounts'> }) {
  const values = [
    ['Yes', counts.yes, 'Y'],
    ['No', counts.no, 'N'],
    ['Abstain', counts.abstain, 'A'],
    ['Did not vote', counts.not_voting, 'X'],
  ] as const;
  const total = values.reduce((sum, [, count]) => sum + (count ?? 0), 0);
  return (
    <div>
      <div className="vote-bar" aria-hidden="true">
        {values.map(([label, count, code]) => (
          <span
            key={code}
            className={`bar-${code}`}
            style={{ width: `${total ? ((count ?? 0) / total) * 100 : 0}%` }}
            title={`${label}: ${number(count)}`}
          />
        ))}
      </div>
      <div className="vote-legend">
        {values.map(([label, count, code]) => (
          <span key={code}>
            <i className={`bar-${code}`} />
            {label} <b>{number(count)}</b>
          </span>
        ))}
      </div>
    </div>
  );
}
