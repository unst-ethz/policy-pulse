import { useState } from 'react';
import { useApi, type Model } from '../api/client';
import { Empty, ErrorPanel, Loading, Panel, score } from '../components/ui';
import type { ViewProps } from './types';

export default function Subjects({ filters, metadata }: ViewProps) {
  const [selection, setSelection] = useState('');
  const [parents, setParents] = useState<string[]>([]);
  const comparison = filters.compare.includes(selection) ? selection : filters.compare[0];
  const parent = parents.at(-1);
  const query = useApi<Model<'SubjectResult'>>(
    `/analysis/subjects/${filters.country}/${comparison}`,
    { start_date: filters.start_date, end_date: filters.end_date, parent },
    Boolean(filters.country && comparison),
  );
  const names = Object.fromEntries(metadata.countries.map((c) => [c.code, c.name]));
  const label = metadata.subjects.find((s) => s.id === parent)?.label;
  if (!filters.country || !comparison)
    return (
      <Panel title="Alignment by subject">
        <Empty title="Choose two countries">
          Select a reference and comparison country, then apply filters.
        </Empty>
      </Panel>
    );
  return (
    <Panel
      title="Alignment by UN subject"
      description="Includes descendants and requires 30 shared votes per subject. Date filters apply; global subject and keyword filters do not."
    >
      <div className="chart-controls">
        <label>
          Compare with
          <select value={comparison} onChange={(e) => setSelection(e.target.value)}>
            {filters.compare.map((c) => (
              <option key={c} value={c}>
                {names[c] ?? c}
              </option>
            ))}
          </select>
        </label>
        {parent && (
          <button onClick={() => setParents((prev) => prev.slice(0, -1))}>
            ← Back to broader subjects
          </button>
        )}
      </div>
      {label && <h3>{label}</h3>}
      {query.isPending ? (
        <Loading />
      ) : query.error ? (
        <ErrorPanel error={query.error} retry={() => query.refetch()} />
      ) : query.data?.items.length ? (
        <div className="subject-bars">
          {query.data.items.map((row) => {
            const expandable = metadata.subjects.some((s) => s.parents.includes(row.subject_id));
            return (
              <div className="subject-row" key={row.subject_id}>
                <div>
                  <button
                    disabled={!expandable}
                    className="subject-name"
                    onClick={() => setParents((prev) => [...prev, row.subject_id])}
                  >
                    {row.subject_label}
                    {expandable ? ' →' : ''}
                  </button>
                  <span>{row.shared_votes} shared votes</span>
                </div>
                <div className="subject-track">
                  <span style={{ width: `${row.score * 100}%` }} />
                </div>
                <strong>{score(row.score)}</strong>
              </div>
            );
          })}
        </div>
      ) : (
        <Empty title="No subjects meet the sample threshold">
          Try a wider date range or return to broader subjects. Scores below 30 shared votes are
          omitted.
        </Empty>
      )}
      <p className="chart-note">
        A resolution is counted once within each subject, even if it has several matching descendant
        labels. Subjects may overlap.
      </p>
    </Panel>
  );
}
