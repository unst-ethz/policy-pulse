import { useState } from 'react';
import { useApi, type Model } from '../api/client';
import { Empty, ErrorPanel, Loading, Panel, score } from '../components/ui';
import { commonParams } from '../lib/filters';
import type { ViewProps } from './types';

export default function Words({
  filters,
  onWord,
  initialMode = 'default',
}: Pick<ViewProps, 'filters'> & {
  onWord: (term: string, subjects: string[]) => void;
  initialMode?: Model<'WordResult'>['mode'];
}) {
  const [mode, setMode] = useState<Model<'WordResult'>['mode']>(initialMode);
  const [coloring, setColoring] = useState<'frequency' | 'consensus'>('frequency');
  const query = useApi<Model<'WordResult'>>(
    `/analysis/words/${mode}`,
    commonParams(filters, 'words'),
  );
  const items = query.data?.items ?? [];
  const max = items[0]?.count ?? 1;
  return (
    <Panel
      title="Subjects and keywords"
      description="Indexed terms associated with the selected resolutions. Select a term to inspect its records."
    >
      <div className="chart-controls">
        <label>
          Term source
          <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
            {[
              ['default', 'Default keywords'],
              ['geopolitical', 'Geopolitical'],
              ['thematic', 'Thematic'],
              ['action', 'Action'],
              ['category', 'UN subjects'],
            ].map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Color by
          <select value={coloring} onChange={(e) => setColoring(e.target.value as typeof coloring)}>
            <option value="frequency">Frequency</option>
            <option value="consensus">Mean resolution consensus</option>
          </select>
        </label>
      </div>
      {query.isPending ? (
        <Loading />
      ) : query.error ? (
        <ErrorPanel error={query.error} retry={() => query.refetch()} />
      ) : !query.data?.available ? (
        <Empty title="Keyword source unavailable">
          The existing keyword asset for this mode is not present. No replacement keywords have been
          inferred.
        </Empty>
      ) : items.length ? (
        <>
          <div className="word-cloud" aria-label="Resolution term cloud">
            {items.map((word) => (
              <button
                key={word.term}
                onClick={() => onWord(word.term, word.subject_ids)}
                style={{
                  fontSize: `${16 + Math.sqrt(word.count / max) * 31}px`,
                  color:
                    coloring === 'consensus'
                      ? word.consensus == null
                        ? '#778598'
                        : `hsl(209 55% ${85 - word.consensus * 55}%)`
                      : `hsl(209 50% ${62 - (word.count / max) * 32}%)`,
                }}
                title={`${word.count} resolutions · Consensus ${score(word.consensus)}`}
              >
                {word.term}
              </button>
            ))}
          </div>
          <details className="data-table-details">
            <summary>View term counts and consensus</summary>
            <table>
              <thead>
                <tr>
                  <th>Term</th>
                  <th>Resolutions</th>
                  <th>Mean consensus</th>
                </tr>
              </thead>
              <tbody>
                {items.map((word) => (
                  <tr key={word.term}>
                    <td>
                      <button
                        className="text-button"
                        onClick={() => onWord(word.term, word.subject_ids)}
                      >
                        {word.term}
                      </button>
                    </td>
                    <td>{word.count}</td>
                    <td>{score(word.consensus)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      ) : (
        <Empty title="No indexed terms in this selection" />
      )}
      <p className="chart-note">
        Each term is counted once per resolution. Keyword assets may not cover newer records.
        Country filters do not apply to this view.
      </p>
    </Panel>
  );
}
