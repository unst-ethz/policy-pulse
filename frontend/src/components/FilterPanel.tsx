import { useEffect, useState } from 'react';
import { RotateCcw, SlidersHorizontal } from 'lucide-react';
import type { Model } from '../api/client';
import { type Filters, type View, emptyFilters } from '../lib/filters';
import { MultiSelect } from './MultiSelect';

export function FilterPanel({
  value,
  view,
  metadata,
  apply,
}: {
  value: Filters;
  view: View;
  metadata: Model<'Metadata'>;
  apply: (v: Filters) => void;
}) {
  const [draft, setDraft] = useState(value);
  const [invalid, setInvalid] = useState('');
  const serialized = JSON.stringify(value);
  useEffect(() => {
    setDraft(JSON.parse(serialized));
    setInvalid('');
  }, [serialized]);
  const set = <K extends keyof Filters>(key: K, v: Filters[K]) =>
    setDraft((prev) => ({ ...prev, [key]: v }));
  const noDates = view === 'timeline';
  const noSubject = noDates || view === 'subjects';
  const noCountry = view === 'words';
  const noComparison = ['map', 'multilateral', 'words'].includes(view);
  const noKeyword = !['resolutions', 'words'].includes(view);
  const nameOptions = metadata.countries.map((c) => ({
    value: c.code,
    label: c.display_name,
    search: c.search_terms,
  }));
  return (
    <aside className="filters">
      <div className="filter-title">
        <SlidersHorizontal size={16} />
        <h2>Filter the Data</h2>
      </div>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!noDates && draft.start_date && draft.end_date && draft.start_date > draft.end_date) {
            setInvalid('Start date must be on or before end date.');
            return;
          }
          setInvalid('');
          apply({ ...draft, compare: draft.compare.filter((c) => c !== draft.country) });
        }}
      >
        <div className="filter-fields">
          <label>
            Reference country
            <select
              disabled={noCountry}
              value={draft.country}
              onChange={(e) =>
                setDraft((prev) => ({
                  ...prev,
                  country: e.target.value,
                  country_mode: e.target.value ? prev.country_mode : 'none',
                  compare: prev.compare.filter((c) => c !== e.target.value),
                }))
              }
            >
              <option value="">Select a country</option>
              {metadata.countries.map((c) => (
                <option key={c.code} value={c.code}>
                  {c.display_name}
                </option>
              ))}
            </select>
          </label>
          <MultiSelect
            label="Comparison countries"
            options={nameOptions.filter((c) => c.value !== draft.country)}
            value={draft.compare}
            onChange={(v) => set('compare', v)}
            maxSelections={50}
            disabled={noComparison}
          />
          <label>
            Comparison preset
            <select
              aria-label="Comparison preset"
              disabled={noComparison}
              value=""
              onChange={(e) => {
                const preset = metadata.country_presets.find((p) => p.id === e.target.value);
                if (preset)
                  set(
                    'compare',
                    preset.countries.filter(
                      (c) => c !== draft.country && metadata.countries.some((x) => x.code === c),
                    ),
                  );
              }}
            >
              <option value="">Choose a group…</option>
              {metadata.country_presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Institutional era
            <select
              disabled={noDates}
              value=""
              onChange={(e) => {
                const era = metadata.eras.find((x) => x.id === e.target.value);
                if (era)
                  setDraft((prev) => ({
                    ...prev,
                    start_date: `${era.start}-01-01`,
                    end_date: era.end ? `${era.end}-12-31` : (metadata.latest_date ?? ''),
                  }));
              }}
            >
              <option value="">Choose a period…</option>
              {metadata.eras.map((era) => (
                <option key={era.id} value={era.id}>
                  {era.label}
                </option>
              ))}
            </select>
          </label>
          <div className="filter-dates">
            <div className="date-fields">
              <label>
                From
                <input
                  type="date"
                  disabled={noDates}
                  value={draft.start_date}
                  onChange={(e) => set('start_date', e.target.value)}
                />
              </label>
              <label>
                To
                <input
                  type="date"
                  disabled={noDates}
                  value={draft.end_date}
                  onChange={(e) => set('end_date', e.target.value)}
                />
              </label>
            </div>
            {!noDates && draft.country && (
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  const country = metadata.countries.find((c) => c.code === draft.country);
                  if (country)
                    setDraft((prev) => ({
                      ...prev,
                      start_date: country.membership_start ?? '',
                      end_date: country.membership_end ?? '',
                    }));
                }}
              >
                Use country membership dates
              </button>
            )}
          </div>
          <MultiSelect
            label="UN subjects"
            options={[
              { value: '__no_subject__', label: 'No subject assigned' },
              ...metadata.subjects.map((s) => ({
                value: s.id,
                label: `${s.top_level ? '◆ ' : ''}${s.label}`,
              })),
            ]}
            value={draft.subject}
            onChange={(v) => set('subject', v)}
            maxSelections={100}
            disabled={noSubject}
          />
          <div className="keyword-field">
            <label>
              Keyword expression
              <input
                disabled={noKeyword}
                value={draft.keyword}
                maxLength={300}
                onChange={(e) => set('keyword', e.target.value)}
                placeholder="e.g. disarmament & nuclear"
              />
            </label>
            {!noKeyword && (
              <p className="field-hint">
                Use commas for OR, &amp; for AND, and quotes for exact keyword entries.
              </p>
            )}
          </div>
          <label>
            Country participation
            <select
              disabled={!draft.country || !['resolutions', 'multilateral'].includes(view)}
              value={draft.country_mode}
              onChange={(e) => set('country_mode', e.target.value as Filters['country_mode'])}
            >
              <option value="none">All selected resolutions</option>
              <option value="voted">Country cast a vote</option>
              <option value="member">Within membership dates</option>
            </select>
          </label>
        </div>
        {invalid && (
          <p role="alert" className="form-error">
            {invalid}
          </p>
        )}
        <div className="filter-actions">
          <button type="submit" className="primary apply-button">
            Apply filters
          </button>
          <button
            className="reset-button"
            type="button"
            onClick={() => {
              setDraft(emptyFilters);
              apply(emptyFilters);
            }}
          >
            <RotateCcw size={13} /> Reset filters
          </button>
        </div>
      </form>
      <div className="filter-note">
        Only adopted UN General Assembly resolutions. Missing votes are never scored as
        disagreement.
      </div>
    </aside>
  );
}
