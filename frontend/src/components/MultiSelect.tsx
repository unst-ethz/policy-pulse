import { useState } from 'react';

type Option = { value: string; label: string; search?: string };
export function MultiSelect({
  label,
  options,
  value,
  onChange,
  disabled = false,
  maxSelections,
}: {
  label: string;
  options: Option[];
  value: string[];
  onChange: (v: string[]) => void;
  disabled?: boolean;
  maxSelections?: number;
}) {
  const [search, setSearch] = useState('');
  if (disabled)
    return (
      <label>
        {label}
        <button className="select-disabled" disabled>
          Not used in this view
        </button>
      </label>
    );
  const filtered = options.filter((o) =>
    `${o.label} ${o.search ?? ''}`.toLocaleLowerCase().includes(search.toLocaleLowerCase()),
  );
  return (
    <div className="multi-select">
      <span className="field-label">{label}</span>
      <details>
        <summary aria-label={label}>
          {value.length ? `${value.length} selected` : `All / none selected`}
        </summary>
        <div className="multi-popover">
          <input
            aria-label={`Search ${label.toLowerCase()}`}
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="multi-actions">
            <button type="button" onClick={() => onChange([])}>
              Clear selection
            </button>
          </div>
          {maxSelections && (
            <p className="field-hint">
              {value.length} of {maxSelections} selections
            </p>
          )}
          <div className="multi-options">
            {filtered.map((option) => (
              <label key={option.value}>
                <input
                  type="checkbox"
                  checked={value.includes(option.value)}
                  disabled={Boolean(
                    maxSelections && value.length >= maxSelections && !value.includes(option.value),
                  )}
                  onChange={(e) =>
                    onChange(
                      e.target.checked
                        ? [...value, option.value]
                        : value.filter((v) => v !== option.value),
                    )
                  }
                />
                {option.label}
              </label>
            ))}
            {!filtered.length && <p>No matches</p>}
          </div>
        </div>
      </details>
    </div>
  );
}
