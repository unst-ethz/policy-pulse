import type { Params } from '../api/client';

export type View = 'resolutions' | 'map' | 'timeline' | 'subjects' | 'multilateral' | 'words';
export const views: { id: View; label: string }[] = [
  { id: 'resolutions', label: 'Resolutions' },
  { id: 'map', label: 'Agreement map' },
  { id: 'timeline', label: 'Agreement timeline' },
  { id: 'subjects', label: 'By subject' },
  { id: 'multilateral', label: 'Multilateral alignment' },
  { id: 'words', label: 'Word cloud' },
];
export type Filters = {
  country: string;
  compare: string[];
  subject: string[];
  start_date: string;
  end_date: string;
  keyword: string;
  country_mode: 'none' | 'voted' | 'member';
};
export const emptyFilters: Filters = {
  country: '',
  compare: [],
  subject: [],
  start_date: '',
  end_date: '',
  keyword: '',
  country_mode: 'none',
};

function values(params: URLSearchParams, key: string, legacy?: string) {
  return params
    .getAll(key)
    .concat(legacy ? params.getAll(legacy) : [])
    .flatMap((v) => v.split(','))
    .filter(Boolean);
}
export function parseFilters(params: URLSearchParams): Filters {
  const mode = params.get('country_mode') ?? params.get('country_filter_mode');
  const country =
    params.get('country') ?? params.get('country1_alpha3') ?? params.get('country1') ?? '';
  return {
    country,
    compare: [...new Set(values(params, 'compare', 'country2'))].filter((c) => c !== country),
    subject: values(params, 'subject', 'subject_ids'),
    start_date: params.get('start_date') ?? '',
    end_date: params.get('end_date') ?? '',
    keyword: params.get('keyword') ?? '',
    country_mode: country && (mode === 'voted' || mode === 'member') ? mode : 'none',
  };
}
export function serializeFilters(filters: Filters, view: View = 'resolutions') {
  const params = new URLSearchParams({ view });
  Object.entries(filters).forEach(([key, value]) => {
    if (!value || (key === 'country_mode' && value === 'none')) return;
    (Array.isArray(value) ? value : [value]).forEach((v) => params.append(key, v));
  });
  return params;
}
export function commonParams(filters: Filters, view: View): Params {
  return {
    start_date: filters.start_date,
    end_date: filters.end_date,
    subject: filters.subject,
    country: view === 'words' ? undefined : filters.country,
    country_mode: ['resolutions', 'multilateral'].includes(view) ? filters.country_mode : 'none',
    keyword: ['resolutions', 'words'].includes(view) ? filters.keyword : undefined,
  };
}
