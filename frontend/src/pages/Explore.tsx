import { Suspense, lazy } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import { useApi, type Model } from '../api/client';
import { FilterPanel } from '../components/FilterPanel';
import { ErrorPanel, Loading } from '../components/ui';
import { parseFilters, serializeFilters, views, type Filters } from '../lib/filters';
import { ResolutionList } from '../views/ResolutionList';

const AgreementMap = lazy(() => import('../views/AgreementMap'));
const Timeline = lazy(() => import('../views/Timeline'));
const Subjects = lazy(() => import('../views/Subjects'));
const Multilateral = lazy(() => import('../views/Multilateral'));
const Words = lazy(() => import('../views/Words'));

export default function Explore() {
  const [params, setParams] = useSearchParams();
  const filters = parseFilters(params);
  const view = views.find((v) => v.id === params.get('view'))?.id ?? 'resolutions';
  const language = params.get('language') ?? 'en';
  const metadata = useApi<Model<'Metadata'>>('/metadata', { language });
  const apply = (value: Filters) => {
    const next = serializeFilters(value, view);
    if (language !== 'en') next.set('language', language);
    setParams(next);
  };
  if (metadata.isPending) return <Loading text="Loading countries and UN subjects…" />;
  if (metadata.error) return <ErrorPanel error={metadata.error} retry={() => metadata.refetch()} />;
  const data = metadata.data!;
  const country = data.countries.find((c) => c.code === filters.country);
  const props = { filters, metadata: data };
  return (
    <div className="page explore">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link to="/">Home</Link>
        <span>›</span>
        <span>Analysis{country ? ` (${country.name})` : ''}</span>
      </nav>
      <div className="page-title">
        <div>
          <h1>Explore GA Votes Yourself</h1>
          <p>Select countries, subjects and a period to explore the voting records.</p>
        </div>
        <div className="page-actions">
          <label className="language-select">
            Country names
            <select
              value={language}
              onChange={(e) => {
                const next = new URLSearchParams(params);
                next.set('language', e.target.value);
                setParams(next);
              }}
            >
              {[
                ['en', 'English'],
                ['fr', 'Français'],
                ['es', 'Español'],
                ['ar', 'العربية'],
                ['zh', '中文'],
                ['ru', 'Русский'],
              ].map(([code, label]) => (
                <option key={code} value={code}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          {country && (
            <Link className="button secondary" to={`/profile?${serializeFilters(filters)}`}>
              Country profile <ArrowUpRight size={14} />
            </Link>
          )}
        </div>
      </div>
      <div className="explore-layout">
        <FilterPanel value={filters} view={view} metadata={data} apply={apply} />
        <div className="explore-content">
          <nav className="view-tabs" aria-label="Analysis views">
            {views.map((v) => (
              <button
                key={v.id}
                aria-current={view === v.id ? 'page' : undefined}
                className={view === v.id ? 'active' : ''}
                onClick={() => {
                  const next = new URLSearchParams(params);
                  next.set('view', v.id);
                  next.delete('offset');
                  next.delete('vote');
                  next.delete('agreement');
                  setParams(next);
                }}
              >
                {v.label}
              </button>
            ))}
          </nav>
          <div className="coverage-line">
            <span className="status-dot" />
            Adopted General Assembly resolutions
            <span>Dataset through {data.latest_date ?? 'unknown'}</span>
          </div>
          <Suspense fallback={<Loading />}>
            {view === 'resolutions' && <ResolutionList {...props} />}
            {view === 'map' && <AgreementMap {...props} />}
            {view === 'timeline' && <Timeline {...props} />}
            {view === 'subjects' && <Subjects {...props} />}
            {view === 'multilateral' && <Multilateral {...props} />}
            {view === 'words' && (
              <Words
                {...props}
                onWord={(term, subjects) => {
                  const next = serializeFilters(
                    {
                      ...filters,
                      country: '',
                      compare: [],
                      country_mode: 'none',
                      keyword: subjects.length
                        ? ''
                        : [filters.keyword, `"${term}"`].filter(Boolean).join(' & '),
                      subject: subjects.length ? subjects : filters.subject,
                    },
                    'resolutions',
                  );
                  setParams(next);
                }}
              />
            )}
          </Suspense>
          <p className="scope-note">
            Scores use the existing Policy Pulse definitions.{' '}
            <Link to="/methodology">Methodology &amp; limitations</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
