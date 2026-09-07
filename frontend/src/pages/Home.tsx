import { Link, useNavigate } from 'react-router-dom';
import { CaseStudies } from '../components/CaseStudies';
import { useApi, type Model } from '../api/client';
import { ErrorPanel, Loading, number, ResolutionCard, VoteComposition } from '../components/ui';
import { emptyFilters, serializeFilters } from '../lib/filters';
import Words from '../views/Words';

export default function Home() {
  const query = useApi<Model<'Overview'>>('/overview');
  const data = query.data;
  const navigate = useNavigate();
  return (
    <div className="page home">
      <section className="hero">
        <img className="hero-image" src="/policy-pulse.webp" alt="" aria-hidden="true" />
        <div className="hero-copy">
          <h1>Welcome to Policy Pulse</h1>
          <p>
            This project aims to make it easier to discover trends in <strong>voting data</strong>{' '}
            from the United Nations. You can currently analyse voting data for{' '}
            <strong>accepted resolutions</strong> of the <strong>UN General Assembly</strong>. More
            data sources and functionalities will be added over time.
          </p>
          <div className="hero-actions">
            <Link className="button primary" to="/trends">
              Explore Data →
            </Link>
            <a className="button case-study-button" href="#case-study">
              Walk me through a case study ↓
            </a>
          </div>
        </div>
      </section>
      {query.isPending ? (
        <Loading />
      ) : query.error ? (
        <ErrorPanel error={query.error} retry={() => query.refetch()} />
      ) : (
        data && (
          <>
            <h2 className="ruled-title">
              <span>Recent updates</span>
            </h2>
            <p className="section-description">
              Most recent resolutions in the dataset, through{' '}
              {data.latest_date ?? 'the latest available date'}.{' '}
              <Link to="/trends">Browse all resolutions →</Link>
            </p>
            <div className="recent-grid">
              {data.recent.map((resolution) => (
                <ResolutionCard key={resolution.id} resolution={resolution} />
              ))}
            </div>
            <h2 className="ruled-title">
              <span>At a quick glance</span>
            </h2>
            <div className="home-glance">
              <section aria-label="Dataset overview" className="home-statistics">
                <dl>
                  {[
                    ['Accepted Resolutions', number(data.resolutions), 'Total records'],
                    [
                      'Year Span',
                      `${data.earliest_date?.slice(0, 4) ?? '—'}–${data.latest_date?.slice(0, 4) ?? '—'}`,
                      'Years covered',
                    ],
                    [
                      'Countries',
                      number(data.countries),
                      'UN member entities appearing in the dataset',
                    ],
                    [
                      'Subjects',
                      number(data.subjects),
                      `${number(data.subject_links)} resolution–subject links`,
                    ],
                  ].map(([label, value, hint]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd>{value}</dd>
                      <small>{hint}</small>
                    </div>
                  ))}
                </dl>
                <h3>Vote Composition Across Accepted Resolutions</h3>
                <VoteComposition counts={data.votes} />
              </section>
              <Words
                filters={emptyFilters}
                initialMode="category"
                onWord={(term, subjects) =>
                  navigate(
                    `/trends?${serializeFilters({ ...emptyFilters, subject: subjects, keyword: subjects.length ? '' : `"${term}"` })}`,
                  )
                }
              />
            </div>
          </>
        )
      )}
      <h2 className="ruled-title">
        <span>About the platform</span>
      </h2>
      <p className="about-intro">
        The Policy Pulse platform is built by volunteers of the United Nations Student Team (UNST)
        at ETH Zürich — a student-run initiative that bridges STEM fields and international policy.
        In collaboration with the UN Dag Hammarskjöld Library, we aim to make the voting data in the
        UN Digital Library more accessible to delegates, students, researchers, and anyone with an
        interest in international relations.
      </p>
      <div className="about-columns">
        <section>
          <h3>Features</h3>
          <ul>
            <li>
              <strong>Resolution List:</strong> Browse recorded votes and filter by country, date,
              subject or keyword.
            </li>
            <li>
              <strong>Agreement Map:</strong> Compare a selected country’s voting with other UN
              members.
            </li>
            <li>
              <strong>Agreement Timeline:</strong> Follow voting agreement session by session.
            </li>
            <li>
              <strong>Agreement by Subject:</strong> Compare voting across UN subject areas.
            </li>
            <li>
              <strong>Multilateral Overview and Word Cloud:</strong> Explore aggregate alignment and
              recurring terms.
            </li>
          </ul>
        </section>
        <section>
          <h3>Limitations</h3>
          <ul>
            <li>
              Only adopted General Assembly resolutions are included. Withdrawn or rejected
              resolutions and other UN bodies are outside this dataset.
            </li>
            <li>
              Subject areas with more resolutions can have more influence on an aggregate agreement
              score.
            </li>
            <li>
              Voting similarity is descriptive; it does not establish political intent or explain
              why a country voted a particular way.
            </li>
          </ul>
          <Link to="/methodology">Read the methodology and limitations →</Link>
        </section>
      </div>
      <CaseStudies />
    </div>
  );
}
