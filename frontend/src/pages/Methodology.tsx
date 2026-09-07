import { useApi, type Model } from '../api/client';
import { ErrorPanel, Loading } from '../components/ui';

export default function Methodology() {
  const query = useApi<Model<'Methodology'>>('/methodology');
  if (query.isPending) return <Loading />;
  if (query.error) return <ErrorPanel error={query.error} retry={() => query.refetch()} />;
  const data = query.data!;
  return (
    <article className="page methodology">
      <span className="eyebrow">SOURCES, DEFINITIONS & LIMITATIONS</span>
      <h1>How to read Policy Pulse</h1>
      <p className="lead">
        {data.scope} These measures describe voting similarity. They do not evaluate a country's
        position or infer its intentions.
      </p>
      <section className="formula-card">
        <h2>The agreement formula</h2>
        <code>{data.formula}</code>
        <div className="encoding">
          Yes = +1 <span>Abstain = 0</span> No = −1
        </div>
        <p>
          X and missing observations are excluded. An undefined score is displayed as “—”, never as
          zero.
        </p>
      </section>
      <div className="definition-grid">
        {Object.entries(data.definitions).map(([key, text]) => (
          <section key={key}>
            <h2>{key.replaceAll('_', ' ')}</h2>
            <p>{text}</p>
          </section>
        ))}
      </div>
      <section className="panel">
        <h2>Minimum sample sizes</h2>
        <table>
          <thead>
            <tr>
              <th>Measure</th>
              <th>Minimum observations</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.thresholds).map(([key, value]) => (
              <tr key={key}>
                <td>{key.replaceAll('_', ' ')}</td>
                <td>{value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section>
        <h2>Limitations</h2>
        <ul className="limitations">
          {data.limitations.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
      </section>
      <section>
        <h2>Data sources</h2>
        <p>
          Records and subject metadata are provided by the UN Digital Library and UNBIS Thesaurus.
        </p>
        <ul>
          {data.sources.map((url, i) => (
            <li key={url}>
              <a href={url} target="_blank" rel="noreferrer">
                {[
                  'General Assembly voting data',
                  'UNBIS Thesaurus',
                  'UN member state authority list',
                ][i] ?? 'Source record'}
              </a>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}
