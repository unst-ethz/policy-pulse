import { Link } from 'react-router-dom';

const examples = [
  {
    title: 'How has Switzerland voted at the UN?',
    href: '/trends?country=CHE&compare=DEU&view=timeline',
    action: 'Explore Switzerland and Germany',
    steps: [
      'Choose Switzerland as the reference country. Browse its recorded votes, then select a UN subject or keyword to narrow the resolution list.',
      'Open the agreement map to inspect pairwise voting similarity. The sample count shows how many resolutions each comparison is based on.',
      'Add Germany as a comparison and open the agreement timeline. It shows full-history session means, with at least three shared votes per point.',
      'Open By subject to compare the same countries across subject areas. Set a date range here and select a broader subject to inspect its descendants. At least 30 shared votes are required.',
      'Try a comparison preset, inspect individual UN source records, and share the filtered URL or download the matching resolution data as CSV.',
    ],
  },
  {
    title: 'Diving deeper: Bulgaria and Angola over time',
    href: '/trends?country=BGR&compare=AGO&view=timeline',
    action: 'Explore Bulgaria and Angola',
    steps: [
      'Choose Bulgaria as the reference country and Angola as the comparison. Examine the session timeline for changes in their recorded voting similarity.',
      'Open By subject to see whether the aggregate pattern is shared across UN subject areas. Sample sizes and the selected resolutions affect these comparisons.',
      'Compare subject results in successive windows around 1990, for example 1980–1985, 1985–1990, 1990–1995 and 1995–2000. The timeline itself remains a full-history view.',
      'Return to Resolutions with the subject and period selected. Use the recorded-vote comparison filter to inspect same, different, or Yes-versus-No votes and follow the original UN records.',
      'Treat shifts as descriptive observations. Voting records alone do not establish what caused a change or explain a country’s intentions. Export the records to examine the composition of each sample.',
    ],
  },
];

export function CaseStudies() {
  return (
    <section className="case-study" id="case-study">
      <h2 className="ruled-title">
        <span>Case Studies: How can you use the platform?</span>
      </h2>
      {examples.map((example) => (
        <details className="case-study-example" key={example.title}>
          <summary>{example.title}</summary>
          <ol>
            {example.steps.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <Link className="button secondary" to={example.href}>
            {example.action} →
          </Link>
        </details>
      ))}
    </section>
  );
}
