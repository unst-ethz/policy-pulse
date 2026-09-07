import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { safeSourceUrl, score, VoteBadge } from './ui';

describe('neutral presentation', () => {
  it('distinguishes undefined scores from observed disagreement', () => {
    expect(score(null)).toBe('—');
    expect(score(0)).toBe('0.000');
  });
  it('distinguishes non-votes from abstentions and missing records', () => {
    render(
      <>
        <VoteBadge vote="X" />
        <VoteBadge vote="A" />
        <VoteBadge vote={null} />
      </>,
    );
    expect(screen.getByText('Did not vote')).toBeInTheDocument();
    expect(screen.getByText('Abstain')).toBeInTheDocument();
    expect(screen.getByText('Non-member / no data')).toBeInTheDocument();
  });
  it('only allows web URLs from source metadata', () => {
    expect(safeSourceUrl('javascript:alert(1)')).toBeUndefined();
    expect(safeSourceUrl('data:text/html,hello')).toBeUndefined();
    expect(safeSourceUrl('https://digitallibrary.un.org/record/123')).toBe(
      'https://digitallibrary.un.org/record/123',
    );
  });
});
