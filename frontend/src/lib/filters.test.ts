import { describe, expect, it } from 'vitest';
import { commonParams, emptyFilters, parseFilters, serializeFilters } from './filters';

describe('shareable filters', () => {
  it('round trips repeated countries and subject URIs without losing keyword syntax', () => {
    const input = {
      ...emptyFilters,
      country: 'USA',
      compare: ['CHE', 'FRA'],
      subject: ['http://metadata.un.org/thesaurus/03'],
      keyword: '"nuclear" & disarmament, cooperation',
      start_date: '2020-01-01',
      country_mode: 'voted' as const,
    };
    expect(parseFilters(serializeFilters(input))).toEqual(input);
  });
  it('preserves legacy shared links', () => {
    const filters = parseFilters(
      new URLSearchParams(
        'country1_alpha3=USA&country2=CHE,FRA&subject_ids=abc&country_filter_mode=voted',
      ),
    );
    expect(filters).toMatchObject({
      country: 'USA',
      compare: ['CHE', 'FRA'],
      subject: ['abc'],
      country_mode: 'voted',
    });
  });
  it('does not send irrelevant country and keyword filters to maps or words', () => {
    const filters = {
      ...emptyFilters,
      country: 'USA',
      keyword: 'nuclear',
      country_mode: 'voted' as const,
    };
    expect(commonParams(filters, 'map')).toMatchObject({
      country_mode: 'none',
      keyword: undefined,
    });
    expect(commonParams(filters, 'words')).toMatchObject({
      country: undefined,
      country_mode: 'none',
      keyword: 'nuclear',
    });
  });
  it('deduplicates comparisons and excludes self-comparisons', () => {
    expect(
      parseFilters(new URLSearchParams('country=USA&compare=USA&compare=CHE&compare=CHE')).compare,
    ).toEqual(['CHE']);
  });
});
