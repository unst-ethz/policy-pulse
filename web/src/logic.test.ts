import { describe, expect, it } from 'vitest'
import { divergingScale } from './charts/scales'
import {
  countActive,
  exactTerms,
  hasTerm,
  parseFilters,
  serializeFilters,
  toApiParams,
  toggleTerm,
} from './composables/useFilters'
import { cleanTitle, sentenceCase } from './format'

describe('URL filters', () => {
  it('round-trips through the query string', () => {
    const filters = {
      from: 1990,
      to: 2020,
      subjects: ['01', '14'],
      country: 'CHE',
      mode: 'member' as const,
      keyword: 'nuclear & weapons',
      sponsor: 'USA',
      original: true,
      adoption: ['recorded' as const, 'without_vote' as const],
    }
    expect(parseFilters(serializeFilters(filters))).toEqual(filters)
  })

  it('ignores junk and normalises codes', () => {
    const parsed = parseFilters({
      from: 'abc',
      to: '1800',
      country: 'che',
      mode: 'sideways',
      subject: '07',
      adopted: ['recorded', 'maybe'],
    })
    expect(parsed).toEqual({
      from: null,
      to: null,
      subjects: ['07'],
      country: 'CHE',
      mode: 'none',
      keyword: '',
      sponsor: null,
      original: false,
      adoption: ['recorded'],
    })
  })

  it('drops the country mode without a country', () => {
    const query = serializeFilters({ ...parseFilters({}), mode: 'voted' })
    expect(query).toEqual({})
  })

  it('drops "original sponsors only" without a sponsor', () => {
    expect(serializeFilters({ ...parseFilters({}), original: true })).toEqual({})
    expect(toApiParams(parseFilters({ original: '1' })).original_sponsor).toBeUndefined()
  })

  it('maps to the API filter parameters', () => {
    expect(
      toApiParams({
        from: 2000,
        to: 2001,
        subjects: [],
        country: 'USA',
        mode: 'voted',
        keyword: ' x ',
        sponsor: 'CHE',
        original: true,
        adoption: ['non_recorded'],
      }),
    ).toEqual({
      start_date: '2000-01-01',
      end_date: '2001-12-31',
      subject: undefined,
      country: 'USA',
      country_mode: 'voted',
      keyword: 'x',
      sponsor: 'CHE',
      original_sponsor: true,
      adoption: ['non_recorded'],
    })
  })

  it('counts active filters the way the clear button reports them', () => {
    expect(countActive(parseFilters({ from: '2000', to: '2010', q: 'a', country: 'CHE', sponsor: 'CHE' }))).toBe(3)
  })
})

describe('diverging scale', () => {
  const scale = divergingScale('#ff0000', '#808080', '#0000ff')

  it('hits the poles and the midpoint exactly', () => {
    expect(scale(0)).toBe('#ff0000')
    expect(scale(0.5)).toBe('#808080')
    expect(scale(1)).toBe('#0000ff')
  })

  it('clamps out-of-range values', () => {
    expect(scale(-1)).toBe('#ff0000')
    expect(scale(2)).toBe('#0000ff')
  })

  it('moves the neutral colour to the given centre', () => {
    const centred = divergingScale('#ff0000', '#808080', '#0000ff', [0, 1], 0.8)
    expect(centred(0.8)).toBe('#808080')
    expect(centred(0.5)).not.toBe('#808080')
  })
})

describe('formatting', () => {
  it('drops the catalogue boilerplate from titles, typos included', () => {
    expect(cleanTitle('Peace : resolution / adopted by the General Assembly')).toBe('Peace')
    expect(cleanTitle('Peace : resolution / adopted by the General Assemly')).toBe('Peace')
    expect(cleanTitle('Peace: a title with a colon')).toBe('Peace: a title with a colon')
    expect(cleanTitle(null)).toBe('')
  })

  it('sentence-cases capitalised thesaurus labels only', () => {
    expect(sentenceCase('SOCIAL CONDITIONS AND EQUITY')).toBe('Social conditions and equity')
    expect(sentenceCase('Already fine')).toBe('Already fine')
    expect(sentenceCase('社会')).toBe('社会')
  })
})

describe('keyword terms', () => {
  it('adds a term as an exact & term and removes it again', () => {
    expect(toggleTerm('', 'oil slick')).toBe('"oil slick"')
    expect(toggleTerm('nuclear', 'oil slick')).toBe('nuclear & "oil slick"')
    expect(toggleTerm('nuclear & "oil slick" & "peace"', 'oil slick')).toBe('nuclear & "peace"')
  })

  it('recognises only exact terms', () => {
    expect(hasTerm('nuclear & "oil slick"', 'oil slick')).toBe(true)
    expect(hasTerm('oil slick', 'oil slick')).toBe(false)
    expect(hasTerm('"oil slick (2006)"', 'oil slick')).toBe(false)
  })

  it('ignores case and spacing, as the search does', () => {
    expect(hasTerm('"Climate change "', 'climate change')).toBe(true)
    expect(toggleTerm('peace & " Climate change"', 'climate change')).toBe('peace')
  })

  it('lists the exact terms as typed', () => {
    expect(exactTerms('nuclear & "Oil slick" & "peace"')).toEqual(['Oil slick', 'peace'])
  })
})
