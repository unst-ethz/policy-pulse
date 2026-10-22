import { describe, expect, it } from 'vitest'
import { divergingScale } from './charts/scales'
import { countActive, parseFilters, serializeFilters, toApiParams } from './composables/useFilters'
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
    }
    expect(parseFilters(serializeFilters(filters))).toEqual(filters)
  })

  it('ignores junk and normalises codes', () => {
    const parsed = parseFilters({ from: 'abc', to: '1800', country: 'che', mode: 'sideways', subject: '07' })
    expect(parsed).toEqual({
      from: null,
      to: null,
      subjects: ['07'],
      country: 'CHE',
      mode: 'none',
      keyword: '',
    })
  })

  it('drops the country mode without a country', () => {
    const query = serializeFilters({ ...parseFilters({}), mode: 'voted' })
    expect(query).toEqual({})
  })

  it('maps to the API filter parameters', () => {
    expect(
      toApiParams({ from: 2000, to: 2001, subjects: [], country: 'USA', mode: 'voted', keyword: ' x ' }),
    ).toEqual({
      start_date: '2000-01-01',
      end_date: '2001-12-31',
      subject: undefined,
      country: 'USA',
      country_mode: 'voted',
      keyword: 'x',
    })
  })

  it('counts active filters the way the clear button reports them', () => {
    expect(countActive(parseFilters({ from: '2000', to: '2010', q: 'a', country: 'CHE' }))).toBe(2)
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
