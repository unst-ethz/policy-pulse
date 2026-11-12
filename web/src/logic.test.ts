import { describe, expect, it } from 'vitest'
import { divergingScale } from './charts/scales'
import {
  countActive,
  exactTerms,
  hasTerm,
  onlyFilters,
  parseFilters,
  serializeFilters,
  toApiParams,
  toggleTerm,
} from './composables/useFilters'
import { composeChart, crc32, slug, toCsv, zip } from './export'
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
      group: ['CHE', 'FRA'],
      together: 'yes' as const,
      allVoted: true,
      historical: true,
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
      group: 'che',
      together: 'sideways',
      allvoted: '1',
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
      group: ['CHE'],
      together: null,
      allVoted: true,
      historical: false,
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
        group: ['CHE', 'FRA'],
        together: 'split',
        allVoted: false,
        historical: true,
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
      group: ['CHE', 'FRA'],
      group_vote: 'split',
      group_all_voted: undefined,
    })
  })

  it('keeps only the filters a page applies', () => {
    const all = parseFilters({ from: '1990', subject: '07', country: 'CHE', adopted: 'recorded', historical: '1' })
    expect(onlyFilters(all, ['from', 'historical'])).toEqual({ ...parseFilters({}), from: 1990, historical: true })
  })

  it('keeps a group without a way of voting in the URL but not in the API filter', () => {
    const filters = { ...parseFilters({}), group: ['CHE', 'FRA'], allVoted: true }
    expect(serializeFilters(filters)).toEqual({ group: ['CHE', 'FRA'] })
    expect(toApiParams(filters).group).toBeUndefined()
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

describe('export files', () => {
  it('computes the standard CRC-32', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926)
  })

  it('quotes every CSV field and escapes quotes', () => {
    expect(toCsv([['a', 'say "hi"'], ['0,85', '']])).toBe('"a","say ""hi"""\r\n"0,85",""\r\n')
  })

  it('writes a stored ZIP with every entry in its directory', async () => {
    const bytes = new Uint8Array(
      await zip([
        { name: 'a.txt', data: 'hello' },
        { name: 'dir/é.csv', data: new Uint8Array([1, 2, 3]) },
      ]).arrayBuffer(),
    )
    const view = new DataView(bytes.buffer)
    const end = bytes.length - 22
    expect(view.getUint32(0, true)).toBe(0x04034b50)
    expect(view.getUint32(end, true)).toBe(0x06054b50)
    expect(view.getUint16(end + 10, true)).toBe(2)
    // the directory starts where the end record says, with the first entry's CRC
    const directory = view.getUint32(end + 16, true)
    expect(view.getUint32(directory, true)).toBe(0x02014b50)
    expect(view.getUint32(directory + 16, true)).toBe(crc32(new TextEncoder().encode('hello')))
  })

  it('puts title, legend and source line around the chart', () => {
    const { svg, height } = composeChart(
      { svg: '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200"><path/></svg>', width: 400, height: 200 },
      {
        title: 'Votes & more',
        legend: {
          items: [{ label: 'France', color: '#f00', shape: 'line' }],
          scale: { colors: ['#f00', '#00f'], low: 'opposed', high: 'agreed', mark: { at: 0.5, label: 'average' } },
        },
        footer: 'Source: <here>',
        colors: { background: '#fff', ink: '#000', muted: '#666', rule: '#ccc' },
        font: 'sans-serif',
      },
    )
    for (const part of ['Votes &amp; more', '>France<', '>opposed<', '>agreed<', 'average', 'Source: &lt;here&gt;']) {
      expect(svg).toContain(part)
    }
    const top = Number(/<svg x="0" y="(\d+)"/.exec(svg)?.[1])
    expect(top).toBeGreaterThan(60) // below the title and legend
    expect(height).toBeGreaterThanOrEqual(top + 200)
  })

  it('slugs titles into file names', () => {
    expect(slug('Agreement with Côte d’Ivoire, 1990–2020')).toBe('agreement-with-cote-d-ivoire-1990-2020')
    expect(slug('…')).toBe('chart')
  })
})
