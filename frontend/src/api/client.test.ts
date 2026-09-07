import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiUrl, get } from './client';

afterEach(() => vi.unstubAllGlobals());
describe('API client', () => {
  it('encodes repeated filters and treats zero as a value', () => {
    expect(
      apiUrl('/resolutions', { compare: ['CHE', 'FRA'], offset: 0, keyword: 'a & b', country: '' }),
    ).toBe('/api/v1/resolutions?compare=CHE&compare=FRA&offset=0&keyword=a+%26+b');
  });
  it('surfaces service failures without substituting empty data', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: { message: 'Dataset unavailable' } }), {
          status: 503,
          headers: { 'X-Request-ID': 'trace-id' },
        }),
      ),
    );
    await expect(get('/overview')).rejects.toMatchObject({
      status: 503,
      message: 'Dataset unavailable',
      requestId: 'trace-id',
    });
  });
  it('passes AbortSignal through to cancel obsolete requests', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response('{}'));
    vi.stubGlobal('fetch', fetch);
    const controller = new AbortController();
    await get('/metadata', {}, controller.signal);
    expect(fetch).toHaveBeenCalledWith(
      '/api/v1/metadata',
      expect.objectContaining({ signal: controller.signal }),
    );
  });
});
