import { useQuery } from '@tanstack/react-query';
import type { components } from './schema';

export type Model<K extends keyof components['schemas']> = components['schemas'][K];
export type Params = Record<string, string | number | boolean | string[] | undefined | null>;
const base = import.meta.env.VITE_API_BASE_URL ?? '';

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public requestId: string | null = null,
  ) {
    super(message);
  }
}

export function apiUrl(path: string, params: Params = {}): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    (Array.isArray(value) ? value : [value]).forEach((v) => query.append(key, String(v)));
  });
  const suffix = query.toString();
  return `${base}/api/v1${path}${suffix ? `?${suffix}` : ''}`;
}

export async function get<T>(path: string, params: Params = {}, signal?: AbortSignal): Promise<T> {
  const response = await fetch(apiUrl(path, params), {
    signal,
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = typeof detail === 'string' ? detail : detail?.message;
    throw new ApiError(
      response.status,
      message || `Request failed (${response.status}).`,
      response.headers.get('X-Request-ID'),
    );
  }
  return response.json() as Promise<T>;
}

export function useApi<T>(path: string, params: Params = {}, enabled = true) {
  return useQuery<T, ApiError>({
    queryKey: [path, params],
    queryFn: ({ signal }) => get<T>(path, params, signal),
    enabled,
    staleTime: 60_000,
    retry: (count, error) => error.status >= 500 && count < 2,
    retryDelay: 1500,
  });
}
