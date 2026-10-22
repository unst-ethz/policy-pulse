/**
 * Typed access to the Policy Pulse API.
 *
 * `schema.d.ts` is generated from the API's OpenAPI document (`npm run gen:api` with the API
 * running), so every path, parameter and response shape here is checked against the backend.
 */
import createClient from 'openapi-fetch'
import type { components, paths } from './schema'

export type Schemas = components['schemas']
export type Country = Schemas['Country']
export type Resolution = Schemas['Resolution']
export type ResolutionPage = Schemas['ResolutionPage']
export type ResolutionDetail = Schemas['ResolutionDetail']
export type TreeNode = Schemas['TreeNode']
export type Vote = 'Y' | 'N' | 'A' | 'X'

export const api = createClient<paths>({
  baseUrl: '',
  // Repeatable params go out as `subject=a&subject=b`, which is what FastAPI expects.
  querySerializer: { array: { style: 'form', explode: true } },
})

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message)
  }
}

/** Unwrap an openapi-fetch result: the data, or an ApiError carrying the API's `detail`. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await call
  if (data !== undefined && response.ok) return data
  const detail = (error as { detail?: unknown } | undefined)?.detail
  const message =
    typeof detail === 'string'
      ? detail
      : Array.isArray(detail)
        ? detail.map((d: { msg?: string }) => d.msg).join('; ')
        : `Request failed (${response.status})`
  throw new ApiError(response.status, message)
}
