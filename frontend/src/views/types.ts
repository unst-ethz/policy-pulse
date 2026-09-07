import type { Model } from '../api/client';
import type { Filters } from '../lib/filters';
export type ViewProps = { filters: Filters; metadata: Model<'Metadata'> };
