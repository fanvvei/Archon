/**
 * Typed config parsing for OpenCode provider defaults.
 * Validates and narrows the opaque assistantConfig to typed fields.
 */
import type { OpencodeProviderDefaults } from '../types';

// Re-export so consumers can import the type from either location
export type { OpencodeProviderDefaults } from '../types';

/**
 * Parse raw assistantConfig into typed OpenCode defaults.
 * Defensive: invalid fields are silently dropped.
 */
export function parseOpencodeConfig(raw: Record<string, unknown>): OpencodeProviderDefaults {
  const result: OpencodeProviderDefaults = {};

  if (typeof raw.model === 'string') {
    result.model = raw.model;
  }

  if (typeof raw.hostname === 'string') {
    result.hostname = raw.hostname;
  }

  if (typeof raw.port === 'number' && Number.isInteger(raw.port) && raw.port > 0) {
    result.port = raw.port;
  }

  if (typeof raw.timeout === 'number' && Number.isInteger(raw.timeout) && raw.timeout > 0) {
    result.timeout = raw.timeout;
  }

  return result;
}
