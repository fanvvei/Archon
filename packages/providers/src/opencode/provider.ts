/**
 * OpenCode SDK wrapper
 * Provides async generator interface for streaming OpenCode responses
 * Connects to a running OpenCode server via HTTP API.
 */
import { createOpencodeClient } from '@opencode-ai/sdk';
import type {
  IAgentProvider,
  SendQueryOptions,
  MessageChunk,
  ProviderCapabilities,
} from '../types';
import { parseOpencodeConfig } from './config';
import { OPENCODE_CAPABILITIES } from './capabilities';
import { createLogger } from '@archon/paths';

/** Lazy-initialized logger */
let cachedLog: ReturnType<typeof createLogger> | undefined;
function getLog(): ReturnType<typeof createLogger> {
  if (!cachedLog) cachedLog = createLogger('provider.opencode');
  return cachedLog;
}

// Singleton client cache (keyed by baseUrl)
let cachedClient: ReturnType<typeof createOpencodeClient> | null = null;
let cachedBaseUrl: string | null = null;

/** Reset singleton state. Exported for tests only. */
export function resetOpencodeSingleton(): void {
  cachedClient = null;
  cachedBaseUrl = null;
}

/**
 * Get or create OpenCode client instance.
 */
function getOpencodeClient(hostname: string, port: number): ReturnType<typeof createOpencodeClient> {
  const baseUrl = `http://${hostname}:${port}`;
  if (cachedClient && cachedBaseUrl === baseUrl) return cachedClient;

  cachedClient = createOpencodeClient({ baseUrl });
  cachedBaseUrl = baseUrl;
  return cachedClient;
}

// ─── Error Classification & Retry ────────────────────────────────────────

const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 2000;
const RATE_LIMIT_PATTERNS = ['rate limit', 'too many requests', '429', 'overloaded'];
const AUTH_PATTERNS = ['unauthorized', 'authentication', 'invalid token', '401', '403'];

function classifyOpencodeError(errorMessage: string): 'rate_limit' | 'auth' | 'crash' | 'unknown' {
  const m = errorMessage.toLowerCase();
  if (RATE_LIMIT_PATTERNS.some(p => m.includes(p))) return 'rate_limit';
  if (AUTH_PATTERNS.some(p => m.includes(p))) return 'auth';
  if (m.includes('econnrefused') || m.includes('econnreset') || m.includes('server error')) return 'crash';
  return 'unknown';
}

function classifyAndEnrichOpencodeError(
  error: Error
): { enrichedError: Error; errorClass: string; shouldRetry: boolean } {
  const errorClass = classifyOpencodeError(error.message);

  if (errorClass === 'auth') {
    const enrichedError = new Error(`OpenCode auth error: ${error.message}`);
    enrichedError.cause = error;
    return { enrichedError, errorClass, shouldRetry: false };
  }

  const enrichedError = new Error(`OpenCode ${errorClass}: ${error.message}`);
  enrichedError.cause = error;
  const shouldRetry = errorClass === 'rate_limit' || errorClass === 'crash';
  return { enrichedError, errorClass, shouldRetry };
}

// ─── Response Normalizer ───────────────────────────────────────────────────

/**
 * Normalize OpenCode prompt response into Archon MessageChunks.
 * OpenCode SDK returns a response object with data.parts array.
 * Each part has a type field (text, tool-invocation, etc.)
 */
function* normalizeOpencodeResponse(
  response: Record<string, unknown>,
  sessionId: string,
  hasOutputFormat: boolean
): Generator<MessageChunk> {
  const data = response.data as Record<string, unknown> | undefined;
  if (!data) {
    getLog().warn('opencode.response_missing_data');
    yield { type: 'result', sessionId, isError: true, errors: ['Empty response from OpenCode'] };
    return;
  }

  const info = data.info as Record<string, unknown> | undefined;
  const parts = data.parts as Array<Record<string, unknown>> | undefined;

  // Process parts
  let accumulatedText = '';
  if (Array.isArray(parts)) {
    for (const part of parts) {
      const partType = part.type as string;
      switch (partType) {
        case 'text':
          if (part.text) {
            const text = part.text as string;
            accumulatedText += text;
            yield { type: 'assistant', content: text };
          }
          break;

        case 'tool-invocation':
        case 'tool_invocation': {
          const toolName = (part.toolName ?? part.tool_name ?? 'unknown_tool') as string;
          const toolInput = (part.args ?? part.input ?? {}) as Record<string, unknown>;
          const toolCallId = part.toolCallId as string | undefined;
          yield { type: 'tool', toolName, toolInput, toolCallId };

          // If result is included
          if (part.result !== undefined) {
            const toolOutput = typeof part.result === 'string'
              ? part.result
              : JSON.stringify(part.result);
            yield { type: 'tool_result', toolName, toolOutput, toolCallId };
          }
          break;
        }

        case 'reasoning':
        case 'thinking':
          if (part.text || part.content) {
            yield { type: 'thinking', content: (part.text ?? part.content) as string };
          }
          break;

        default:
          getLog().debug({ partType }, 'opencode.unknown_part_type');
          break;
      }
    }
  }

  // Handle structured output
  let structuredOutput: unknown;
  if (hasOutputFormat && info) {
    const so = (info as Record<string, unknown>).structured_output;
    if (so !== undefined) {
      structuredOutput = so;
      getLog().debug('opencode.structured_output_received');
    } else if (accumulatedText) {
      // Fallback: try to parse accumulated text as JSON
      try {
        structuredOutput = JSON.parse(accumulatedText);
        getLog().debug('opencode.structured_output_parsed_from_text');
      } catch {
        getLog().warn(
          { outputPreview: accumulatedText.slice(0, 200) },
          'opencode.structured_output_not_json'
        );
        yield {
          type: 'system',
          content:
            '⚠️ Structured output requested but OpenCode returned non-JSON text. ' +
            'Downstream $nodeId.output.field references may not evaluate correctly.',
        };
      }
    }
  }

  // Check for errors
  const errorInfo = info
    ? (info as Record<string, unknown>).error as Record<string, unknown> | undefined
    : undefined;
  if (errorInfo?.name === 'StructuredOutputError') {
    yield {
      type: 'system',
      content: `⚠️ Structured output error: ${(errorInfo.message as string) ?? 'unknown'}`,
    };
  }

  // Result chunk
  yield {
    type: 'result',
    sessionId,
    ...(structuredOutput !== undefined ? { structuredOutput } : {}),
  };
}

// ─── OpenCode Provider ──────────────────────────────────────────────────────

/**
 * OpenCode AI agent provider.
 * Implements IAgentProvider with OpenCode SDK integration.
 * Connects to a running OpenCode server via HTTP API.
 */
export class OpencodeProvider implements IAgentProvider {
  private readonly retryBaseDelayMs: number;

  constructor(options?: { retryBaseDelayMs?: number }) {
    this.retryBaseDelayMs = options?.retryBaseDelayMs ?? RETRY_BASE_DELAY_MS;
  }

  getCapabilities(): ProviderCapabilities {
    return OPENCODE_CAPABILITIES;
  }

  async *sendQuery(
    prompt: string,
    cwd: string,
    resumeSessionId?: string,
    requestOptions?: SendQueryOptions
  ): AsyncGenerator<MessageChunk> {
    const assistantConfig = requestOptions?.assistantConfig ?? {};
    const config = parseOpencodeConfig(assistantConfig);

    const hostname = config.hostname ?? '127.0.0.1';
    const port = config.port ?? 4096;

    // 1. Get client
    const client = getOpencodeClient(hostname, port);

    if (requestOptions?.abortSignal?.aborted) {
      throw new Error('Query aborted');
    }

    // 2. Resolve session ID
    let sessionId = resumeSessionId;
    let sessionResumeFailed = false;

    if (!sessionId) {
      getLog().debug({ cwd }, 'creating_new_session');
      try {
        const session = await client.session.create({
          body: { title: `Archon session` },
        });
        const sessionData = session.data ?? session;
        sessionId = (sessionData as Record<string, unknown>).id as string;
        getLog().debug({ sessionId }, 'session_created');
      } catch (error) {
        const err = error as Error;
        throw new Error(`OpenCode session creation failed: ${err.message}`);
      }
    } else {
      // Verify the session exists by trying to get it
      getLog().debug({ sessionId: resumeSessionId }, 'resuming_session');
      try {
        await client.session.get({ path: { id: sessionId } });
      } catch {
        getLog().error({ sessionId: resumeSessionId }, 'resume_session_failed');
        // Create a new session as fallback
        try {
          const session = await client.session.create({
            body: { title: `Archon session` },
          });
          const sessionData = session.data ?? session;
          sessionId = (sessionData as Record<string, unknown>).id as string;
          sessionResumeFailed = true;
        } catch (createError) {
          const err = createError as Error;
          throw new Error(`OpenCode session creation failed: ${err.message}`);
        }
      }
    }

    if (sessionResumeFailed) {
      yield {
        type: 'system',
        content: '⚠️ Could not resume previous session. Starting fresh conversation.',
      };
    }

    // 3. Build prompt body
    const hasOutputFormat = !!(
      requestOptions?.outputFormat ?? requestOptions?.nodeConfig?.output_format
    );

    const promptBody: Record<string, unknown> = {
      parts: [{ type: 'text', text: prompt }],
    };

    // Model configuration
    const model = requestOptions?.model ?? config.model;
    if (model) {
      // OpenCode model format: { providerID: "provider", modelID: "model-id" }
      if (model.includes('/')) {
        const [providerID, ...rest] = model.split('/');
        promptBody.model = { providerID, modelID: rest.join('/') };
      } else {
        promptBody.model = { modelID: model };
      }
    }

    // Structured output format
    if (requestOptions?.outputFormat) {
      promptBody.format = {
        type: 'json_schema',
        schema: requestOptions.outputFormat.schema,
      };
    } else if (requestOptions?.nodeConfig?.output_format) {
      promptBody.format = {
        type: 'json_schema',
        schema: requestOptions.nodeConfig.output_format,
      };
    }

    // 4. Send prompt with retry
    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (requestOptions?.abortSignal?.aborted) {
        throw new Error('Query aborted');
      }

      if (attempt > 0) {
        const delayMs = this.retryBaseDelayMs * Math.pow(2, attempt - 1);
        getLog().info({ attempt, delayMs }, 'retrying_query');
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }

      try {
        // Apply timeout control
        const timeoutMs = config.timeout ?? 5000;
        const timeoutController = new AbortController();
        const existingSignal = requestOptions?.abortSignal;

        // Combine user abort signal with timeout
        if (existingSignal?.aborted) {
          throw new Error('Query aborted');
        }
        const abortHandler = existingSignal
          ? () => timeoutController.abort()
          : undefined;
        if (existingSignal && abortHandler) {
          existingSignal.addEventListener('abort', abortHandler, { once: true });
        }

        const timer = setTimeout(() => timeoutController.abort(), timeoutMs);

        let response;
        try {
          response = await client.session.prompt({
            path: { id: sessionId! },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            body: promptBody as any,
          });
        } finally {
          clearTimeout(timer);
          if (existingSignal && abortHandler) {
            existingSignal.removeEventListener('abort', abortHandler);
          }
        }

        // 5. Normalize response
        yield* normalizeOpencodeResponse(
          response as unknown as Record<string, unknown>,
          sessionId!,
          hasOutputFormat
        );
        return;
      } catch (error) {
        const err = error as Error;

        if (requestOptions?.abortSignal?.aborted) {
          throw new Error('Query aborted');
        }

        const { enrichedError, errorClass, shouldRetry } = classifyAndEnrichOpencodeError(err);

        getLog().error(
          { err, errorClass, attempt, maxRetries: MAX_RETRIES },
          'query_error'
        );

        if (!shouldRetry || attempt >= MAX_RETRIES) {
          throw enrichedError;
        }

        lastError = enrichedError;
      }
    }

    throw lastError ?? new Error('OpenCode query failed after retries');
  }

  getType(): string {
    return 'opencode';
  }
}
