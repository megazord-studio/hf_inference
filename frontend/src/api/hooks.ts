import { useMutation } from '@tanstack/react-query';
import type { AxiosError } from 'axios';
import { api } from './client';
import type {
  InferenceRequest,
  StreamingDoneEvent,
  StreamingErrorEvent,
  StreamingTokenEvent,
} from '../types';
import type {
  InferenceResponsePayload,
  ErrorResponse,
} from '../../generated/contracts_pb';

export class BackendError extends Error {
  constructor(public payload: ErrorResponse) {
    super(payload.message ?? 'Inference request failed');
    this.name = 'BackendError';
  }
}

export const toBackendError = (payload: ErrorResponse) => new BackendError(payload);

const isAxiosErrorWithBackendPayload = (
  error: unknown,
): error is AxiosError<{ error?: ErrorResponse }> =>
  typeof error === 'object' && error !== null && (error as AxiosError).isAxiosError === true;

export const extractBackendError = (error: unknown): ErrorResponse | null => {
  if (!isAxiosErrorWithBackendPayload(error)) {
    return null;
  }
  return error.response?.data?.error ?? null;
};

export function useInference() {
  return useMutation<InferenceResponsePayload, Error, InferenceRequest>({
    mutationFn: async (body) => {
      try {
        const { data } = await api.post<InferenceResponsePayload>('/inference', body);
        if (data.error) {
          throw toBackendError(data.error);
        }
        return data;
      } catch (error: unknown) {
        const backendError = extractBackendError(error);
        if (backendError) {
          throw toBackendError(backendError);
        }
        throw error instanceof Error ? error : new Error('Inference request failed');
      }
    },
  });
}

/** Streaming metrics returned after text generation completes */
export interface StreamingMetrics {
  tokens?: number;
  runtime_ms?: number;
  first_token_latency_ms?: number;
  tokens_per_second?: number;
}

/** Result from useTextGenerationStream mutation */
export interface TextGenerationStreamResult {
  tokens: string[];
  metrics?: StreamingMetrics;
  error?: string;
}

/** Input parameters for text generation streaming */
export interface TextGenerationStreamInput {
  model_id: string;
  prompt: string;
  params?: {
    max_new_tokens?: number;
    temperature?: number;
    top_p?: number;
  };
}

/**
 * Hook for streaming text generation via SSE.
 * Connects to /api/inference/stream and collects tokens until done.
 */
const isTokenEvent = (payload: unknown): payload is StreamingTokenEvent =>
  typeof payload === 'object' && payload !== null && (payload as StreamingTokenEvent).type === 'token';

const isDoneEvent = (payload: unknown): payload is StreamingDoneEvent =>
  typeof payload === 'object' && payload !== null && (payload as StreamingDoneEvent).type === 'done';

const isErrorEvent = (payload: unknown): payload is StreamingErrorEvent =>
  typeof payload === 'object' && payload !== null && (payload as StreamingErrorEvent).type === 'error';

export function useTextGenerationStream() {
  return useMutation<TextGenerationStreamResult, Error, TextGenerationStreamInput>({
    mutationFn: async ({ model_id, prompt, params }) => {
      const tokens: string[] = [];
      let metrics: StreamingMetrics | undefined;
      let streamError: string | undefined;

      const queryParams = new URLSearchParams({
        model_id,
        prompt,
        max_new_tokens: String(params?.max_new_tokens ?? 50),
        temperature: String(params?.temperature ?? 1.0),
        top_p: String(params?.top_p ?? 1.0),
      });

      const response = await fetch(`/api/inference/stream?${queryParams.toString()}`);
      if (!response.ok) {
        throw new Error(`Stream request failed: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const dataLine = line.slice(6);
          if (!dataLine.trim()) {
            continue;
          }
          try {
            const payload: unknown = JSON.parse(dataLine) as unknown;
            if (isTokenEvent(payload)) {
              tokens.push(payload.text);
            } else if (isDoneEvent(payload)) {
              metrics = {
                tokens: payload.tokens,
                runtime_ms: payload.runtime_ms,
                first_token_latency_ms: payload.first_token_latency_ms,
                tokens_per_second: payload.tokens_per_second,
              };
            } else if (isErrorEvent(payload)) {
              streamError = payload.message;
            }
          } catch (parseError) {
            console.warn('Malformed SSE payload', parseError);
          }
        }
      }

      if (streamError) {
        return { tokens, metrics, error: streamError };
      }

      return { tokens, metrics };
    },
  });
}
