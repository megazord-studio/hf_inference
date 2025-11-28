import { useMutation } from '@tanstack/react-query';
import { api } from './client';
import type { InferenceRequest } from '../types';
import type {
  InferenceResponsePayload,
  ErrorResponse,
} from '../generated/contracts_pb';

export function useInference() {
  return useMutation<InferenceResponsePayload, ErrorResponse | unknown, InferenceRequest>({
    mutationFn: async (body) => {
      try {
        const { data } = await api.post<InferenceResponsePayload>('/inference', body);
        if (data.error) {
          throw data.error;
        }
        return data;
      } catch (err: any) {
        const backendError = err?.response?.data?.error;
        if (backendError) {
          throw backendError as ErrorResponse;
        }
        throw err;
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
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('event: error')) {
            // Next data line contains the error
            continue;
          }
          if (line.startsWith('data: ')) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.type === 'token' && typeof payload.text === 'string') {
                tokens.push(payload.text);
              } else if (payload.type === 'done') {
                metrics = {
                  tokens: payload.tokens,
                  runtime_ms: payload.runtime_ms,
                  first_token_latency_ms: payload.first_token_latency_ms,
                  tokens_per_second: payload.tokens_per_second,
                };
              } else if (payload.message) {
                // Error event
                streamError = payload.message;
              }
            } catch {
              // Ignore malformed JSON
            }
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
