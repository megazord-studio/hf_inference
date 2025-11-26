import { useMutation } from '@tanstack/react-query';
import { api } from './client';
import type {
  InferenceRequest,
  InferenceResponse,
} from '../types';

export function useInference() {
  return useMutation<InferenceResponse, unknown, InferenceRequest>({
    mutationFn: async (body) => {
      const { data } = await api.post('/inference', body);
      return data;
    },
  });
}

export function useTextGenerationStream() {
  return useMutation<{
    tokens: string[];
    metrics?: any;
    error?: string;
  },
  unknown,
  {
    model_id: string;
    prompt: string;
    params?: {
      max_new_tokens?: number;
      temperature?: number;
      top_p?: number;
    };
  }>({
    mutationFn: async ({ model_id, prompt, params }) => {
      return new Promise((resolve) => {
        const q = new URLSearchParams({ model_id, prompt });
        if (params?.max_new_tokens)
          q.set('max_new_tokens', String(params.max_new_tokens));
        if (params?.temperature != null) q.set('temperature', String(params.temperature));
        if (params?.top_p != null) q.set('top_p', String(params.top_p));
        const es = new EventSource(`/api/inference/stream?${q.toString()}`);
        const tokens: string[] = [];
        let metrics: any = undefined;
        es.addEventListener('token', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            if (data.text) tokens.push(data.text);
          } catch { /* ignore */ }
        });
        es.addEventListener('done', (e: MessageEvent) => {
          try { metrics = JSON.parse(e.data); } catch { /* ignore */ }
          es.close();
          resolve({ tokens, metrics });
        });
        es.addEventListener('error', (e: MessageEvent) => {
          es.close();
          resolve({ tokens, error: 'stream_error' });
        });
      });
    },
  });
}
