import { useMutation } from '@tanstack/react-query';
import { api } from './client';
import type {
  InferenceRequest,
  InferenceResponse,
  CurlExampleResponse,
} from '../types';

export function useInference() {
  return useMutation<InferenceResponse, unknown, InferenceRequest>({
    mutationFn: async (body) => {
      const { data } = await api.post('/inference', body);
      return data;
    },
  });
}

export function useCurlExample() {
  return useMutation<CurlExampleResponse, unknown, InferenceRequest>({
    mutationFn: async (body) => {
      const { data } = await api.post('/curl-example', body);
      return data;
    },
  });
}

