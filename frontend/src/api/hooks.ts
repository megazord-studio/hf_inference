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
