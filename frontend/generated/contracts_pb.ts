// Auto-generated from proto/contracts.proto. Do not edit manually.

export interface ErrorResponse {
  code: string | null;
  message: string | null;
  details?: Record<string, unknown> | null;
}

export interface TaskOutputMetadata {
  task: string | null;
  runtime_ms_model?: number | null;
  resolved_model_id?: string | null;
  backend?: string | null;
}

export interface InferenceResult {
  task_output: Record<string, unknown> | null;
  echo: Record<string, unknown> | null;
  info: Record<string, unknown> | null;
  metadata: TaskOutputMetadata | null;
}

export interface ModelMeta {
  id: string | null;
  model_id?: string | null;
  author?: string | null;
  gated?: boolean | null;
  private?: boolean | null;
  last_modified?: string | null;
  created_at?: string | null;
  likes?: number | null;
  trending_score?: number | null;
  downloads?: number | null;
  pipeline_tag?: string | null;
  library_name?: string | null;
  sha?: string | null;
  tags: string[];
  config?: Record<string, unknown> | null;
  card_data?: Record<string, unknown> | null;
  siblings: Record<string, unknown>[];
  fallback?: boolean | null;
}

export interface ModelSummary {
  id?: string | null;
  pipeline_tag?: string | null;
  tags: string[];
  gated?: boolean | null;
  likes?: number | null;
  downloads?: number | null;
  card_data?: Record<string, unknown> | null;
}

export interface InferenceResponsePayload {
  result?: InferenceResult | null;
  runtime_ms?: number | null;
  model_id?: string | null;
  model_meta?: ModelMeta | null;
  error?: ErrorResponse | null;
}

export interface InferenceErrorPayload {
  error: ErrorResponse | null;
}

export interface StreamingEvent {
  type?: string | null;
  version?: string | null;
  correlation_id?: string | null;
  task?: string | null;
  model_id?: string | null;
  payload?: Record<string, unknown> | null;
  error?: Record<string, unknown> | null;
}
