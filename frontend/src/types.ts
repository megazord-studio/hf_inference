import type { InferenceResponsePayload, ErrorResponse, ModelMeta, TaskOutputMetadata, InferenceResult } from '../generated/contracts_pb';

export type InputType = 'text' | 'image' | 'audio' | 'video' | 'document' | 'multimodal';

export interface Intent {
  id: string;
  label: string;
  description: string;
  input_types: InputType[];
  hf_tasks: string[];
}

export interface ModelSummary {
  id: string;
  pipeline_tag?: string;
  tags?: string[];
  gated?: boolean | string;
  likes?: number;
  downloads?: number;
  trendingScore?: number; // optional server-provided trending score
  createdAt?: string; // ISO timestamp for recency-based fallback
  cardData?: unknown;
}

export interface InferenceRequest {
  model_id: string;
  intent_id?: string;
  input_type: InputType;
  task?: string;
  inputs: Record<string, unknown>;
  options?: Record<string, unknown>;
}

// Goal-oriented taxonomy additions (non-breaking)
export type OutputType = 'text' | 'image' | 'embedding' | 'mask' | 'boxes' | 'depth' | 'multimodal' | 'video' | 'audio' | '3d' | 'labels' | 'scores' | 'points' | 'segments' | 'depth-map' | 'image+text';

export interface GoalSubcategory {
  id: string;               // e.g. 'summarize'
  label: string;            // Human label
  description: string;      // Short guidance
  tasks: string[];          // HF task identifiers that fulfill this goal
  input_types: InputType[]; // Modalities consumed
  output_types: OutputType[]; // Modalities produced
  examples?: string[];      // optional sample prompts
}

export interface GoalCategory {
  id: string;               // e.g. 'text-understanding'
  label: string;            // Human display label
  subcategories: GoalSubcategory[];
}

export interface GoalPreferences {
  qualityBias?: boolean;      // prefer higher likes
  speedBias?: boolean;        // prefer higher downloads
  openSourceOnly?: boolean;   // exclude gated models
  popularityFloorBoost?: number; // dynamic min likes boost
}

export interface GoalSelectionState {
  selectedGoalCategory?: string | null;
  selectedGoalSubcategory?: string | null;
  preferences: GoalPreferences;
}

export interface StreamingErrorEvent {
  type: 'error';
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface StreamingTokenEvent {
  type: 'token';
  index: number;
  text: string;
}

export interface StreamingProgressEvent {
  type: 'progress';
  [key: string]: unknown;
}

export interface StreamingDoneEvent {
  type: 'done';
  [key: string]: unknown;
}

export interface RunRecord {
  id: string;
  createdAt: string;
  inputType: InputType;
  intent: Intent;
  model: ModelSummary;
  inputText: string;
  result?: InferenceResult['task_output'];
  curl?: string;
  requestInputs?: Record<string, unknown>;
  runtime_ms?: number;
  model_meta?: ModelMeta;
  streaming?: boolean;
  streamingTokens?: string[];
  streamingMetrics?: {
    tokens?: number;
    runtime_ms?: number;
    first_token_latency_ms?: number;
    tokens_per_second?: number;
  };
  streamingError?: string;
  error?: ErrorResponse;
}
