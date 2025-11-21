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
  gated?: string;
  likes?: number;
  downloads?: number;
  trendingScore?: number; // optional server-provided trending score
  createdAt?: string; // ISO timestamp for recency-based fallback
  cardData?: unknown;
}

export interface InferenceRequest {
  model_id: string;
  intent_id: string;
  input_type: InputType;
  inputs: Record<string, unknown>; // may contain text, image_base64, extra_args
}

export interface InferenceResponse {
  result: unknown;
  runtime_ms?: number;
  model_id?: string;
  model_meta?: Record<string, unknown>;
}

export interface CurlExampleResponse {
  command: string;
}

export interface RunRecord {
  id: string;
  createdAt: string;
  inputType: InputType;
  intent: Intent;
  model: ModelSummary;
  inputText: string;
  result?: unknown;
  curl?: string;
  requestInputs?: Record<string, unknown>; // debug: inputs sent to backend
  runtime_ms?: number;
  model_meta?: Record<string, unknown>; // enriched model metadata from backend
  streaming?: boolean;
  streamingTokens?: string[]; // incremental tokens for streaming runs
  streamingMetrics?: {
    tokens?: number;
    runtime_ms?: number;
    first_token_latency_ms?: number;
    tokens_per_second?: number;
  };
  streamingError?: string;
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
