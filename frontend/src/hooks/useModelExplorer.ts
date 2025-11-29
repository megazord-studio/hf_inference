import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { useInference } from '../api/hooks';
import { useTextGenerationStream, type TextGenerationStreamResult } from '../api/hooks';
import type { InputType, OutputType, ModelSummary, Intent, RunRecord, InferenceRequest } from '../types';
import type { ErrorResponse } from '../../generated/contracts_pb';
import { taskModalities } from '../constants/taskModalities';
import { tasksInfo } from '../constants/tasksCatalog';
import Fuse from 'fuse.js';
import { BackendError } from '../api/hooks';

const ALL_INPUT_MODALITIES: InputType[] = ['text','image','audio','video','document'];
const ALL_OUTPUT_MODALITIES: OutputType[] = ['text','image','audio','video','embedding','boxes','mask','depth','3d'];

const randomId = () => `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;

const buildRunPlaceholderIntent = (): Intent => ({ id: 'none', label: 'None', description: '', input_types: [], hf_tasks: [] });

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

// Minimal modality + run model explorer state container
export function useModelExplorer() {
  // Modality chips
  const [selectedInputModalities, setSelectedInputModalities] = useState<InputType[]>([]);
  const [selectedOutputModalities, setSelectedOutputModalities] = useState<OutputType[]>([]);

  // Models & intents
  const modelsQuery = useQuery<ModelSummary[]>({
     queryKey: ['preloaded-models'],
     queryFn: async () => (await api.get<ModelSummary[]>('/models/preloaded', { params: { limit: 50000 } })).data,
     staleTime: 60*60*1000,
   });
  const intentsQuery = useQuery<Intent[]>({
    queryKey: ['intents'],
    queryFn: async () => (await api.get<Intent[]>('/intents')).data,
    staleTime: 60*60*1000,
  });
  const allModels = useMemo<ModelSummary[]>(() => modelsQuery.data ?? [], [modelsQuery.data]);
  const allIntents = useMemo<Intent[]>(() => intentsQuery.data ?? [], [intentsQuery.data]);

  // Selected model
  const [selectedModel, setSelectedModel] = useState<ModelSummary | null>(null);
  const selectModel = (m: ModelSummary) => setSelectedModel(m);

  // Derive tasks for selection (OR semantics)
  const tasksFromIO = useMemo<string[]>(() => {
    if (!selectedInputModalities.length && !selectedOutputModalities.length) return [];
    const multiInputs = selectedInputModalities.length > 1;
    const matches: string[] = [];
    for (const [task, tm] of Object.entries(taskModalities)) {
      if (multiInputs && !tm.multiInputSupport) continue; // cannot satisfy multi-input combo
      const inputOk = selectedInputModalities.every(i => tm.input.includes(i));
      const outputOk = selectedOutputModalities.every(o => tm.output.includes(o));
      if (inputOk && outputOk) matches.push(task);
    }
    return matches;
  }, [selectedInputModalities, selectedOutputModalities]);

  // Filter models by pipeline_tag inside tasksFromIO
  const filteredModels = useMemo<ModelSummary[]>(() => {
    if (!tasksFromIO.length) return allModels;
    return allModels.filter(m => m.pipeline_tag && tasksFromIO.includes(m.pipeline_tag));
  }, [allModels, tasksFromIO]);

  // Pagination
  const [modelChunkSize, setModelChunkSize] = useState(60);
  const [sortBy, setSortBy] = useState<'downloads' | 'likes' | 'trending' | 'name'>('trending');
  const sortedModels = useMemo(() => {
    const arr = filteredModels.slice();
    arr.sort((a,b) => {
      if (sortBy === 'downloads') return (b.downloads ?? 0) - (a.downloads ?? 0);
      if (sortBy === 'likes') return (b.likes ?? 0) - (a.likes ?? 0);
      if (sortBy === 'trending') return (b.trendingScore ?? -1) - (a.trendingScore ?? -1); // no calculation, rely solely on provided score
      return a.id.localeCompare(b.id);
    });
    return arr;
  }, [filteredModels, sortBy]);
  const [searchQuery, setSearchQuery] = useState('');
  const fuse = useMemo(() => new Fuse(sortedModels, { keys: ['id','pipeline_tag','tags'], threshold: 0.35, ignoreLocation: true }), [sortedModels]);
  const searchedModels = useMemo(() => {
    if (!searchQuery.trim()) return sortedModels;
    return fuse.search(searchQuery.trim()).map(r => r.item);
  }, [sortedModels, fuse, searchQuery]);
  const visibleModels = useMemo(() => searchedModels.slice(0, modelChunkSize), [searchedModels, modelChunkSize]);
  const loadMoreModels = useCallback(() => setModelChunkSize(s => Math.min(filteredModels.length, s + 60)), [filteredModels.length]);

  // Availability helpers
  const isInputComboViable = useCallback((inputs: InputType[], outputs: OutputType[]) => {
    for (const tm of Object.values(taskModalities)) {
      if (inputs.length > 1 && !tm.multiInputSupport) continue;
      if (!inputs.every(i => tm.input.includes(i))) continue;
      if (!outputs.every(o => tm.output.includes(o))) continue;
      return true;
    }
    return false;
  }, []);
  const isOutputComboViable = useCallback((inputs: InputType[], outputs: OutputType[]) => {
    for (const tm of Object.values(taskModalities)) {
      if (inputs.length > 1 && !tm.multiInputSupport) continue;
      if (!inputs.every(i => tm.input.includes(i))) continue;
      if (!outputs.every(o => tm.output.includes(o))) continue;
      return true;
    }
    return false;
  }, []);

  const availableInputModalities = useMemo(() => {
    if (!selectedInputModalities.length && !selectedOutputModalities.length) return ALL_INPUT_MODALITIES;
    const candidates: InputType[] = [];
    for (const m of ALL_INPUT_MODALITIES) {
      if (selectedInputModalities.includes(m)) { candidates.push(m); continue; }
      const next = [...selectedInputModalities, m];
      if (isInputComboViable(next, selectedOutputModalities)) candidates.push(m);
    }
    return candidates;
  }, [selectedInputModalities, selectedOutputModalities, isInputComboViable]);

  const availableOutputModalities = useMemo(() => {
    if (!selectedInputModalities.length && !selectedOutputModalities.length) return ALL_OUTPUT_MODALITIES;
    const candidates: OutputType[] = [];
    for (const o of ALL_OUTPUT_MODALITIES) {
      if (selectedOutputModalities.includes(o)) { candidates.push(o); continue; }
      const nextOut = [...selectedOutputModalities, o];
      if (isOutputComboViable(selectedInputModalities, nextOut)) candidates.push(o);
    }
    return candidates;
  }, [selectedInputModalities, selectedOutputModalities, isOutputComboViable]);

  const toggleInputModality = useCallback((m: InputType) => setSelectedInputModalities(prev => prev.includes(m) ? prev.filter(x => x!==m) : [...prev, m]), []);
  const toggleOutputModality = useCallback((m: OutputType) => setSelectedOutputModalities(prev => prev.includes(m) ? prev.filter(x => x!==m) : [...prev, m]), []);
  const clearFilters = useCallback(() => { setSelectedInputModalities([]); setSelectedOutputModalities([]); setModelChunkSize(60); }, []);

  // Model tasks (from pipeline_tag + tag intersection with known tasks)
  const modelTasks = useMemo<string[]>(() => {
    if (!selectedModel) return [];
    const set = new Set<string>();
    if (selectedModel.pipeline_tag) set.add(selectedModel.pipeline_tag);
    if (selectedModel.tags) {
      const knownTasks = new Set(Object.keys(taskModalities));
      selectedModel.tags.forEach(t => { if (knownTasks.has(t)) set.add(t); });
    }
    return Array.from(set);
  }, [selectedModel]);

  // Candidate intents (those whose hf_tasks intersect model tasks OR pipeline tag)
  const candidateIntents = useMemo<Intent[]>(() => {
    if (!selectedModel) return [];
    return allIntents.filter(it => it.hf_tasks.some(t => t === selectedModel.pipeline_tag || modelTasks.includes(t)));
  }, [selectedModel, allIntents, modelTasks]);

  // Run intent selection
  const [runIntentId, setRunIntentId] = useState<string | null>(null);
  const runIntent = useMemo(() => runIntentId ? allIntents.find(i => i.id === runIntentId) : undefined, [runIntentId, allIntents]);
  // Ensure default intent when model selected
  useEffect(() => {
    if (selectedModel && !runIntentId && candidateIntents.length) {
      setRunIntentId(candidateIntents[0].id);
    }
  }, [selectedModel, runIntentId, candidateIntents]);

   // Selected task
   const [selectedTask, setSelectedTask] = useState<string | null>(null);
   useEffect(() => {
     if (selectedModel && !selectedTask && modelTasks.length) {
       setSelectedTask(modelTasks[0]);
     }
   }, [selectedModel, selectedTask, modelTasks]);

  // Input states
  const [textInput, setTextInput] = useState('');
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [audioB64, setAudioB64] = useState<string | null>(null);
  const [videoB64, setVideoB64] = useState<string | null>(null);
  const [documentB64, setDocumentB64] = useState<string | null>(null);
  const handleGenericFile = useCallback((file: File, setter: (value: string) => void) => new Promise<void>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'));
    reader.onload = () => {
      const r = reader.result;
      if (typeof r === 'string') {
        setter(r);
        resolve();
        return;
      }
      reject(new Error('Expected base64 string result'));
    };
    reader.readAsDataURL(file);
  }), []);
  const handleImageFile = useCallback((file: File) => handleGenericFile(file, v=>setImageB64(v)), [handleGenericFile]);
  const handleAudioFile = useCallback((file: File) => handleGenericFile(file, v=>setAudioB64(v)), [handleGenericFile]);
  const handleVideoFile = useCallback((file: File) => handleGenericFile(file, v=>setVideoB64(v)), [handleGenericFile]);
  const handleDocumentFile = useCallback((file: File) => handleGenericFile(file, v=>setDocumentB64(v)), [handleGenericFile]);
  const [extraArgsJson, setExtraArgsJson] = useState<string>('{}');
  const [extraArgsError, setExtraArgsError] = useState<string | null>(null);
  const parseExtraArgs = useCallback((): Record<string, unknown> => {
     if (!extraArgsJson.trim()) {
       setExtraArgsError(null);
       return {};
     }
     try {
       const parsed: unknown = JSON.parse(extraArgsJson);
       if (isRecord(parsed)) {
         setExtraArgsError(null);
         return { ...parsed };
       }
       setExtraArgsError('Must be JSON object');
       return {};
     } catch (error) {
       setExtraArgsError(error instanceof Error ? error.message : 'Invalid JSON');
       return {};
     }
  }, [extraArgsJson]);

  // Determine required modalities from taskModalities dynamically
  const effectiveTasks = useMemo(() => (selectedTask ? [selectedTask] : (runIntent ? runIntent.hf_tasks : [])), [selectedTask, runIntent]);
  const requiredInputsSet = useMemo(() => {
    const s = new Set<InputType>();
    effectiveTasks.forEach(t => { const tm = taskModalities[t]; if (tm) tm.input.forEach(i => s.add(i)); });
    return s;
  }, [effectiveTasks]);
  const requiresImage = requiredInputsSet.has('image');
  const requiresText = requiredInputsSet.has('text');
  const requiresAudio = requiredInputsSet.has('audio');
  const requiresVideo = requiredInputsSet.has('video');
  const requiresDocument = requiredInputsSet.has('document');

  // Selected task info
  const selectedTaskInfo = selectedTask ? tasksInfo[selectedTask] : undefined;
  const inputPattern = (selectedTaskInfo?.input || '').toLowerCase();
  const combinationRequired = /image\+text|text\+image/.test(inputPattern);

  // Runs history
  const [runs, setRuns] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [latestCurl, setLatestCurl] = useState<string | null>(null);
  const clearRuns = useCallback(() => {
    setRuns([]);
    setSelectedRunId(null);
  }, []);

  // Inference hooks
  const inference = useInference();
  const streamGen = useTextGenerationStream();
  const [streamEnabled, setStreamEnabled] = useState<boolean>(true);

  // Determine primary input_type for request (fallback to text)
  const primaryInputType: InputType = requiresImage ? 'image' : requiresAudio ? 'audio' : requiresVideo ? 'video' : requiresDocument ? 'document' : 'text';

  // Evaluate readiness: explicit combo tasks need all listed combo inputs; otherwise at least one among allowed modalities
  const provided = useMemo<Record<InputType, boolean>>(() => ({
     text: !!textInput.trim(),
     image: !!imageB64,
     audio: !!audioB64,
     video: !!videoB64,
     document: !!documentB64,
     multimodal: false,
  }), [textInput, imageB64, audioB64, videoB64, documentB64]);
  const allowedModalities = Array.from(requiredInputsSet);
  const hasAtLeastOne = allowedModalities.some(m => provided[m]);
  const allComboSatisfied = combinationRequired ? (provided['text'] && provided['image']) : true;
  const canRun = !!(selectedModel && (runIntent || selectedTask) && allComboSatisfied && (combinationRequired ? true : hasAtLeastOne));

  const buildInputs = useCallback(() => {
    const inputs: Record<string, unknown> = {};
    // For combo tasks include only combo modalities if present; else include any provided among allowed set
    if (combinationRequired) {
      if (provided.text) inputs.text = textInput.trim();
      if (provided.image) inputs.image_base64 = imageB64;
    } else {
      if (requiresText && provided.text) inputs.text = textInput.trim();
      if (requiresImage && provided.image) inputs.image_base64 = imageB64;
      if (requiresAudio && provided.audio) inputs.audio_base64 = audioB64;
      if (requiresVideo && provided.video) inputs.video_base64 = videoB64;
      if (requiresDocument && provided.document) inputs.document_base64 = documentB64;
    }
    const extra = parseExtraArgs();
    if (selectedTask) (extra)._task = selectedTask;
    if (Object.keys(extra).length) inputs.extra_args = extra;
    return inputs;
  }, [combinationRequired, provided, textInput, imageB64, audioB64, videoB64, documentB64, requiresText, requiresImage, requiresAudio, requiresVideo, requiresDocument, parseExtraArgs, selectedTask]);

  const buildInferenceRequest = useCallback((): InferenceRequest | null => {
     if (!canRun || !selectedModel) return null;
     const inputs = buildInputs();
     if (!Object.keys(inputs).length) {
       return null;
     }
     return {
       model_id: selectedModel.id,
       intent_id: runIntent?.id,
       input_type: primaryInputType,
       inputs,
       task: selectedTask || undefined,
       options: parseExtraArgs(),
     };
   }, [buildInputs, canRun, parseExtraArgs, primaryInputType, runIntent?.id, selectedModel, selectedTask]);

   const newRunFromError = useCallback((err: ErrorResponse | undefined): RunRecord => ({
     id: randomId(),
     createdAt: new Date().toISOString(),
     inputType: primaryInputType,
     intent: runIntent || buildRunPlaceholderIntent(),
     model: selectedModel!,
     inputText: textInput,
     error: err,
   }), [primaryInputType, runIntent, selectedModel, textInput]);

   const runModel = useCallback(() => {
    if (!canRun || !selectedModel) return;
    const inputs = buildInputs();
    const isTextGen = selectedTask === 'text-generation' && !requiresImage && !requiresAudio && !requiresVideo && !requiresDocument;
    if (isTextGen && streamEnabled && textInput.trim()) {
      const extraArgs = (inputs.extra_args as Record<string, unknown> | undefined) ?? {};
      void streamGen.mutateAsync({
         model_id: selectedModel.id,
         prompt: textInput.trim(),
         params: {
           max_new_tokens: typeof extraArgs.max_new_tokens === 'number' ? extraArgs.max_new_tokens : 50,
           temperature: typeof extraArgs.temperature === 'number' ? extraArgs.temperature : 1.0,
           top_p: typeof extraArgs.top_p === 'number' ? extraArgs.top_p : 1.0,
         },
       })
         .then((data: TextGenerationStreamResult) => {
          const id = randomId();
          const newRun: RunRecord = {
            id,
            createdAt: new Date().toISOString(),
            inputType: primaryInputType,
            intent: runIntent || buildRunPlaceholderIntent(),
            model: selectedModel,
            inputText: textInput,
            streaming: true,
            streamingTokens: data.tokens,
            streamingMetrics: data.metrics,
            streamingError: data.error,
          };
          setRuns(prev => [...prev, newRun]);
          setSelectedRunId(id);
        })
        .catch((error: unknown) => {
          const backendPayload = error instanceof BackendError ? error.payload : undefined;
          const id = randomId();
          const newRun: RunRecord = {
            id,
            createdAt: new Date().toISOString(),
            inputType: primaryInputType,
            intent: runIntent || buildRunPlaceholderIntent(),
            model: selectedModel,
            inputText: textInput,
            error: backendPayload,
          };
          setRuns(prev => [...prev, newRun]);
          setSelectedRunId(id);
        });
      return;
    }
    const body = buildInferenceRequest();
    if (!body) return;
    void inference.mutateAsync(body)
      .then((data) => {
        if (!data.result) {
          setRuns(prev => [...prev, newRunFromError(data.error ?? undefined)]);
          return;
        }
        const id = randomId();
        const newRun: RunRecord = {
          id,
          createdAt: new Date().toISOString(),
          inputType: primaryInputType,
          intent: runIntent || buildRunPlaceholderIntent(),
          model: selectedModel,
          inputText: textInput,
          result: data.result.task_output,
          runtime_ms: data.runtime_ms ?? undefined,
          requestInputs: body.inputs,
          model_meta: data.model_meta ?? undefined,
          error: data.error ?? undefined,
        };
        setRuns(prev => [...prev, newRun]);
        setSelectedRunId(id);
      })
      .catch((err: unknown) => {
        const errorResponse = err instanceof BackendError ? err.payload : undefined;
        const id = randomId();
        const newRun: RunRecord = {
          id,
          createdAt: new Date().toISOString(),
          inputType: primaryInputType,
          intent: runIntent || buildRunPlaceholderIntent(),
          model: selectedModel,
          inputText: textInput,
          error: errorResponse,
        };
        setRuns(prev => [...prev, newRun]);
        setSelectedRunId(id);
      });
  }, [buildInputs, buildInferenceRequest, canRun, inference, primaryInputType, requiresAudio, requiresDocument, requiresImage, requiresVideo, runIntent, selectedModel, selectedTask, streamEnabled, streamGen, textInput, newRunFromError]);

  const showCurl = () => {
    // No-op now that curl is generated client-side via useEffect; keep for button wiring.
  };

  const selectRun = (id: string) => {
     setSelectedRunId(id);
     const r = runs.find(rr => rr.id === id);
     if (r) {
       setSelectedModel(r.model);
       setTextInput(r.inputText);
       setRunIntentId(r.intent.id);
     }
   };

  // Live-update latestCurl when inputs change, using placeholders for file inputs
  useEffect(() => {
    const body = buildInferenceRequest();
    if (!body) { setLatestCurl(null); return; }
    const sanitized = { ...body, inputs: { ...body.inputs } } as InferenceRequest;
    if (sanitized.inputs.image_base64) sanitized.inputs.image_base64 = '@IMAGE_PATH';
    if (sanitized.inputs.audio_base64) sanitized.inputs.audio_base64 = '@AUDIO_PATH';
    if (sanitized.inputs.video_base64) sanitized.inputs.video_base64 = '@VIDEO_PATH';
    if (sanitized.inputs.document_base64) sanitized.inputs.document_base64 = '@DOCUMENT_PATH';
    try {
      const json = JSON.stringify(sanitized, null, 0);
      const cmd = "curl -s -X POST 'http://localhost:8000/api/inference' " +
        "-H 'Content-Type: application/json' " +
        "-d '" + json.replace(/'/g, "'\\''") + "'";
      setLatestCurl(cmd);
    } catch {
      setLatestCurl(null);
    }
  }, [
    buildInferenceRequest,
    documentB64,
    imageB64,
    audioB64,
    videoB64,
    canRun,
    extraArgsJson,
    selectedModel?.id,
    runIntent?.id,
    primaryInputType,
    selectedTask,
    textInput,
  ]);

  // Ensure selectedTask aligns with current model's tasks when model changes
  useEffect(() => {
    if (selectedModel && selectedTask && !modelTasks.includes(selectedTask)) {
      setSelectedTask(modelTasks[0] || null);
    }
    // Clear modality buffers that are no longer required after task/model change
    if (selectedModel) {
      if (!requiresImage && imageB64) setImageB64(null);
      if (!requiresAudio && audioB64) setAudioB64(null);
      if (!requiresVideo && videoB64) setVideoB64(null);
      if (!requiresDocument && documentB64) setDocumentB64(null);
    }
  }, [selectedModel, selectedTask, modelTasks, requiresImage, requiresAudio, requiresVideo, requiresDocument, imageB64, audioB64, videoB64, documentB64]);

  // Gated overlay: lazily fetched per visible model ids
  const [gatedById, setGatedById] = useState<Record<string, boolean | undefined>>({});
  const enrichedIdsRef = useRef<Set<string>>(new Set());

  // When visible models change, enrich gated flag for those not yet known
  useEffect(() => {
    const ids = visibleModels.map(m => m.id);
    const toFetch = ids.filter(id => !enrichedIdsRef.current.has(id));
    if (!toFetch.length) return;
    const batch = toFetch.slice(0, 96); // keep requests small
    void (async () => {
       try {
         const { data } = await api.post<Array<{ id: string; gated?: boolean }>>('/models/enrich', batch);
         const next: Record<string, boolean | undefined> = {};
         for (const m of data) {
           next[m.id] = m.gated;
           enrichedIdsRef.current.add(m.id);
         }
         setGatedById(prev => ({ ...prev, ...next }));
       } catch {
          // ignore; we can retry on next visibility change
        }
     })();
   }, [visibleModels]);


  return {
    // Modality state
    selectedInputModalities, toggleInputModality,
    selectedOutputModalities, toggleOutputModality,
    ALL_INPUT_MODALITIES, ALL_OUTPUT_MODALITIES,
    availableInputModalities, availableOutputModalities,
    // Models
    filteredModels: sortedModels, visibleModels, loadMoreModels, modelChunkSize,
    selectedModel, selectModel,
    // Intents & tasks
    candidateIntents, runIntentId, setRunIntentId, runIntent,
    modelTasks, selectedTask, setSelectedTask,
    // Input handling
    textInput, setTextInput,
    requiresImage, requiresText, requiresAudio, requiresVideo, requiresDocument,
    imageB64, handleImageFile,
    audioB64, handleAudioFile,
    videoB64, handleVideoFile,
    documentB64, handleDocumentFile,
    extraArgsJson, setExtraArgsJson,
    extraArgsError,
    // Execution
    canRun, runModel, showCurl,
    inferencePending: inference.isPending || streamGen.isPending,
    curlPending: false,
    // Runs history
    runs, selectedRunId, selectRun,
    clearRuns,
    latestCurl,
    // Misc
    tasksFromIO,
    clearFilters,
    sortBy, setSortBy,
    isLoadingModels: modelsQuery.isLoading, isLoadingIntents: intentsQuery.isLoading,
    searchQuery, setSearchQuery,
    streamEnabled, setStreamEnabled,
    gatedById,
  };
}

export type UseModelExplorerReturn = ReturnType<typeof useModelExplorer>;
