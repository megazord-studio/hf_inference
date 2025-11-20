import { useState, useMemo, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { useInference, useCurlExample } from '../api/hooks';
import type { InputType, OutputType, ModelSummary, Intent, RunRecord, InferenceRequest } from '../types';
import { taskModalities } from '../constants/taskModalities';
import { tasksInfo } from '../constants/tasksCatalog';
import Fuse from 'fuse.js';

// Minimal modality + run model explorer state container
export function useModelExplorer() {
  // Modality chips
  const [selectedInputModalities, setSelectedInputModalities] = useState<InputType[]>([]);
  const [selectedOutputModalities, setSelectedOutputModalities] = useState<OutputType[]>([]);
  const ALL_INPUT_MODALITIES: InputType[] = ['text','image','audio','video','document'];
  const ALL_OUTPUT_MODALITIES: OutputType[] = ['text','image','audio','video','embedding','boxes','mask','depth','3d'];

  // Models & intents
  const modelsQuery = useQuery<ModelSummary[]>({
    queryKey: ['preloaded-models'],
    queryFn: async () => (await api.get('/models/preloaded', { params: { limit: 50000 } })).data,
    staleTime: 60*60*1000,
  });
  const intentsQuery = useQuery<Intent[]>({
    queryKey: ['intents'],
    queryFn: async () => (await api.get('/intents')).data,
    staleTime: 60*60*1000,
  });
  const allModels = modelsQuery.data || [];
  const allIntents = intentsQuery.data || [];

  // Selected model
  const [selectedModel, setSelectedModel] = useState<ModelSummary | null>(null);
  const selectModel = (m: ModelSummary) => setSelectedModel(m);

  // Derive tasks for selection (OR semantics)
  const tasksFromIO = useMemo(() => {
    if (!selectedInputModalities.length && !selectedOutputModalities.length) return [] as string[];
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
  const filteredModels = useMemo(() => {
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
  const loadMoreModels = () => setModelChunkSize(s => Math.min(filteredModels.length, s + 60));

  // Availability (disable impossible combos after selection in other group)
  const isInputComboViable = (inputs: InputType[], outputs: OutputType[]) => {
    for (const tm of Object.values(taskModalities)) {
      if (inputs.length > 1 && !tm.multiInputSupport) continue;
      if (!inputs.every(i => tm.input.includes(i))) continue;
      if (!outputs.every(o => tm.output.includes(o))) continue;
      return true;
    }
    return false;
  };
  const isOutputComboViable = (inputs: InputType[], outputs: OutputType[]) => {
    for (const tm of Object.values(taskModalities)) {
      if (inputs.length > 1 && !tm.multiInputSupport) continue;
      if (!inputs.every(i => tm.input.includes(i))) continue;
      if (!outputs.every(o => tm.output.includes(o))) continue;
      return true;
    }
    return false;
  };

  const availableInputModalities = useMemo(() => {
    if (!selectedInputModalities.length && !selectedOutputModalities.length) return ALL_INPUT_MODALITIES;
    const candidates: InputType[] = [];
    for (const m of ALL_INPUT_MODALITIES) {
      if (selectedInputModalities.includes(m)) { candidates.push(m); continue; }
      const next = [...selectedInputModalities, m];
      if (isInputComboViable(next, selectedOutputModalities)) candidates.push(m);
    }
    return candidates;
  }, [selectedInputModalities, selectedOutputModalities]);

  const availableOutputModalities = useMemo(() => {
    if (!selectedInputModalities.length && !selectedOutputModalities.length) return ALL_OUTPUT_MODALITIES;
    const candidates: OutputType[] = [];
    for (const o of ALL_OUTPUT_MODALITIES) {
      if (selectedOutputModalities.includes(o)) { candidates.push(o); continue; }
      const nextOut = [...selectedOutputModalities, o];
      if (isOutputComboViable(selectedInputModalities, nextOut)) candidates.push(o);
    }
    return candidates;
  }, [selectedInputModalities, selectedOutputModalities]);

  const toggleInputModality = (m: InputType) => setSelectedInputModalities(prev => prev.includes(m) ? prev.filter(x => x!==m) : [...prev, m]);
  const toggleOutputModality = (m: OutputType) => setSelectedOutputModalities(prev => prev.includes(m) ? prev.filter(x => x!==m) : [...prev, m]);
  const clearFilters = () => { setSelectedInputModalities([]); setSelectedOutputModalities([]); setModelChunkSize(60); };

  // Model tasks (from pipeline_tag + tag intersection with known tasks)
  const modelTasks = useMemo(() => {
    if (!selectedModel) return [] as string[];
    const set = new Set<string>();
    if (selectedModel.pipeline_tag) set.add(selectedModel.pipeline_tag);
    if (selectedModel.tags) {
      const knownTasks = new Set(Object.keys(taskModalities));
      selectedModel.tags.forEach(t => { if (knownTasks.has(t)) set.add(t); });
    }
    return Array.from(set);
  }, [selectedModel]);

  // Candidate intents (those whose hf_tasks intersect model tasks OR pipeline tag)
  const candidateIntents = useMemo(() => {
    if (!selectedModel) return [] as Intent[];
    return allIntents.filter(it => it.hf_tasks.some(t => t === selectedModel.pipeline_tag || modelTasks.includes(t)));
  }, [selectedModel, allIntents, modelTasks]);

  // Run intent selection
  const [runIntentId, setRunIntentId] = useState<string | null>(null);
  const runIntent = useMemo(() => runIntentId ? allIntents.find(i => i.id === runIntentId) : undefined, [runIntentId, allIntents]);
  // Ensure default intent when model selected
  if (selectedModel && !runIntentId && candidateIntents.length) setRunIntentId(candidateIntents[0].id);

  // Selected task
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  if (selectedModel && !selectedTask && modelTasks.length) setSelectedTask(modelTasks[0]);

  // Input states
  const [textInput, setTextInput] = useState('');
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [audioB64, setAudioB64] = useState<string | null>(null);
  const [videoB64, setVideoB64] = useState<string | null>(null);
  const [documentB64, setDocumentB64] = useState<string | null>(null);
  const handleGenericFile = (file: File, setter: (v:string)=>void) => new Promise<void>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => { const r = reader.result; if (typeof r === 'string') setter(r); resolve(); };
    reader.readAsDataURL(file);
  });
  const handleImageFile = (file: File) => handleGenericFile(file, v=>setImageB64(v));
  const handleAudioFile = (file: File) => handleGenericFile(file, v=>setAudioB64(v));
  const handleVideoFile = (file: File) => handleGenericFile(file, v=>setVideoB64(v));
  const handleDocumentFile = (file: File) => handleGenericFile(file, v=>setDocumentB64(v));
  const [extraArgsJson, setExtraArgsJson] = useState<string>('{}');
  const [extraArgsError, setExtraArgsError] = useState<string | null>(null);
  const parseExtraArgs = () => {
    try { if (!extraArgsJson.trim()) return {}; const o = JSON.parse(extraArgsJson); if (typeof o === 'object') { setExtraArgsError(null); return o; } setExtraArgsError('Must be JSON object'); return {}; } catch(e:any){ setExtraArgsError(e.message); return {}; }
  };

  // Determine required modalities from taskModalities dynamically
  const effectiveTasks = selectedTask ? [selectedTask] : (runIntent ? runIntent.hf_tasks : []);
  const requiredInputsSet = useMemo(() => {
    const s = new Set<InputType>();
    effectiveTasks.forEach(t => { const tm = taskModalities[t]; if (tm) tm.input.forEach(i => s.add(i as InputType)); });
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

  // Inference hooks
  const inference = useInference();
  const curlExample = useCurlExample();
  const inferencePending = inference.isPending;
  const curlPending = curlExample.isPending;

  // Determine primary input_type for request (fallback to text)
  const primaryInputType: InputType = requiresImage ? 'image' : requiresAudio ? 'audio' : requiresVideo ? 'video' : requiresDocument ? 'document' : 'text';

  // Evaluate readiness: explicit combo tasks need all listed combo inputs; otherwise at least one among allowed modalities
  const provided: Record<InputType, boolean> = {
    text: !!textInput.trim(),
    image: !!imageB64,
    audio: !!audioB64,
    video: !!videoB64,
    document: !!documentB64,
    multimodal: false,
  };
  const allowedModalities = Array.from(requiredInputsSet);
  const hasAtLeastOne = allowedModalities.some(m => provided[m]);
  const allComboSatisfied = combinationRequired ? (provided['text'] && provided['image']) : true;
  const canRun = !!(selectedModel && runIntent && allComboSatisfied && (combinationRequired ? true : hasAtLeastOne));

  const buildInputs = () => {
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
    if (selectedTask) (extra as any)._task = selectedTask;
    if (Object.keys(extra).length) inputs.extra_args = extra;
    return inputs;
  };

  const runModel = () => {
    if (!canRun || !selectedModel || !runIntent) return;
    const inputs = buildInputs();
    const body: InferenceRequest = { model_id: selectedModel.id, intent_id: runIntent.id, input_type: primaryInputType, inputs };
    inference.mutate(body, { onSuccess: (data) => {
      const id = `${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
      const newRun: RunRecord = { id, createdAt: new Date().toISOString(), inputType: primaryInputType, intent: runIntent, model: selectedModel, inputText: textInput, result: data.result, runtime_ms: data.runtime_ms, requestInputs: inputs };
      setRuns(prev => [...prev, newRun]); setSelectedRunId(id);
    }});
  };

  const showCurl = () => {
    if (!canRun || !selectedModel || !runIntent) return;
    const inputs = buildInputs();
    const body: InferenceRequest = { model_id: selectedModel.id, intent_id: runIntent.id, input_type: primaryInputType, inputs };
    curlExample.mutate(body, { onSuccess: (data) => {
      setRuns(prev => prev.map(r => r.id === selectedRunId ? { ...r, curl: data.command } : r));
    }});
  };

  const selectRun = (id: string) => { setSelectedRunId(id); const r = runs.find(rr => rr.id === id); if (r) { setSelectedModel(r.model); setTextInput(r.inputText); setRunIntentId(r.intent.id); } };

  const effectiveRunIntent = runIntent; // alias for panel

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
    candidateIntents, runIntentId, setRunIntentId, runIntent, effectiveRunIntent,
    modelTasks, selectedTask, setSelectedTask,
    // Input handling
    textInput, setTextInput,
    requiresImage, requiresText, requiresAudio, requiresVideo, requiresDocument,
    imageB64, handleImageFile,
    audioB64, handleAudioFile,
    videoB64, handleVideoFile,
    documentB64, handleDocumentFile,
    extraArgsJson, setExtraArgsJson, extraArgsError,
    // Execution
    canRun, runModel, showCurl, inferencePending, curlPending,
    // Runs history
    runs, selectedRunId, selectRun,
    // Misc
    tasksFromIO,
    clearFilters,
    sortBy, setSortBy,
    isLoadingModels: modelsQuery.isLoading, isLoadingIntents: intentsQuery.isLoading,
    searchQuery, setSearchQuery,
  };
}

export type UseModelExplorerReturn = ReturnType<typeof useModelExplorer>;
