import { useState, useEffect, useRef } from 'react';
import { Image as ImageIcon, Type as TextIcon, Code, Play, Loader2, Clipboard, CheckCircle2, ChevronDown, ChevronRight, HelpCircle } from 'lucide-react';
import { tasksInfo } from '../../constants/tasksCatalog';
import type { UseModelExplorerReturn } from '../../hooks/useModelExplorer';

interface RunPanelProps { m: UseModelExplorerReturn }

export function RunPanel({ m }: RunPanelProps) {
  const [tab, setTab] = useState<'input'|'output'>('input');
  const run = m.runs.find(r => r.id === m.selectedRunId);
  const previousRunIdRef = useRef<string | null>(m.selectedRunId);
  useEffect(() => {
    if (m.selectedRunId && m.selectedRunId !== previousRunIdRef.current) {
      // auto-switch to output when a new run appears
      setTab('output');
      previousRunIdRef.current = m.selectedRunId;
    }
  }, [m.selectedRunId]);
  const [copied, setCopied] = useState(false);
  const copyOutput = () => { if (!run?.result) return; navigator.clipboard.writeText(typeof run.result==='string'?run.result:JSON.stringify(run.result, null, 2)); setCopied(true); setTimeout(()=>setCopied(false),1500); };
  const [showRaw, setShowRaw] = useState(false);

  // Derive capabilities from modelTasks (descriptions & IO hints)
  const capabilityTasks = m.modelTasks.length ? m.modelTasks : (m.selectedModel?.pipeline_tag ? [m.selectedModel.pipeline_tag] : []);
  const capabilities = capabilityTasks.map(t => ({ id: t, info: tasksInfo[t] }));
  const textHints: string[] = [];
  const imageHints: string[] = [];
  const extraArgsHints: string[] = [];
  for (const c of capabilities) {
    const id = c.id;
    if (/translation/i.test(id)) textHints.push("Enter source text (e.g. 'Guten Morgen') – Extra args: { target_language: 'en' }");
    if (/summarization/i.test(id)) textHints.push("Paste long text to summarize – e.g. news article");
    if (/text-generation|text2text/i.test(id)) textHints.push("Provide a prompt. Use extra args for temperature, max_new_tokens");
    if (/question-answering/i.test(id)) textHints.push("Enter question and context separated by \n\n");
    if (/image-to-text/i.test(id)) imageHints.push("Upload an image to caption.");
    if (/text-to-image/i.test(id)) textHints.push("Describe the image you want (prompt)");
    if (/image-segmentation|mask-generation/i.test(id)) imageHints.push("Upload image; model returns segmentation mask(s)");
    if (/object-detection|zero-shot-object-detection/i.test(id)) imageHints.push("Upload image; returns bounding boxes & labels");
    if (/depth-estimation/i.test(id)) imageHints.push("Upload image; returns per-pixel depth map");
    if (/feature-extraction|sentence-similarity|embedding/i.test(id)) textHints.push("Enter sentence(s); model returns embeddings (extra args: { pooling: 'mean' })");
  }
  const unique = (arr: string[]) => Array.from(new Set(arr));
  const combinedTextHint = unique(textHints)[0];
  const combinedImageHint = unique(imageHints)[0];
  const combinedExtraArgsHint = unique(extraArgsHints)[0];

  const modalityIcons = (
    <span className="flex gap-1 items-center">
      {m.requiresImage && <ImageIcon className="w-3 h-3 text-info"/>}
      {m.requiresText && <TextIcon className="w-3 h-3 text-success"/>}
    </span>
  );

  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-md font-semibold flex items-center gap-2">Run / Inspect {modalityIcons}</h3>
          {m.selectedModel && (
            <span className="badge badge-outline badge-sm max-w-[16rem] truncate" title={m.selectedModel.id}>{m.selectedModel.id}</span>
          )}
        </div>
        {!m.selectedModel && <p className="text-xs opacity-60">Select a model to enable execution.</p>}
        {m.selectedModel && (
          <div className="space-y-3">
            {/* Capabilities list */}
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs font-semibold"><HelpCircle className="w-4 h-4"/>Capabilities</div>
              <div className="flex flex-wrap gap-2">
                {capabilities.map(c => (
                  <span key={c.id} className="badge badge-outline gap-1" title={c.info?.description || c.id}>{c.info?.label || c.id}</span>
                ))}
              </div>
              {capabilities.length===0 && <p className="text-[11px] opacity-60">No explicit task metadata; using pipeline tag only.</p>}
            </div>
            {/* Removed manual intent & task selection */}
            {/* Tabs */}
            <div className="join join-xs">
              <button className={`btn join-item btn-xs ${tab==='input'?'btn-primary':'btn-ghost'}`} onClick={()=>setTab('input')}>Input</button>
              <button className={`btn join-item btn-xs ${tab==='output'?'btn-primary':'btn-ghost'}`} onClick={()=>setTab('output')} disabled={!run}>Output</button>
            </div>
            {tab==='input' && (
              <div className="space-y-3">
                {m.requiresText && (
                  <div>
                    <label className="label text-xs font-semibold flex justify-between"><span>Text / Prompt</span>{combinedTextHint && <span className="opacity-60 text-[10px]">{combinedTextHint}</span>}</label>
                    <textarea
                      className="textarea textarea-bordered w-full min-h-28"
                      placeholder={combinedTextHint || "Enter text input"}
                      value={m.textInput}
                      onChange={e => m.setTextInput(e.target.value)}
                    />
                  </div>
                )}
                {m.requiresImage && (
                  <div>
                    <label className="label text-xs font-semibold flex justify-between"><span>Image</span>{combinedImageHint && <span className="opacity-60 text-[10px]">{combinedImageHint}</span>}</label>
                    <div
                      className="border border-dashed border-base-300 rounded-md p-4 flex flex-col items-center justify-center gap-2 cursor-pointer hover:border-base-400"
                      onDragOver={e => e.preventDefault()}
                      onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) void m.handleImageFile(f); }}
                    >
                      <input
                        type="file"
                        accept="image/*"
                        className="file-input file-input-bordered file-input-xs w-full max-w-xs"
                        onChange={e => { const f = e.target.files?.[0]; if (f) void m.handleImageFile(f); }}
                      />
                      {!m.imageB64 && <p className="text-[11px] opacity-60">Drag & drop or choose image.</p>}
                      {m.imageB64 && <img src={m.imageB64} alt="preview" className="max-h-40 rounded-md shadow" />}
                    </div>
                  </div>
                )}
                {m.requiresAudio && (
                  <div>
                    <label className="label text-xs font-semibold flex justify-between"><span>Audio</span><span className="opacity-60 text-[10px]">Upload audio file (wav/mp3)</span></label>
                    <div className="border border-dashed border-base-300 rounded-md p-4 flex flex-col items-center gap-2 hover:border-base-400">
                      <input type="file" accept="audio/*" className="file-input file-input-bordered file-input-xs w-full max-w-xs" onChange={e=>{const f=e.target.files?.[0]; if(f) void m.handleAudioFile(f);}} />
                      {!m.audioB64 && <p className="text-[11px] opacity-60">Select or drag an audio clip.</p>}
                      {m.audioB64 && <p className="text-[11px] text-success">Audio loaded.</p>}
                    </div>
                  </div>
                )}
                {m.requiresVideo && (
                  <div>
                    <label className="label text-xs font-semibold flex justify-between"><span>Video</span><span className="opacity-60 text-[10px]">Upload short video (mp4)</span></label>
                    <div className="border border-dashed border-base-300 rounded-md p-4 flex flex-col items-center gap-2 hover:border-base-400">
                      <input type="file" accept="video/*" className="file-input file-input-bordered file-input-xs w-full max-w-xs" onChange={e=>{const f=e.target.files?.[0]; if(f) void m.handleVideoFile(f);}} />
                      {!m.videoB64 && <p className="text-[11px] opacity-60">Select or drag a video clip.</p>}
                      {m.videoB64 && <video src={m.videoB64} className="max-h-40" controls />}
                    </div>
                  </div>
                )}
                {m.requiresDocument && (
                  <div>
                    <label className="label text-xs font-semibold flex justify-between"><span>Document</span><span className="opacity-60 text-[10px]">Upload document (pdf)</span></label>
                    <div className="border border-dashed border-base-300 rounded-md p-4 flex flex-col items-center gap-2 hover:border-base-400">
                      <input type="file" accept="application/pdf" className="file-input file-input-bordered file-input-xs w-full max-w-xs" onChange={e=>{const f=e.target.files?.[0]; if(f) void m.handleDocumentFile(f);}} />
                      {!m.documentB64 && <p className="text-[11px] opacity-60">Select or drag a PDF.</p>}
                      {m.documentB64 && <p className="text-[11px] text-success">Document loaded.</p>}
                    </div>
                  </div>
                )}
                <div>
                  <label className="label text-xs font-semibold flex justify-between"><span>Extra args (JSON)</span>{combinedExtraArgsHint && <span className="opacity-60 text-[10px]">{combinedExtraArgsHint}</span>}</label>
                  <textarea
                    className={`textarea textarea-bordered w-full font-mono text-xs min-h-20 ${m.extraArgsError ? 'textarea-error' : ''}`}
                    placeholder='{"temperature":0.7}'
                    value={m.extraArgsJson}
                    onChange={e => m.setExtraArgsJson(e.target.value)}
                  />
                  {m.extraArgsError && <p className="text-[11px] text-error mt-1">{m.extraArgsError}</p>}
                </div>
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="btn btn-sm btn-primary gap-1"
                    onClick={m.runModel}
                    disabled={!m.canRun || m.inferencePending}
                  >{m.inferencePending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />} Run</button>
                  <button
                    type="button"
                    className="btn btn-sm btn-outline gap-1"
                    onClick={m.showCurl}
                    disabled={!m.canRun || m.curlPending}
                  >{m.curlPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Code className="w-4 h-4" />} Curl</button>
                </div>
                {run && (
                  <div className="mt-2 alert alert-info overflow-auto">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-[11px]">Latest output preview</span>
                      <button type="button" className="btn btn-xxs btn-outline" onClick={()=>setTab('output')}>Open full</button>
                    </div>
                    <pre className="text-[10px] whitespace-pre-wrap line-clamp-6 max-h-32">{JSON.stringify(run.result, null, 2)}</pre>
                  </div>
                )}
              </div>
            )}
            {tab==='output' && run && (
              <div className="space-y-3">
                <div className="alert alert-success overflow-auto relative">
                  <div className="flex items-center justify-between">
                    <div className="font-semibold text-xs mb-1 flex items-center gap-2">Model output <span className="badge badge-xs badge-outline" title="Latency">{run.runtime_ms ?? '—'} ms</span></div>
                    <div className="flex gap-2 items-center">
                      <button className="btn btn-xxs btn-outline" onClick={copyOutput}>{copied? <CheckCircle2 className="w-3 h-3" /> : <Clipboard className="w-3 h-3" />}</button>
                      <button className="btn btn-xxs btn-outline" onClick={()=>setShowRaw(r=>!r)}>{showRaw?'Pretty':'Raw'}</button>
                    </div>
                  </div>
                  <pre className="text-xs whitespace-pre-wrap leading-relaxed">
{showRaw ? (typeof run.result==='string'? run.result : JSON.stringify(run.result)) : JSON.stringify(run.result, null, 2)}
                  </pre>
                  {run.requestInputs && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-[10px] opacity-70 flex items-center gap-1"><ChevronRight className="w-3 h-3" /> Request inputs</summary>
                      <pre className="text-[10px] whitespace-pre-wrap">{JSON.stringify(run.requestInputs, null, 2)}</pre>
                    </details>
                  )}
                  {run.curl && (
                    <details className="mt-2">
                      <summary className="cursor-pointer text-[10px] opacity-70 flex items-center gap-1"><ChevronDown className="w-3 h-3" /> curl</summary>
                      <pre className="text-[10px] whitespace-pre-wrap">{run.curl}</pre>
                    </details>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
