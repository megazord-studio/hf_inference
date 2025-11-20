import { Image, Type as TextIcon, Music, Video, Box, Layers, Activity, ScanEye, Binary } from 'lucide-react';
import type { UseModelExplorerReturn } from '../../hooks/useModelExplorer';

interface ExplorerFiltersProps { m: UseModelExplorerReturn }

export function ExplorerFilters({ m }: ExplorerFiltersProps) {
  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body space-y-5">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Model Explorer</h2>
          <div className="flex gap-2">
            <button
              className="btn btn-xs btn-outline"
              disabled={m.selectedInputModalities.length===0 && m.selectedOutputModalities.length===0}
              onClick={m.clearFilters}
            >Reset</button>
          </div>
        </div>
        <p className="text-xs opacity-70">Filter models by desired input and output modalities. Select multiple to broaden results (OR match per group).</p>

        {/* Active modality summary */}
        {(m.selectedInputModalities.length>0 || m.selectedOutputModalities.length>0) && (
          <div className="flex flex-wrap gap-2">
            {m.selectedInputModalities.map(im => <div key={im} className="badge badge-primary gap-1 cursor-pointer" onClick={()=>m.toggleInputModality(im)}>{im} ×</div>)}
            {m.selectedOutputModalities.map(om => <div key={om} className="badge badge-secondary gap-1 cursor-pointer" onClick={()=>m.toggleOutputModality(om)}>{om} ×</div>)}
          </div>
        )}

        {/* Input modalities */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-xs">Input modalities {m.selectedInputModalities.length ? `(${m.selectedInputModalities.length})` : ''}</span>
            {m.selectedInputModalities.length>0 && <button className="btn btn-ghost btn-xs" onClick={()=>m.selectedInputModalities.forEach(im=>m.toggleInputModality(im))}>Clear</button>}
          </div>
          <div className="flex flex-wrap gap-2">
            {m.ALL_INPUT_MODALITIES.map(inp => {
              const disabled = !m.availableInputModalities.includes(inp);
              return (
                <button
                  key={inp}
                  type="button"
                  disabled={disabled}
                  onClick={()=>m.toggleInputModality(inp)}
                  className={`btn btn-xs gap-1 ${disabled ? 'btn-disabled opacity-40 cursor-not-allowed' : m.selectedInputModalities.includes(inp)?'btn-primary':'btn-outline'}`}
                >{inp==='text' && <TextIcon size={14}/>}{inp==='image' && <Image size={14}/>}{inp==='audio' && <Music size={14}/>}{inp==='video' && <Video size={14}/>}{inp==='document' && <Layers size={14}/>}{inp.charAt(0).toUpperCase()+inp.slice(1)}</button>
              );
            })}
          </div>
        </div>

        {/* Output modalities */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-xs">Output modalities {m.selectedOutputModalities.length ? `(${m.selectedOutputModalities.length})` : ''}</span>
            {m.selectedOutputModalities.length>0 && <button className="btn btn-ghost btn-xs" onClick={()=>m.selectedOutputModalities.forEach(om=>m.toggleOutputModality(om))}>Clear</button>}
          </div>
          <div className="flex flex-wrap gap-2">
            {m.ALL_OUTPUT_MODALITIES.map(out => {
              const disabled = !m.availableOutputModalities.includes(out);
              return (
                <button
                  key={out}
                  type="button"
                  disabled={disabled}
                  onClick={()=>m.toggleOutputModality(out)}
                  className={`btn btn-xs gap-1 ${disabled ? 'btn-disabled opacity-40 cursor-not-allowed' : m.selectedOutputModalities.includes(out)?'btn-secondary':'btn-outline'}`}
                >{out==='text' && <TextIcon size={14}/>}{out==='image' && <Image size={14}/>}{out==='audio' && <Music size={14}/>}{out==='video' && <Video size={14}/>}{out==='embedding' && <Binary size={14}/>}{out==='boxes' && <ScanEye size={14}/>}{out==='mask' && <Activity size={14}/>}{out==='depth' && <Box size={14}/>}{out==='3d' && <Box size={14}/>}{out.toUpperCase()}</button>
              );
            })}
          </div>
        </div>

        {/* Sort controls */}
        <div className="space-y-2">
          <span className="font-semibold text-xs">Sort by</span>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={()=>m.setSortBy('trending')} className={`btn btn-xs ${m.sortBy==='trending'?'btn-accent':'btn-outline'}`}>Trending</button>
            <button type="button" onClick={()=>m.setSortBy('downloads')} className={`btn btn-xs ${m.sortBy==='downloads'?'btn-accent':'btn-outline'}`}>Downloads</button>
            <button type="button" onClick={()=>m.setSortBy('likes')} className={`btn btn-xs ${m.sortBy==='likes'?'btn-accent':'btn-outline'}`}>Likes</button>
            <button type="button" onClick={()=>m.setSortBy('name')} className={`btn btn-xs ${m.sortBy==='name'?'btn-accent':'btn-outline'}`}>Name</button>
          </div>
        </div>

        {/* Fuzzy search */}
        <div className="space-y-2">
          <span className="font-semibold text-xs">Search models</span>
          <div className="flex gap-2">
            <input
              type="text"
              className="input input-bordered input-xs flex-1"
              placeholder="Fuzzy search (id, tag)"
              value={m.searchQuery}
              onChange={e=>m.setSearchQuery(e.target.value)}
            />
            {m.searchQuery && <button type="button" className="btn btn-xs" onClick={()=>m.setSearchQuery('')}>Clear</button>}
          </div>
        </div>

        {/* Summary */}
        <div className="flex flex-wrap gap-2 text-[10px] opacity-70">
          <span>Visible: {m.visibleModels.length}/{m.filteredModels.length}</span>
          {m.searchQuery && <span>Search: {m.searchQuery}</span>}
          {m.selectedInputModalities.length>0 && <span>Input mods: {m.selectedInputModalities.length}</span>}
          {m.selectedOutputModalities.length>0 && <span>Output mods: {m.selectedOutputModalities.length}</span>}
        </div>
      </div>
    </div>
  );
}
