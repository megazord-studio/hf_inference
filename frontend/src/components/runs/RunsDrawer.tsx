import { useState, useMemo } from 'react';
import type { UseModelExplorerReturn } from '../../hooks/useModelExplorer';
import { Trash2, Star, StarOff, Search } from 'lucide-react';

interface RunsDrawerProps {
  m: UseModelExplorerReturn;
  open: boolean;
  onClose: () => void;
  onSelectRun: (id: string) => void;
  onClearRuns?: () => void;
}

export function RunsDrawer({ m, open, onClose, onSelectRun, onClearRuns }: RunsDrawerProps) {
  const [query, setQuery] = useState('');
  const [favorites, setFavorites] = useState<Set<string>>(new Set());

  const toggleFavorite = (id: string) => {
    setFavorites(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const filteredRuns = useMemo(() => {
    let list = m.runs.slice().sort((a,b)=> (a.createdAt < b.createdAt ? 1 : -1));
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(r => r.model.id.toLowerCase().includes(q) || r.intent.label.toLowerCase().includes(q));
    }
    // favorites at top
    return list.sort((a,b)=> {
      const af = favorites.has(a.id) ? 0 : 1;
      const bf = favorites.has(b.id) ? 0 : 1;
      if (af !== bf) return af - bf;
      return 0;
    });
  }, [m.runs, query, favorites]);

  return (
    <aside
      className={`fixed top-14 left-0 h-[calc(100vh-3.5rem)] w-80 bg-base-100 border-r border-base-300 shadow-sm z-30 transform transition-transform duration-300 ease-out flex flex-col ${open ? 'translate-x-0' : '-translate-x-full'}`}
    >
      <div className="px-3 py-2 flex items-center justify-between border-b border-base-300">
        <span className="text-xs font-semibold">Recent runs ({m.runs.length})</span>
        <div className="flex items-center gap-2">
          {m.runs.length > 0 && (
            <button className="btn btn-ghost btn-xs" title="Clear all" onClick={() => { setFavorites(new Set()); onClearRuns?.(); }}><Trash2 className="w-3 h-3" /></button>
          )}
          <button className="btn btn-ghost btn-xs" onClick={onClose}>✕</button>
        </div>
      </div>
      <div className="p-2 space-y-2 flex-1 overflow-hidden">
        <div className="flex gap-2 items-center">
          <Search className="w-4 h-4 opacity-60" />
          <input
            type="text"
            placeholder="Filter runs"
            className="input input-bordered input-xs flex-1"
            value={query}
            onChange={e=>setQuery(e.target.value)}
          />
        </div>
        <ul className="menu menu-xs px-0 flex-1 overflow-auto">
          {filteredRuns.length === 0 && <li className="px-2 py-2 text-[11px] opacity-60">No matching runs.</li>}
          {filteredRuns.map(run => (
            <li key={run.id} className="px-1">
              <div
                className={`flex flex-col gap-1 rounded-md border px-2 py-1.5 text-left cursor-pointer transition-colors ${run.id===m.selectedRunId ? 'border-primary bg-base-200' : 'border-base-300 hover:bg-base-200'} ${favorites.has(run.id)?'bg-base-200/70':''}`}
                onClick={() => { onSelectRun(run.id); onClose(); }}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[11px] font-semibold truncate" title={run.model.id}>{run.intent.label} · {run.model.id}</span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-xxs"
                    onClick={(e)=>{ e.stopPropagation(); toggleFavorite(run.id); }}
                  >{favorites.has(run.id) ? <Star className="w-3 h-3 text-warning" /> : <StarOff className="w-3 h-3 opacity-40" />}</button>
                </div>
                <div className="flex items-center justify-between text-[10px] opacity-70">
                  <span>{new Date(run.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  {run.runtime_ms != null && <span>{run.runtime_ms}ms</span>}
                </div>
                {Boolean(run.result) && (
                  <pre className="text-[9px] whitespace-pre-wrap max-h-16 overflow-hidden opacity-70">
                    {(() => {
                      const raw: string = typeof run.result === 'string' ? run.result : JSON.stringify(run.result ?? {});
                      return `${raw.slice(0,60)}${raw.length>60 ? '…' : ''}`;
                    })()}
                  </pre>
                )}
              </div>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
