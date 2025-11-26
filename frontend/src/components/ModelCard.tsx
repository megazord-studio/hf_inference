import { ExternalLink } from 'lucide-react';
import type { ModelSummary } from '../types';

interface ModelCardProps {
  model: ModelSummary;
  taskLabel?: string;
  selected?: boolean;
  onSelect?: () => void;
  gatedOverride?: boolean;
}

export function ModelCard({ model, taskLabel, selected, onSelect, gatedOverride }: ModelCardProps) {
  const hfUrl = `https://huggingface.co/${model.id}`;
  const gv = gatedOverride !== undefined ? gatedOverride : (model as any).gated;
  const isGated = gv === true || (typeof gv === 'string' && gv.trim().toLowerCase() !== 'none');

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`card bg-base-100 border cursor-pointer text-left transition-colors w-full ${
        selected ? 'border-primary shadow-sm' : 'border-base-300 hover:border-base-200'
      }`}
    >
      <div className="card-body p-4 space-y-1">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <h3 className="font-semibold text-sm truncate" title={model.id}>{model.id}</h3>
            <a
              href={hfUrl}
              target="_blank"
              rel="noreferrer"
              className="btn btn-ghost btn-xxs p-0 h-4 min-h-0 text-base-content/70 hover:text-base-content"
              onClick={e => e.stopPropagation()}
              aria-label="Open on Hugging Face"
            >
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
          {isGated && (
            <div className="tooltip tooltip-left" data-tip="Gated repository on Hugging Face">
              <span className="badge badge-xs badge-warning font-mono">G</span>
            </div>
          )}
        </div>
        {taskLabel && (
          <p className="text-xs opacity-70">Task: {taskLabel}</p>
        )}
        {model.likes != null && (
          <p className="text-xs opacity-60">
            ❤ {model.likes} · ⬇ {model.downloads}
          </p>
        )}
      </div>
    </button>
  );
}
