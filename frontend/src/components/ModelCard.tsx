import type { ModelSummary } from '../types';

interface ModelCardProps {
  model: ModelSummary;
  taskLabel?: string;
  selected?: boolean;
  onSelect?: () => void;
}

export function ModelCard({ model, taskLabel, selected, onSelect }: ModelCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`card bg-base-100 border cursor-pointer text-left transition-colors w-full ${
        selected ? 'border-primary shadow-sm' : 'border-base-300 hover:border-base-200'
      }`}
    >
      <div className="card-body p-4 space-y-1">
        <h3 className="font-semibold text-sm truncate" title={model.id}>
          {model.id}
        </h3>
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

