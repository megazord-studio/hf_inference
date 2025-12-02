import { RunRecord } from '../types';

interface RunHistoryProps {
  runs: RunRecord[];
  selectedRunId: string | null;
  onSelect: (id: string) => void;
}

export function RunHistory({ runs, selectedRunId, onSelect }: RunHistoryProps) {
  if (!runs.length) {
    return (
      <div className="text-xs opacity-60 italic px-2 py-1">
        No runs yet. Run a model to see history here.
      </div>
    );
  }

  return (
    <ul className="menu menu-sm px-0 py-1 space-y-1 max-h-64 overflow-auto">
      {runs
        .slice()
        .sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1))
        .map((run) => (
          <li key={run.id}>
            <button
              type="button"
              className={`flex flex-col items-start text-left rounded-md px-2 py-1.5 border bg-base-100 hover:bg-base-200 transition-colors ${
                run.id === selectedRunId ? 'border-primary' : 'border-base-300'
              }`}
              onClick={() => onSelect(run.id)}
            >
              <span className="text-xs font-semibold truncate">
                {run.intent.label} · {run.model.id}
              </span>
              <span className="text-[10px] opacity-70 truncate">
                {new Date(run.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </button>
          </li>
        ))}
    </ul>
  );
}

