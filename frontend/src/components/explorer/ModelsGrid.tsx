import { ModelCard } from '../ModelCard';
import type { UseModelExplorerReturn } from '../../hooks/useModelExplorer';

interface ModelsGridProps { m: UseModelExplorerReturn }

export function ModelsGrid({ m }: ModelsGridProps) {
  return (
    <div className="card bg-base-100 shadow-sm border border-base-300">
      <div className="card-body pb-3">
        {m.isLoadingModels && !m.visibleModels.length && <span className="loading loading-spinner" />}
        {!m.isLoadingModels && !m.visibleModels.length && <p className="text-xs opacity-70">No models match current filters.</p>}
        <div className="grid md:grid-cols-2 gap-3 max-h-[55vh] overflow-auto pr-1">
          {m.visibleModels.map(model => (
            <ModelCard
              key={model.id}
              model={model}
              taskLabel={model.pipeline_tag}
              selected={m.selectedModel?.id === model.id}
              onSelect={() => m.selectModel(model)}
            />
          ))}
        </div>
        {m.visibleModels.length < m.filteredModels.length && (
          <div className="mt-3 flex justify-center">
            <button className="btn btn-xs btn-outline" onClick={m.loadMoreModels}>Load more ({m.visibleModels.length}/{m.filteredModels.length})</button>
          </div>
        )}
      </div>
    </div>
  );
}
