import { useState } from 'react';
import { useModelExplorer } from '../hooks/useModelExplorer';
import { ExplorerFilters } from '../components/explorer/ExplorerFilters';
import { ModelsGrid } from '../components/explorer/ModelsGrid';
import { RunPanel } from '../components/run/RunPanel';
import { RunsDrawer } from '../components/runs/RunsDrawer';

function ModelExplorerPage() {
  const m = useModelExplorer();
  const [runsOpen, setRunsOpen] = useState(false);

  return (
    <div className="relative">
      {/* Runs drawer toggle button only when closed */}
      {!runsOpen && (
        <button
          type="button"
          className="btn btn-xs btn-outline absolute top-2 left-2 z-40"
          onClick={() => setRunsOpen(true)}
        >Show runs</button>
      )}
      {/* Sliding drawer for recent runs */}
      <RunsDrawer m={m} open={runsOpen} onClose={() => setRunsOpen(false)} onSelectRun={m.selectRun} />

      {/* Main two-column content shifts when drawer open */}
      <div className={`flex flex-col xl:flex-row gap-4 px-2 pt-10 ${runsOpen ? 'xl:ml-72' : ''}`}>
        {/* Explorer: 7/12 on xl */}
        <div className="space-y-6 flex-1 xl:w-7/12">
          <ExplorerFilters m={m} />
          <ModelsGrid m={m} />
        </div>

        {/* Run / Inspect: 5/12 on xl */}
        <div className="space-y-4 flex-1 xl:w-5/12">
          <RunPanel m={m} />
        </div>
      </div>
    </div>
  );
}

export default ModelExplorerPage;
