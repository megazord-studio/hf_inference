import { Route, Routes, Navigate } from 'react-router-dom';
import ModelExplorerPage from './pages/ModelExplorerPage';

function App() {
  return (
    <div className="min-h-screen bg-base-200 text-base-content">
      <header className="navbar bg-base-100 border-b border-base-300 px-4">
        <div className="flex-1 flex items-center gap-2">
          <span className="font-semibold">HF Inference Lab</span>
          <span className="badge badge-outline badge-xs">explorer</span>
        </div>
      </header>
      <main className="p-4">
        <Routes>
          <Route path="/" element={<ModelExplorerPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
