/**
 * Main Application Component
 *
 * 4-View Layout:
 * ┌─────────────────────────────────┬─────────────────────────────────┐
 * │                                 │                                 │
 * │     View 1: Computation Graph   │   View 2: Timeline & Trace      │
 * │                                 │                                 │
 * ├─────────────────────────────────┼─────────────────────────────────┤
 * │                                 │                                 │
 * │     View 3: Memory Heatmap      │   View 4: 3D Transformer        │
 * │                                 │                                 │
 * └─────────────────────────────────┴─────────────────────────────────┘
 */

import { useEffect } from 'react';
import { useAppStore } from './stores/useAppStore';
import { GraphView } from './components/GraphView';
import { TraceView } from './components/TraceView';
import { HeatmapView } from './components/HeatmapView';
import { TransformerView } from './components/TransformerView';

function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center h-screen bg-gray-950">
      <div className="text-center">
        <div className="inline-block w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
        <div className="text-white text-lg">Loading tensor trace data...</div>
      </div>
    </div>
  );
}

function ErrorDisplay({ error }: { error: string }) {
  return (
    <div className="flex items-center justify-center h-screen bg-gray-950">
      <div className="max-w-2xl p-8 bg-red-900/20 border border-red-500 rounded-lg">
        <div className="text-red-400 text-xl font-semibold mb-2">Error Loading Data</div>
        <div className="text-red-300">{error}</div>
        <div className="mt-4 text-gray-400 text-sm">
          Make sure the data files are in the <code className="bg-gray-800 px-1 rounded">public/data/</code> directory:
          <ul className="list-disc list-inside mt-2">
            <li>memory-map.json</li>
            <li>buffer-timeline.json (optional)</li>
            <li>graphs/token-00000.json</li>
            <li>traces/token-00000.json</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function Header() {
  const { currentTokenId, graphData, traceData, memoryMap } = useAppStore();

  return (
    <div className="h-14 bg-gray-950 border-b border-gray-800 flex items-center px-6">
      <div className="flex-1">
        <h1 className="text-xl font-bold text-white">
          Tensor Trace Visualizer
        </h1>
        <div className="text-xs text-gray-500">
          llama.cpp inference analysis tool
        </div>
      </div>

      {memoryMap && graphData && traceData && (
        <div className="flex gap-6 text-sm">
          <div>
            <span className="text-gray-500">Token:</span>{' '}
            <span className="text-white font-mono">{currentTokenId}</span>
          </div>
          <div>
            <span className="text-gray-500">Model:</span>{' '}
            <span className="text-white">{memoryMap.model_name}</span>
          </div>
          <div>
            <span className="text-gray-500">Nodes:</span>{' '}
            <span className="text-white">{graphData.metadata.total_nodes}</span>
          </div>
          <div>
            <span className="text-gray-500">Traces:</span>{' '}
            <span className="text-white">{traceData.metadata.total_entries}</span>
          </div>
          <div>
            <span className="text-gray-500">Duration:</span>{' '}
            <span className="text-white">{traceData.metadata.duration_ms.toFixed(2)}ms</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  const { loadMemoryMap, loadBufferStats, loadTokenData, isLoading, loadingError, fullScreenView, setFullScreen } = useAppStore();

  // Load data on mount
  useEffect(() => {
    const initializeData = async () => {
      await loadMemoryMap();
      await loadBufferStats();  // Load buffer timeline (optional, won't fail if missing)
      await loadTokenData(0);   // Start with token 0
    };

    initializeData();
  }, [loadMemoryMap, loadBufferStats, loadTokenData]);

  // ESC key to exit full-screen
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && fullScreenView) {
        setFullScreen(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fullScreenView, setFullScreen]);

  // Show loading state
  if (isLoading && !loadingError) {
    return <LoadingSpinner />;
  }

  // Show error state
  if (loadingError) {
    return <ErrorDisplay error={loadingError} />;
  }

  // Main layout - either 4-view grid or full-screen single view
  return (
    <div className="w-screen h-screen bg-gray-950 flex flex-col overflow-hidden">
      <Header />

      {fullScreenView ? (
        // Full-screen single view
        <div className="flex-1 p-4 min-h-0">
          <div className="w-full h-full relative">
            {/* Exit full-screen button */}
            <button
              onClick={() => setFullScreen(null)}
              className="absolute top-4 right-4 z-50 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg shadow-lg flex items-center gap-2"
              title="Exit full-screen (ESC)"
            >
              <span>✕</span>
              <span>Exit Full-Screen</span>
            </button>

            {/* Render the appropriate full-screen view */}
            {fullScreenView === 'graph' && <GraphView isFullScreen={true} />}
            {fullScreenView === 'trace' && <TraceView isFullScreen={true} />}
            {fullScreenView === 'heatmap' && <HeatmapView isFullScreen={true} />}
            {fullScreenView === 'transformer' && <TransformerView isFullScreen={true} />}
          </div>
        </div>
      ) : (
        // 4-View Grid Layout
        <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-4 p-4 min-h-0">
          {/* Top-left: Computation Graph */}
          <div className="min-h-0 min-w-0">
            <GraphView isFullScreen={false} />
          </div>

          {/* Top-right: Timeline & Trace */}
          <div className="min-h-0 min-w-0">
            <TraceView isFullScreen={false} />
          </div>

          {/* Bottom-left: Memory Heatmap */}
          <div className="min-h-0 min-w-0">
            <HeatmapView isFullScreen={false} />
          </div>

          {/* Bottom-right: 3D Transformer */}
          <div className="min-h-0 min-w-0">
            <TransformerView isFullScreen={false} />
          </div>
        </div>
      )}
    </div>
  );
}
