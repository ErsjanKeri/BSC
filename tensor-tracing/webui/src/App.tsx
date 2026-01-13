/**
 * Main Application Component
 *
 * 3-View Dynamic Layout:
 * ┌─────────────────────────────────┬─────────────────────────────────┐
 * │                                 │                                 │
 * │     View 1: Computation Graph   │   View 2: Trace & Timeline      │
 * │            (50%)                │            (50%)                │
 * ├─────────────────────────────────┴─────────────────────────────────┤
 * │                                                                   │
 * │              View 3: Memory Heatmap (100%)                        │
 * │                                                                   │
 * └───────────────────────────────────────────────────────────────────┘
 *
 * Features:
 * - Drag & drop views to reorder
 * - Toggle views on/off with header buttons
 * - Dynamic layout adjusts to number of visible views
 * - Inter-connected: Click in one view highlights in others
 */

import { useEffect, useState } from 'react';
import { useAppStore } from './stores/useAppStore';
import type { ViewType } from './stores/useAppStore';
import { GraphView } from './components/GraphView';
import { TraceView } from './components/TraceView';
import { HeatmapView } from './components/HeatmapView';
import { ViewContainer } from './components/ViewContainer';

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

// Render the appropriate view component
function renderView(viewType: ViewType, isFullScreen: boolean) {
  switch (viewType) {
    case 'graph':
      return <GraphView isFullScreen={isFullScreen} />;
    case 'trace':
      return <TraceView isFullScreen={isFullScreen} />;
    case 'heatmap':
      return <HeatmapView isFullScreen={isFullScreen} />;
    default:
      return null;
  }
}

// Dynamic layout component that adapts to number of visible views
interface DynamicLayoutProps {
  visibleViews: ViewType[];
  viewOrder: ViewType[];
  onToggleView: (view: ViewType) => void;
  onReorderViews: (newOrder: ViewType[]) => void;
  draggedView: ViewType | null;
  setDraggedView: (view: ViewType | null) => void;
}

function DynamicLayout({
  visibleViews,
  viewOrder,
  onToggleView,
  onReorderViews,
  draggedView,
  setDraggedView,
}: DynamicLayoutProps) {
  // Get visible views in order
  const orderedVisibleViews = viewOrder.filter(v => visibleViews.includes(v));
  const numViews = orderedVisibleViews.length;

  const handleDrop = (targetView: ViewType) => {
    if (!draggedView || draggedView === targetView) return;

    // Swap positions
    const newOrder = [...viewOrder];
    const dragIndex = newOrder.indexOf(draggedView);
    const dropIndex = newOrder.indexOf(targetView);

    if (dragIndex !== -1 && dropIndex !== -1) {
      [newOrder[dragIndex], newOrder[dropIndex]] = [newOrder[dropIndex], newOrder[dragIndex]];
      onReorderViews(newOrder);
    }
  };

  if (numViews === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <span className="text-gray-500">No views visible. Use buttons above to show views.</span>
      </div>
    );
  }

  if (numViews === 1) {
    // Single view: 100% width → isFullScreen = true
    const view = orderedVisibleViews[0];
    return (
      <div className="flex-1 p-4 min-h-0">
        <ViewContainer
          viewType={view}
          onClose={() => onToggleView(view)}
          onDragStart={setDraggedView}
          onDragEnd={() => setDraggedView(null)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {renderView(view, true)}
        </ViewContainer>
      </div>
    );
  }

  if (numViews === 2) {
    // Two views: stacked vertically, each 100% width → isFullScreen = true
    return (
      <div className="flex-1 flex flex-col gap-4 p-4 min-h-0">
        {orderedVisibleViews.map(view => (
          <div key={view} className="flex-1 min-h-0">
            <ViewContainer
              viewType={view}
              onClose={() => onToggleView(view)}
              onDragStart={setDraggedView}
              onDragEnd={() => setDraggedView(null)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              {renderView(view, true)}
            </ViewContainer>
          </div>
        ))}
      </div>
    );
  }

  // Three views: top 50/50 (not full), bottom 100% (full)
  return (
    <div className="flex-1 flex flex-col gap-4 p-4 min-h-0">
      {/* Top row: first 2 views side-by-side (50% each → isFullScreen = false) */}
      <div className="flex-1 grid grid-cols-2 gap-4 min-h-0">
        {orderedVisibleViews.slice(0, 2).map(view => (
          <div key={view} className="min-h-0 min-w-0">
            <ViewContainer
              viewType={view}
              onClose={() => onToggleView(view)}
              onDragStart={setDraggedView}
              onDragEnd={() => setDraggedView(null)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
            >
              {renderView(view, false)}
            </ViewContainer>
          </div>
        ))}
      </div>

      {/* Bottom row: third view full width (100% → isFullScreen = true) */}
      <div className="flex-1 min-h-0">
        <ViewContainer
          viewType={orderedVisibleViews[2]}
          onClose={() => onToggleView(orderedVisibleViews[2])}
          onDragStart={setDraggedView}
          onDragEnd={() => setDraggedView(null)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          {renderView(orderedVisibleViews[2], true)}
        </ViewContainer>
      </div>
    </div>
  );
}

function Header() {
  const { currentTokenId, graphData, traceData, memoryMap, visibleViews, toggleView } = useAppStore();

  const viewButtons: { type: ViewType; label: string }[] = [
    { type: 'graph', label: 'Graph' },
    { type: 'trace', label: 'Logs' },
    { type: 'heatmap', label: 'Heatmap' },
  ];

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

      {/* View toggle buttons */}
      <div className="flex gap-2 mr-6">
        {viewButtons.map(({ type, label }) => {
          const isVisible = visibleViews.includes(type);
          return (
            <button
              key={type}
              onClick={() => toggleView(type)}
              className={`px-3 py-1 text-sm rounded transition-colors ${
                isVisible
                  ? 'bg-blue-600 text-white hover:bg-blue-500'
                  : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
              title={isVisible ? `Hide ${label}` : `Show ${label}`}
            >
              {label} {isVisible ? 'ON' : 'OFF'}
            </button>
          );
        })}
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
  const {
    loadMemoryMap,
    loadBufferStats,
    loadTokenData,
    isLoading,
    loadingError,
    fullScreenView,
    setFullScreen,
    visibleViews,
    viewOrder,
    toggleView,
    reorderViews,
  } = useAppStore();

  const [draggedView, setDraggedView] = useState<ViewType | null>(null);

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
          </div>
        </div>
      ) : (
        // Dynamic 3-View Grid Layout with Drag & Drop
        <DynamicLayout
          visibleViews={visibleViews}
          viewOrder={viewOrder}
          onToggleView={toggleView}
          onReorderViews={reorderViews}
          draggedView={draggedView}
          setDraggedView={setDraggedView}
        />
      )}
    </div>
  );
}
