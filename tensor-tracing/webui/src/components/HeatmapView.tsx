/**
 * Memory Heatmap View - Simple and Precise
 *
 * Shows GGUF file layout with exact tensor positions (0-700MB)
 * Colors tensors by cumulative access count from trace logs
 * Horizontally scrollable for precision
 */

import { useMemo, useState } from 'react';
import { useAppStore } from '../stores/useAppStore';

interface HeatmapViewProps {
  isFullScreen: boolean;
}

// Simple helper to match tensor names with 19-char truncation
function findMatchingTensor(traceName: string, memoryMapTensors: any[]) {
  const exact = memoryMapTensors.find(t => t.name === traceName);
  if (exact) return exact;

  if (traceName.length === 19) {
    const prefix = memoryMapTensors.find(t => t.name.startsWith(traceName));
    if (prefix) return prefix;
  }

  return null;
}

export function HeatmapView({ isFullScreen }: HeatmapViewProps) {
  const { memoryMap, traceData, setFullScreen } = useAppStore();
  const [showAccessCounts, setShowAccessCounts] = useState(false);
  const [hoveredTensor, setHoveredTensor] = useState<any>(null);

  // Calculate cumulative access counts per tensor
  const tensorAccesses = useMemo(() => {
    if (!traceData || !memoryMap) return new Map();

    const counts = new Map<string, number>();

    // Initialize all tensors with 0
    memoryMap.tensors.forEach((t: any) => counts.set(t.name, 0));

    // Count accesses from traces
    traceData.entries.forEach((entry: any) => {
      entry.sources.forEach((source: any) => {
        if (source.memory_source === 'DISK') {
          const tensor = findMatchingTensor(source.name, memoryMap.tensors);
          if (tensor) {
            counts.set(tensor.name, (counts.get(tensor.name) || 0) + 1);
          }
        }
      });
    });

    return counts;
  }, [traceData, memoryMap]);

  // Find max access count for color scaling
  const maxAccess = useMemo(() => {
    return Math.max(...Array.from(tensorAccesses.values()), 1);
  }, [tensorAccesses]);

  // Get heat color
  const getHeatColor = (count: number) => {
    if (count === 0) return '#374151'; // gray-700
    const intensity = count / maxAccess;

    if (intensity < 0.3) {
      return `rgb(59, 130, 246)`; // blue
    } else if (intensity < 0.7) {
      return `rgb(234, 179, 8)`; // yellow
    } else {
      return `rgb(239, 68, 68)`; // red
    }
  };

  // Format bytes
  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
  };

  if (!memoryMap) {
    return (
      <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex items-center justify-center">
        <span className="text-gray-500">No memory map loaded</span>
      </div>
    );
  }

  // Scale: 1 pixel per MB (700MB = 700px width minimum)
  const PIXELS_PER_MB = 1;
  const totalWidthPx = Math.ceil(memoryMap.total_size_bytes / (1024 * 1024) * PIXELS_PER_MB);
  const TRACK_HEIGHT = 120;

  return (
    <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex flex-col">
      {/* Header */}
      <div className="h-12 border-b border-gray-700 flex items-center justify-between px-4">
        <div>
          <span className="text-white font-semibold">Memory Heatmap</span>
          <span className="ml-3 text-gray-400 text-sm">
            {memoryMap.tensors.length} tensors, {formatBytes(memoryMap.total_size_bytes)}
          </span>
        </div>

        <div className="flex items-center gap-4">
          <button
            onClick={() => setShowAccessCounts(!showAccessCounts)}
            className={`px-2 py-1 text-xs rounded ${
              showAccessCounts ? 'bg-amber-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            {showAccessCounts ? '# On' : '# Off'}
          </button>

          {!isFullScreen && (
            <button
              onClick={() => setFullScreen('heatmap')}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
            >
              ⛶ Full Screen
            </button>
          )}

          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-400">Heat:</span>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 bg-gray-700"></div>
              <span className="text-gray-500">0</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 bg-blue-500"></div>
              <span className="text-gray-500">Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 bg-yellow-500"></div>
              <span className="text-gray-500">Med</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 bg-red-500"></div>
              <span className="text-gray-500">High</span>
            </div>
          </div>
        </div>
      </div>

      {/* DISK Track - Horizontally Scrollable */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="h-8 px-4 bg-gray-800 border-b border-gray-700 flex items-center">
          <span className="text-white font-semibold text-sm">DISK (GGUF File Layout)</span>
          <span className="ml-3 text-gray-400 text-xs">Scroll horizontally →</span>
        </div>

        <div className="flex-1 overflow-x-auto overflow-y-hidden custom-scrollbar p-4">
          <div className="relative" style={{ width: `${totalWidthPx}px`, height: `${TRACK_HEIGHT}px` }}>
            {/* Ruler */}
            <div className="absolute top-0 left-0 right-0 h-6 border-b border-gray-600">
              {Array.from({ length: 8 }, (_, i) => {
                const bytes = (i / 7) * memoryMap.total_size_bytes;
                const x = (bytes / memoryMap.total_size_bytes) * totalWidthPx;
                return (
                  <div
                    key={i}
                    className="absolute text-xs text-gray-400"
                    style={{ left: `${x}px`, transform: 'translateX(-50%)' }}
                  >
                    {formatBytes(bytes)}
                  </div>
                );
              })}
            </div>

            {/* Tensors */}
            <div className="absolute top-8 left-0 right-0 bottom-0">
              {memoryMap.tensors.map((tensor: any, idx: number) => {
                const startX = (tensor.offset_start / memoryMap.total_size_bytes) * totalWidthPx;
                const endX = (tensor.offset_end / memoryMap.total_size_bytes) * totalWidthPx;
                const width = Math.max(endX - startX, 1);
                const accessCount = tensorAccesses.get(tensor.name) || 0;
                const color = getHeatColor(accessCount);

                return (
                  <div
                    key={idx}
                    className="absolute border border-gray-600 cursor-pointer hover:border-blue-400 transition-colors"
                    style={{
                      left: `${startX}px`,
                      width: `${width}px`,
                      height: '100%',
                      backgroundColor: color,
                    }}
                    onMouseEnter={() => setHoveredTensor({ tensor, accessCount, x: startX, y: 0 })}
                    onMouseLeave={() => setHoveredTensor(null)}
                    title={`${tensor.name}\n${formatBytes(tensor.size_bytes)}\nAccesses: ${accessCount}`}
                  >
                    {showAccessCounts && accessCount > 0 && width > 30 && (
                      <div className="text-white text-xs font-bold text-center mt-2">
                        {accessCount}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Hover tooltip */}
            {hoveredTensor && (
              <div
                className="fixed bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg pointer-events-none z-50"
                style={{
                  left: `${hoveredTensor.x + 200}px`,
                  top: '200px',
                }}
              >
                <div className="text-white font-semibold text-sm mb-2">
                  {hoveredTensor.tensor.name}
                </div>
                <div className="space-y-1 text-xs">
                  <div>
                    <span className="text-gray-400">Size:</span>{' '}
                    <span className="text-white">{formatBytes(hoveredTensor.tensor.size_bytes)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Offset:</span>{' '}
                    <span className="text-white font-mono">
                      {formatBytes(hoveredTensor.tensor.offset_start)} - {formatBytes(hoveredTensor.tensor.offset_end)}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Accesses:</span>{' '}
                    <span className="text-white font-semibold">{hoveredTensor.accessCount}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* BUFFER Track - Simple list */}
      <div className="h-48 border-t border-gray-700 overflow-y-auto custom-scrollbar">
        <div className="h-8 px-4 bg-gray-800 border-b border-gray-700 flex items-center">
          <span className="text-white font-semibold text-sm">BUFFER (Runtime Tensors)</span>
        </div>
        <div className="p-4 text-gray-400 text-sm">
          Buffer visualization coming next...
        </div>
      </div>
    </div>
  );
}
