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

// Match tensor names exactly (no truncation with 128-byte names!)
function findMatchingTensor(traceName: string, memoryMapTensors: any[]) {
  // With 1024-byte format and 128-byte names, we have full names - exact match only
  return memoryMapTensors.find(t => t.name === traceName) || null;
}

export function HeatmapView({ isFullScreen }: HeatmapViewProps) {
  const { memoryMap, traceData, bufferTimeline, heatmapMode, setHeatmapMode, timeline } = useAppStore();
  const [showAccessCounts, setShowAccessCounts] = useState(false);
  const [hoveredTensor, setHoveredTensor] = useState<any>(null);
  const [zoomLevel, setZoomLevel] = useState(10); // Scale factor: 1x, 10x, 50x, 100x, 500x

  // Calculate temporal access counts per tensor (respects timeline.currentTime and heatmapMode)
  const tensorAccesses = useMemo(() => {
    if (!traceData || !memoryMap) {
      return {
        counts: new Map<string, number>(),
        timestamps: new Map<string, number[]>(),
        currentLayer: null,
      };
    }

    const counts = new Map<string, number>();
    const timestamps = new Map<string, number[]>(); // Track access times for tooltip

    // Initialize all tensors with 0
    memoryMap.tensors.forEach((t: any) => {
      counts.set(t.name, 0);
      timestamps.set(t.name, []);
    });

    // Determine which layer is currently active based on timeline
    let currentLayer: number | null = null;
    if (heatmapMode === 'current-layer' && traceData.entries.length > 0) {
      // Find the current layer by looking at entries up to current time
      for (let i = traceData.entries.length - 1; i >= 0; i--) {
        const entry = traceData.entries[i];
        if (entry.timestamp_relative_ms <= timeline.currentTime && entry.layer_id !== null) {
          currentLayer = entry.layer_id;
          break;
        }
      }
    }

    // Count accesses from traces up to current time
    traceData.entries.forEach((entry: any) => {
      // Only count entries up to current timeline position
      if (entry.timestamp_relative_ms > timeline.currentTime) return;

      // For "current-layer" mode, only count entries from the current layer
      if (heatmapMode === 'current-layer') {
        if (currentLayer === null || entry.layer_id !== currentLayer) {
          return; // Skip entries not in current layer
        }
      }

      // Count DISK accesses only (GGUF weights)
      entry.sources.forEach((source: any) => {
        if (source.memory_source === 'DISK') {
          const tensor = findMatchingTensor(source.name, memoryMap.tensors);
          if (tensor) {
            counts.set(tensor.name, (counts.get(tensor.name) || 0) + 1);
            timestamps.get(tensor.name)?.push(entry.timestamp_relative_ms);
          }
        }
      });
    });

    return { counts, timestamps, currentLayer };
  }, [traceData, memoryMap, timeline.currentTime, heatmapMode]);

  // Find max access count for color scaling
  const maxAccess = useMemo(() => {
    return Math.max(...Array.from(tensorAccesses.counts.values()), 1);
  }, [tensorAccesses]);

  // Get heat color - continuous gradient from dark to red
  const getHeatColor = (count: number) => {
    if (count === 0) return '#374151'; // gray-700 (dark, no access)

    const intensity = Math.min(count / maxAccess, 1.0);

    // Gradient: dark red → bright red
    // Low access: dark red (rgb(139, 0, 0))
    // High access: bright red (rgb(255, 0, 0))
    const r = Math.floor(139 + (255 - 139) * intensity);
    const g = 0;
    const b = 0;

    return `rgb(${r}, ${g}, ${b})`;
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

  // Scale with zoom: base 1 pixel per MB, multiplied by zoom level
  const PIXELS_PER_MB = 1 * zoomLevel;
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
          {/* Zoom Control */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs">Zoom:</span>
            <div className="flex gap-1 bg-gray-800 rounded p-1">
              {[1, 10, 50, 100, 500].map(zoom => (
                <button
                  key={zoom}
                  onClick={() => setZoomLevel(zoom)}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    zoomLevel === zoom
                      ? 'bg-green-600 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                  title={`${zoom}x zoom (${(2098 * zoom).toFixed(0)}px wide)`}
                >
                  {zoom}x
                </button>
              ))}
            </div>
          </div>

          {/* Heatmap Mode Toggle */}
          <div className="flex gap-1 bg-gray-800 rounded p-1">
            <button
              onClick={() => setHeatmapMode('total-accumulated')}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                heatmapMode === 'total-accumulated'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Total
            </button>
            <button
              onClick={() => setHeatmapMode('current-layer')}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                heatmapMode === 'current-layer'
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Layer {tensorAccesses.currentLayer !== null ? tensorAccesses.currentLayer : '?'}
            </button>
          </div>

          <button
            onClick={() => setShowAccessCounts(!showAccessCounts)}
            className={`px-2 py-1 text-xs rounded ${
              showAccessCounts ? 'bg-amber-600 text-white' : 'bg-gray-700 text-gray-300'
            }`}
          >
            {showAccessCounts ? '# On' : '# Off'}
          </button>

          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-400">Heat:</span>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 bg-gray-700"></div>
              <span className="text-gray-500">0</span>
            </div>
            <div className="flex items-center gap-0">
              <div className="w-16 h-3" style={{
                background: 'linear-gradient(to right, rgb(139, 0, 0), rgb(255, 0, 0))'
              }}></div>
              <span className="text-gray-500 ml-1">Max</span>
            </div>
          </div>
        </div>
      </div>

      {/* DISK Track - Horizontally Scrollable */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="h-8 px-4 bg-gray-800 border-b border-gray-700 flex items-center justify-between">
          <div>
            <span className="text-white font-semibold text-sm">DISK (GGUF File Layout)</span>
            <span className="ml-3 text-gray-400 text-xs">
              {zoomLevel}x zoom • {totalWidthPx.toLocaleString()}px wide • Scroll horizontally →
            </span>
          </div>
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
                // Minimum width: 2px so even tiny 8KB tensors are visible when zoomed
                const width = Math.max(endX - startX, 2);
                const accessCount = tensorAccesses.counts.get(tensor.name) || 0;
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
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setHoveredTensor({
                        tensor,
                        accessCount,
                        accessTimestamps: tensorAccesses.timestamps.get(tensor.name) || [],
                        x: rect.left,
                        y: rect.top,
                      });
                    }}
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

            {/* Enhanced Hover Tooltip */}
            {hoveredTensor && (
              <div
                className="fixed bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg pointer-events-none z-50 max-w-md"
                style={{
                  left: `${Math.min(hoveredTensor.x + 20, window.innerWidth - 350)}px`,
                  top: `${hoveredTensor.y + 80}px`,
                }}
              >
                <div className="text-white font-semibold text-sm mb-2">
                  {hoveredTensor.tensor.name}
                </div>
                <div className="space-y-1 text-xs">
                  <div>
                    <span className="text-gray-400">Layer:</span>{' '}
                    <span className="text-white">
                      {hoveredTensor.tensor.layer_id !== null ? `L${hoveredTensor.tensor.layer_id}` : 'N/A'}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Size:</span>{' '}
                    <span className="text-white">{formatBytes(hoveredTensor.tensor.size_bytes)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Offset:</span>{' '}
                    <span className="text-white font-mono text-xs">
                      {hoveredTensor.tensor.offset_start.toLocaleString()} - {hoveredTensor.tensor.offset_end.toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Category:</span>{' '}
                    <span className="text-white">{hoveredTensor.tensor.category}</span>
                  </div>
                  <div className="pt-2 border-t border-gray-700">
                    <span className="text-gray-400">Access Count:</span>{' '}
                    <span className="text-white font-semibold">{hoveredTensor.accessCount}</span>
                    {hoveredTensor.accessCount > 0 && (
                      <span className="ml-2 text-gray-500">
                        (up to {timeline.currentTime.toFixed(1)}ms)
                      </span>
                    )}
                  </div>
                  {hoveredTensor.accessTimestamps && hoveredTensor.accessTimestamps.length > 0 && (
                    <div className="pt-1">
                      <span className="text-gray-400">Access Times:</span>{' '}
                      <div className="text-white font-mono text-xs mt-1 max-h-20 overflow-y-auto">
                        {hoveredTensor.accessTimestamps.slice(0, 10).map((t: number, i: number) => (
                          <div key={i}>{t.toFixed(2)}ms</div>
                        ))}
                        {hoveredTensor.accessTimestamps.length > 10 && (
                          <div className="text-gray-500">...and {hoveredTensor.accessTimestamps.length - 10} more</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* BUFFER Track - Runtime Buffers */}
      <div className="h-48 border-t border-gray-700 overflow-y-auto custom-scrollbar">
        <div className="h-8 px-4 bg-gray-800 border-b border-gray-700 flex items-center justify-between">
          <span className="text-white font-semibold text-sm">BUFFER (Runtime Tensors)</span>
          {bufferTimeline && (
            <span className="text-gray-400 text-xs">
              {bufferTimeline.metadata.total_buffers} buffers •
              Peak: {bufferTimeline.metadata.peak_occupancy_mb.toFixed(1)} MB
            </span>
          )}
        </div>

        {bufferTimeline ? (
          <div className="p-4 space-y-3">
            {bufferTimeline.buffers.map((buffer: any, idx: number) => {
              // Check if buffer is active at current timeline position
              // Note: buffer timestamps might be absolute, need to handle carefully
              const isActive = buffer.dealloc_time_ms === null ||
                              buffer.dealloc_time_ms > timeline.currentTime;

              return (
                <div
                  key={idx}
                  className={`p-3 rounded border ${
                    isActive ? 'border-green-600 bg-green-900/20' : 'border-gray-600 bg-gray-800/50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-white font-semibold text-sm">{buffer.name}</span>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      isActive ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                    }`}>
                      {isActive ? 'ACTIVE' : 'FREED'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                    <div>
                      <span className="text-gray-500">Size:</span>{' '}
                      <span className="text-white">{(buffer.size / (1024 * 1024)).toFixed(1)} MB</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Backend:</span>{' '}
                      <span className="text-white">{buffer.backend}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Usage:</span>{' '}
                      <span className="text-white">{buffer.usage_name}</span>
                    </div>
                    <div>
                      <span className="text-gray-500">Layer:</span>{' '}
                      <span className="text-white">
                        {buffer.layer !== 65535 ? `L${buffer.layer}` : 'Global'}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}

            {bufferTimeline.buffers.length === 0 && (
              <div className="text-gray-500 text-sm text-center py-8">
                No buffer events recorded
              </div>
            )}
          </div>
        ) : (
          <div className="p-4 text-gray-400 text-sm text-center py-8">
            No buffer timeline data
          </div>
        )}
      </div>
    </div>
  );
}
