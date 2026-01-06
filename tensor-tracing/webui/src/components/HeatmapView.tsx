/**
 * View 3: Memory Heatmap Visualization
 *
 * MINIMIZED VIEW:
 * - Single horizontal bar (50px) showing linear memory layout
 * - Vertical lines marking tensor boundaries
 * - Hover tooltips with tensor details
 * - Toggle: Current Only vs Cumulative access highlighting
 *
 * FULL SCREEN VIEW:
 * - Vertically stacked bars (50px each), one per layer
 * - Scrollable if many layers
 * - Bottom split: trace log panel
 * - Per-layer distinct accesses, cumulative over time
 */

import { useEffect, useRef, useState, useMemo } from 'react';
import { useAppStore } from '../stores/useAppStore';
import type { MemoryTensor, TraceEntry } from '../types/data';

interface HoveredTensor {
  tensor: MemoryTensor;
  x: number;
  y: number;
  accessCount: number;
}

interface HeatmapViewProps {
  isFullScreen: boolean;
}

type AccessMode = 'current' | 'cumulative';

export function HeatmapView({ isFullScreen }: HeatmapViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const [hoveredTensor, setHoveredTensor] = useState<HoveredTensor | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 400 });
  const [accessMode, setAccessMode] = useState<AccessMode>('cumulative');
  const [showAccessCounts, setShowAccessCounts] = useState(false);
  const [selectedMemoryRegion, setSelectedMemoryRegion] = useState<string | null>(null);

  const { memoryMap, traceData, correlationIndex, selectedNode, selectNode, setFullScreen, timeline } = useAppStore();

  const BAR_HEIGHT = 50;
  const BAR_PADDING = 10;
  const LABEL_WIDTH = 40;

  /**
   * Calculate access counts for each tensor based on mode
   * - current: only accesses at current timeline position
   * - cumulative: all accesses from start to current time
   */
  const calculateAccessCounts = useMemo(() => {
    if (!traceData || !memoryMap) return new Map<string, number>();

    const counts = new Map<string, number>();
    const currentTime = timeline.currentTime;

    memoryMap.tensors.forEach(tensor => {
      let count = 0;

      traceData.entries.forEach(entry => {
        if (entry.tensor_name !== tensor.name) return;

        if (accessMode === 'current') {
          // Only count accesses within 10ms window of current time
          if (Math.abs(entry.timestamp_relative_ms - currentTime) < 10) {
            count++;
          }
        } else {
          // Cumulative: count all accesses up to current time
          if (entry.timestamp_relative_ms <= currentTime) {
            count++;
          }
        }
      });

      counts.set(tensor.name, count);
    });

    return counts;
  }, [traceData, memoryMap, timeline.currentTime, accessMode]);

  /**
   * Calculate per-layer access counts (for full-screen view)
   */
  const calculateLayerAccessCounts = useMemo(() => {
    if (!traceData || !memoryMap) return new Map<string, Map<number, number>>();

    // Map: tensor_name -> (layer_id -> count)
    const layerCounts = new Map<string, Map<number, number>>();
    const currentTime = timeline.currentTime;

    memoryMap.tensors.forEach(tensor => {
      layerCounts.set(tensor.name, new Map());
    });

    traceData.entries.forEach(entry => {
      if (entry.timestamp_relative_ms > currentTime) return;
      if (entry.layer_id === 65535) return; // Skip N/A layers

      const tensorCounts = layerCounts.get(entry.tensor_name);
      if (tensorCounts) {
        const layerCount = tensorCounts.get(entry.layer_id) || 0;
        tensorCounts.set(entry.layer_id, layerCount + 1);
      }
    });

    return layerCounts;
  }, [traceData, memoryMap, timeline.currentTime]);

  // Get max access count for color scaling
  const maxAccessCount = useMemo(() => {
    return Math.max(...Array.from(calculateAccessCounts.values()), 1);
  }, [calculateAccessCounts]);

  // Update canvas size on container resize
  useEffect(() => {
    if (!containerRef.current) return;

    const updateSize = () => {
      const rect = containerRef.current!.getBoundingClientRect();

      if (isFullScreen) {
        // Full screen: 70% for heatmap, 30% for trace log
        const heatmapHeight = Math.floor(rect.height * 0.7);
        setCanvasSize({ width: rect.width, height: heatmapHeight });
      } else {
        // Minimized: account for header
        setCanvasSize({ width: rect.width, height: rect.height - 60 });
      }
    };

    updateSize();

    const resizeObserver = new ResizeObserver(updateSize);
    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, [isFullScreen]);

  /**
   * Get color based on access intensity
   */
  const getHeatColor = (accessCount: number, maxCount: number): string => {
    if (accessCount === 0) return '#374151'; // gray-700 (no access)

    const intensity = accessCount / maxCount;

    if (intensity < 0.3) {
      return `rgb(59, 130, ${Math.floor(246 * (1 - intensity / 0.3))})`;  // blue
    } else if (intensity < 0.7) {
      const t = (intensity - 0.3) / 0.4;
      return `rgb(${Math.floor(59 + 196 * t)}, ${Math.floor(130 + 116 * t)}, 0)`;  // blue → yellow
    } else {
      const t = (intensity - 0.7) / 0.3;
      return `rgb(255, ${Math.floor(246 * (1 - t))}, 0)`;  // yellow → red
    }
  };

  /**
   * Draw minimized view: single 50px horizontal bar
   */
  const drawMinimizedView = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!memoryMap) return;

    const totalSize = memoryMap.total_size_bytes;
    const padding = LABEL_WIDTH + 20;
    const usableWidth = canvas.width - 2 * padding;
    const barY = (canvas.height - BAR_HEIGHT) / 2; // Center vertically

    // Draw tensors
    memoryMap.tensors.forEach(tensor => {
      const startX = padding + (tensor.offset_start / totalSize) * usableWidth;
      const endX = padding + (tensor.offset_end / totalSize) * usableWidth;
      const width = Math.max(endX - startX, 1);

      const accessCount = calculateAccessCounts.get(tensor.name) || 0;
      const color = getHeatColor(accessCount, maxAccessCount);

      // Draw tensor rectangle
      ctx.fillStyle = color;
      ctx.fillRect(startX, barY, width, BAR_HEIGHT);

      // Draw border
      ctx.strokeStyle = '#1f2937'; // gray-800
      ctx.lineWidth = 0.5;
      ctx.strokeRect(startX, barY, width, BAR_HEIGHT);

      // Draw access count if enabled
      if (showAccessCounts && width > 25 && accessCount > 0) {
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(accessCount.toString(), startX + width / 2, barY + BAR_HEIGHT / 2);
      }
    });

    // Draw vertical separator lines at tensor boundaries
    ctx.strokeStyle = '#4b5563'; // gray-600
    ctx.lineWidth = 1;
    memoryMap.tensors.forEach(tensor => {
      const endX = padding + (tensor.offset_end / totalSize) * usableWidth;
      ctx.beginPath();
      ctx.moveTo(endX, barY);
      ctx.lineTo(endX, barY + BAR_HEIGHT);
      ctx.stroke();
    });

    // Draw byte position labels
    ctx.fillStyle = '#9ca3af'; // gray-400
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    const numLabels = 5;
    for (let i = 0; i <= numLabels; i++) {
      const x = padding + (i / numLabels) * usableWidth;
      const bytes = (i / numLabels) * totalSize;
      const label = formatBytes(bytes);
      ctx.fillText(label, x, barY - 10);
    }
  };

  /**
   * Draw full-screen view: vertically stacked bars (one per layer)
   */
  const drawFullScreenView = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) => {
    if (!memoryMap) return;

    const numLayers = memoryMap.metadata.n_layers;
    const totalSize = memoryMap.total_size_bytes;
    const padding = 20;
    const usableWidth = canvas.width - LABEL_WIDTH - 2 * padding;

    // Set canvas height to accommodate all layers
    const totalHeight = numLayers * (BAR_HEIGHT + BAR_PADDING) + 2 * padding;
    if (canvas.height !== totalHeight) {
      canvas.height = totalHeight;
    }

    // Draw each layer
    for (let layer = 0; layer < numLayers; layer++) {
      const barY = padding + layer * (BAR_HEIGHT + BAR_PADDING);

      // Draw layer label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`L${layer}`, LABEL_WIDTH, barY + BAR_HEIGHT / 2);

      // Draw tensors for this layer
      memoryMap.tensors.filter(t => t.layer_id === layer).forEach(tensor => {
        const startX = LABEL_WIDTH + padding + (tensor.offset_start / totalSize) * usableWidth;
        const endX = LABEL_WIDTH + padding + (tensor.offset_end / totalSize) * usableWidth;
        const width = Math.max(endX - startX, 1);

        // Get access count for this specific layer
        const layerCounts = calculateLayerAccessCounts.get(tensor.name);
        const accessCount = layerCounts?.get(layer) || 0;
        const maxLayerCount = Math.max(...Array.from(calculateLayerAccessCounts.values()).flatMap(m => Array.from(m.values())), 1);
        const color = getHeatColor(accessCount, maxLayerCount);

        // Draw tensor rectangle
        ctx.fillStyle = color;
        ctx.fillRect(startX, barY, width, BAR_HEIGHT);

        // Draw border
        ctx.strokeStyle = '#1f2937';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(startX, barY, width, BAR_HEIGHT);

        // Draw access count
        if (showAccessCounts && width > 25 && accessCount > 0) {
          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 10px monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(accessCount.toString(), startX + width / 2, barY + BAR_HEIGHT / 2);
        }
      });

      // Draw vertical separator lines
      ctx.strokeStyle = '#4b5563';
      ctx.lineWidth = 1;
      memoryMap.tensors.filter(t => t.layer_id === layer).forEach(tensor => {
        const endX = LABEL_WIDTH + padding + (tensor.offset_end / totalSize) * usableWidth;
        ctx.beginPath();
        ctx.moveTo(endX, barY);
        ctx.lineTo(endX, barY + BAR_HEIGHT);
        ctx.stroke();
      });
    }

    // Draw byte position labels at top
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    const numLabels = 5;
    for (let i = 0; i <= numLabels; i++) {
      const x = LABEL_WIDTH + padding + (i / numLabels) * usableWidth;
      const bytes = (i / numLabels) * totalSize;
      const label = formatBytes(bytes);
      ctx.fillText(label, x, 12);
    }
  };

  // Main draw effect
  useEffect(() => {
    if (!canvasRef.current || !memoryMap || !traceData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution
    canvas.width = canvasSize.width;
    if (!isFullScreen) {
      canvas.height = canvasSize.height;
    }

    // Clear canvas
    ctx.fillStyle = '#111827'; // gray-900
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (isFullScreen) {
      drawFullScreenView(ctx, canvas);
    } else {
      drawMinimizedView(ctx, canvas);
    }
  }, [canvasSize, memoryMap, traceData, isFullScreen, calculateAccessCounts, calculateLayerAccessCounts, maxAccessCount, showAccessCounts]);

  // Handle mouse move for hover tooltips
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !memoryMap || !traceData) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const totalSize = memoryMap.total_size_bytes;
    const padding = isFullScreen ? 20 : LABEL_WIDTH + 20;
    const usableWidth = canvas.width - (isFullScreen ? LABEL_WIDTH : 0) - 2 * padding;

    let found = false;

    if (isFullScreen) {
      // Check which layer bar was hovered
      const numLayers = memoryMap.metadata.n_layers;
      for (let layer = 0; layer < numLayers; layer++) {
        const barY = padding + layer * (BAR_HEIGHT + BAR_PADDING);

        if (y >= barY && y <= barY + BAR_HEIGHT) {
          // Check tensors in this layer
          for (const tensor of memoryMap.tensors.filter(t => t.layer_id === layer)) {
            const startX = LABEL_WIDTH + padding + (tensor.offset_start / totalSize) * usableWidth;
            const endX = LABEL_WIDTH + padding + (tensor.offset_end / totalSize) * usableWidth;

            if (x >= startX && x <= endX) {
              const layerCounts = calculateLayerAccessCounts.get(tensor.name);
              const accessCount = layerCounts?.get(layer) || 0;

              setHoveredTensor({
                tensor,
                x: e.clientX,
                y: e.clientY,
                accessCount,
              });
              canvas.style.cursor = 'pointer';
              found = true;
              break;
            }
          }
        }
        if (found) break;
      }
    } else {
      // Minimized view: single bar
      const barY = (canvas.height - BAR_HEIGHT) / 2;

      if (y >= barY && y <= barY + BAR_HEIGHT) {
        for (const tensor of memoryMap.tensors) {
          const startX = padding + (tensor.offset_start / totalSize) * usableWidth;
          const endX = padding + (tensor.offset_end / totalSize) * usableWidth;

          if (x >= startX && x <= endX) {
            const accessCount = calculateAccessCounts.get(tensor.name) || 0;

            setHoveredTensor({
              tensor,
              x: e.clientX,
              y: e.clientY,
              accessCount,
            });
            canvas.style.cursor = 'pointer';
            found = true;
            break;
          }
        }
      }
    }

    if (!found) {
      setHoveredTensor(null);
      canvas.style.cursor = 'default';
    }
  };

  // Handle click to select memory region (for trace log filtering)
  const handleClick = () => {
    if (!hoveredTensor) return;

    setSelectedMemoryRegion(hoveredTensor.tensor.name);

    // Also select node in graph if correlation exists
    if (correlationIndex) {
      const node = Array.from(correlationIndex.addressToNode.values()).find(
        n => n.label === hoveredTensor.tensor.name
      );

      if (node) {
        selectNode({
          nodeId: node.id,
          address: node.address,
          label: node.label,
          layer_id: node.layer_id,
        });
      }
    }
  };

  // Highlight memory region when graph node is selected
  useEffect(() => {
    if (selectedNode && selectedNode.label) {
      setSelectedMemoryRegion(selectedNode.label);
    }
  }, [selectedNode]);

  // Filter trace entries for selected memory region
  const filteredTraceEntries = useMemo(() => {
    if (!traceData || !selectedMemoryRegion) return [];

    return traceData.entries.filter(entry => entry.tensor_name === selectedMemoryRegion);
  }, [traceData, selectedMemoryRegion]);

  if (!memoryMap) {
    return (
      <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex items-center justify-center">
        <span className="text-gray-500">No memory map loaded</span>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex flex-col">
      {/* Header */}
      <div className="h-12 border-b border-gray-700 flex items-center justify-between px-4">
        <div>
          <span className="text-white font-semibold">Memory Heatmap</span>
          <span className="ml-3 text-gray-400 text-sm">
            {memoryMap.tensors.length} tensors, {formatBytes(memoryMap.total_size_bytes)}
          </span>
        </div>

        <div className="flex items-center gap-4">
          {/* Access mode toggle */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-xs">Mode:</span>
            <div className="flex gap-1">
              <button
                onClick={() => setAccessMode('current')}
                className={`px-2 py-1 text-xs rounded ${
                  accessMode === 'current'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                }`}
                title="Show only current accesses at timeline position"
              >
                Current Only
              </button>
              <button
                onClick={() => setAccessMode('cumulative')}
                className={`px-2 py-1 text-xs rounded ${
                  accessMode === 'cumulative'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                }`}
                title="Show cumulative accesses from start to current time"
              >
                Cumulative
              </button>
            </div>
          </div>

          {/* Show access counts toggle */}
          <button
            onClick={() => setShowAccessCounts(!showAccessCounts)}
            className={`px-2 py-1 text-xs rounded ${
              showAccessCounts
                ? 'bg-amber-600 hover:bg-amber-500 text-white'
                : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
            }`}
            title="Toggle access count numbers"
          >
            {showAccessCounts ? '# On' : '# Off'}
          </button>

          {!isFullScreen && (
            <button
              onClick={() => setFullScreen('heatmap')}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
              title="Enter full-screen mode"
            >
              ⛶ Full Screen
            </button>
          )}

          {/* Legend */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-400">Heat:</span>
            <div className="flex items-center gap-1">
              <div className="w-6 h-3 bg-gray-700"></div>
              <span className="text-gray-500">None</span>
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

      {/* Canvas container */}
      <div className={`flex-1 ${isFullScreen ? 'overflow-y-auto' : ''}`} ref={scrollContainerRef}>
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredTensor(null)}
          onClick={handleClick}
          className="w-full"
        />

        {/* Hover tooltip */}
        {hoveredTensor && (
          <div
            className="fixed bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg pointer-events-none z-10"
            style={{
              left: hoveredTensor.x + 10,
              top: hoveredTensor.y + 10,
            }}
          >
            <div className="text-white font-semibold text-sm mb-2">
              {hoveredTensor.tensor.name}
            </div>
            <div className="space-y-1 text-xs">
              <div>
                <span className="text-gray-400">Category:</span>{' '}
                <span className="text-white">{hoveredTensor.tensor.category}</span>
              </div>
              <div>
                <span className="text-gray-400">Layer:</span>{' '}
                <span className="text-white">
                  {hoveredTensor.tensor.layer_id !== null ? hoveredTensor.tensor.layer_id : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Size:</span>{' '}
                <span className="text-white font-mono">
                  {formatBytes(hoveredTensor.tensor.size_bytes)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Offset:</span>{' '}
                <span className="text-white font-mono">
                  0x{hoveredTensor.tensor.offset_start.toString(16)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Shape:</span>{' '}
                <span className="text-white font-mono">
                  [{hoveredTensor.tensor.shape.join(', ')}]
                </span>
              </div>
              <div>
                <span className="text-gray-400">Accesses:</span>{' '}
                <span className="text-white font-semibold">
                  {hoveredTensor.accessCount}
                </span>
              </div>
            </div>
            <div className="mt-2 text-xs text-gray-500 italic">
              Click to view trace entries
            </div>
          </div>
        )}
      </div>

      {/* Trace log panel (full-screen only) */}
      {isFullScreen && (
        <div className="h-[30%] border-t border-gray-700 flex flex-col bg-gray-800">
          <div className="h-10 border-b border-gray-700 flex items-center justify-between px-4">
            <span className="text-white font-semibold">
              Trace Log {selectedMemoryRegion && `- ${selectedMemoryRegion}`}
            </span>
            <span className="text-gray-400 text-sm">
              {filteredTraceEntries.length} entries
            </span>
          </div>

          {/* Trace entries */}
          <div className="flex-1 overflow-y-auto custom-scrollbar">
            {selectedMemoryRegion ? (
              filteredTraceEntries.length > 0 ? (
                <div className="text-xs">
                  {filteredTraceEntries.map((entry, idx) => (
                    <div
                      key={idx}
                      className="flex items-center gap-3 px-3 py-2 border-b border-gray-700 hover:bg-gray-700 cursor-pointer"
                    >
                      <span className="text-gray-500 w-12 text-right font-mono">
                        {entry.entry_id}
                      </span>
                      <span className="text-gray-300 w-20 text-right font-mono">
                        {entry.timestamp_relative_ms.toFixed(2)}ms
                      </span>
                      <span className="text-white w-24 font-mono">
                        {entry.operation_type}
                      </span>
                      <span className="text-gray-400 w-12 text-center font-mono">
                        L{entry.layer_id === 65535 ? '-' : entry.layer_id}
                      </span>
                      <span className="text-gray-500 w-20 text-right font-mono">
                        {formatBytes(entry.size_bytes)}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  No trace entries for selected tensor
                </div>
              )
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                Click a memory region to view trace entries
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Helper to format bytes
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)}GB`;
}
