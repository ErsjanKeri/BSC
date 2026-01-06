/**
 * View 3: Memory Heatmap Visualization
 *
 * Canvas-based visualization showing:
 * - GGUF file memory layout (tensors positioned by offset)
 * - Access frequency heatmap
 * - Hover tooltips with tensor details
 * - Click to highlight in graph
 */

import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../stores/useAppStore';
import type { MemoryTensor } from '../types/data';

interface HoveredTensor {
  tensor: MemoryTensor;
  x: number;
  y: number;
  accessCount: number;
}

export function HeatmapView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredTensor, setHoveredTensor] = useState<HoveredTensor | null>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 400 });

  const { memoryMap, traceData, correlationIndex, selectNode } = useAppStore();

  // Update canvas size on container resize
  useEffect(() => {
    if (!containerRef.current) return;

    const updateSize = () => {
      const rect = containerRef.current!.getBoundingClientRect();
      setCanvasSize({ width: rect.width, height: rect.height - 60 }); // Account for header
    };

    updateSize();

    const resizeObserver = new ResizeObserver(updateSize);
    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, []);

  // Draw heatmap
  useEffect(() => {
    if (!canvasRef.current || !memoryMap || !traceData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    // Clear canvas
    ctx.fillStyle = '#111827'; // gray-900
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const totalSize = memoryMap.total_size_bytes;
    const padding = 20;
    const usableWidth = canvas.width - 2 * padding;
    const usableHeight = canvas.height - 2 * padding;

    // Calculate access counts per tensor
    const accessCounts = new Map<string, number>();
    if (correlationIndex) {
      memoryMap.tensors.forEach(tensor => {
        let count = 0;
        correlationIndex.addressToTraces.forEach(traces => {
          traces.forEach(trace => {
            if (trace.tensor_name === tensor.name) {
              count++;
            }
          });
        });
        accessCounts.set(tensor.name, count);
      });
    }

    const maxAccessCount = Math.max(...Array.from(accessCounts.values()), 1);

    // Draw tensors as rectangles
    memoryMap.tensors.forEach(tensor => {
      const startX = padding + (tensor.offset_start / totalSize) * usableWidth;
      const endX = padding + (tensor.offset_end / totalSize) * usableWidth;
      const width = Math.max(endX - startX, 1); // At least 1px wide

      // Map layer_id to vertical position
      const maxLayers = memoryMap.metadata.n_layers;
      let yPos: number;

      if (tensor.layer_id !== null) {
        yPos = padding + (tensor.layer_id / maxLayers) * usableHeight;
      } else {
        // Non-layer tensors go at the bottom
        yPos = padding + usableHeight - 20;
      }

      const height = 15;

      // Get access count and calculate heat color
      const accessCount = accessCounts.get(tensor.name) || 0;
      const intensity = accessCount / maxAccessCount;

      // Color based on access frequency (blue → yellow → red)
      let color: string;
      if (intensity === 0) {
        color = '#374151'; // gray-700 (no access)
      } else if (intensity < 0.3) {
        color = `rgb(59, 130, ${Math.floor(246 * (1 - intensity / 0.3))})`;  // blue
      } else if (intensity < 0.7) {
        const t = (intensity - 0.3) / 0.4;
        color = `rgb(${Math.floor(59 + 196 * t)}, ${Math.floor(130 + 116 * t)}, 0)`;  // blue → yellow
      } else {
        const t = (intensity - 0.7) / 0.3;
        color = `rgb(255, ${Math.floor(246 * (1 - t))}, 0)`;  // yellow → red
      }

      ctx.fillStyle = color;
      ctx.fillRect(startX, yPos, width, height);

      // Border
      ctx.strokeStyle = '#1f2937'; // gray-800
      ctx.lineWidth = 0.5;
      ctx.strokeRect(startX, yPos, width, height);
    });

    // Draw labels
    ctx.fillStyle = '#9ca3af'; // gray-400
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    // File position labels (top)
    const numLabels = 5;
    for (let i = 0; i <= numLabels; i++) {
      const x = padding + (i / numLabels) * usableWidth;
      const bytes = (i / numLabels) * totalSize;
      const label = formatBytes(bytes);
      ctx.fillText(label, x, 12);
    }

    // Layer labels (left)
    const maxLayers = memoryMap.metadata.n_layers;
    for (let layer = 0; layer < maxLayers; layer += 5) {
      const y = padding + (layer / maxLayers) * usableHeight;
      ctx.fillText(`L${layer}`, 10, y + 10);
    }

  }, [canvasSize, memoryMap, traceData, correlationIndex]);

  // Handle mouse move for hover tooltips
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !memoryMap || !traceData) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const totalSize = memoryMap.total_size_bytes;
    const padding = 20;
    const usableWidth = canvas.width - 2 * padding;
    const usableHeight = canvas.height - 2 * padding;

    // Calculate access counts
    const accessCounts = new Map<string, number>();
    if (correlationIndex) {
      memoryMap.tensors.forEach(tensor => {
        let count = 0;
        correlationIndex.addressToTraces.forEach(traces => {
          traces.forEach(trace => {
            if (trace.tensor_name === tensor.name) {
              count++;
            }
          });
        });
        accessCounts.set(tensor.name, count);
      });
    }

    // Find hovered tensor
    const maxLayers = memoryMap.metadata.n_layers;

    for (const tensor of memoryMap.tensors) {
      const startX = padding + (tensor.offset_start / totalSize) * usableWidth;
      const endX = padding + (tensor.offset_end / totalSize) * usableWidth;

      let yPos: number;
      if (tensor.layer_id !== null) {
        yPos = padding + (tensor.layer_id / maxLayers) * usableHeight;
      } else {
        yPos = padding + usableHeight - 20;
      }

      const height = 15;

      if (x >= startX && x <= endX && y >= yPos && y <= yPos + height) {
        setHoveredTensor({
          tensor,
          x: e.clientX,
          y: e.clientY,
          accessCount: accessCounts.get(tensor.name) || 0,
        });
        canvas.style.cursor = 'pointer';
        return;
      }
    }

    setHoveredTensor(null);
    canvas.style.cursor = 'default';
  };

  // Handle click to select node in graph
  const handleClick = () => {
    if (!hoveredTensor || !correlationIndex) return;

    // Find node by tensor name
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
  };

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

        {/* Legend */}
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-400">Access frequency:</span>
          <div className="flex items-center gap-1">
            <div className="w-8 h-3 bg-gray-700"></div>
            <span className="text-gray-500">None</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-8 h-3 bg-blue-500"></div>
            <span className="text-gray-500">Low</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-8 h-3 bg-yellow-500"></div>
            <span className="text-gray-500">Med</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-8 h-3 bg-red-500"></div>
            <span className="text-gray-500">High</span>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div className="flex-1 relative">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredTensor(null)}
          onClick={handleClick}
          className="w-full h-full"
        />

        {/* Hover tooltip */}
        {hoveredTensor && (
          <div
            className="absolute bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg pointer-events-none z-10"
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
              Click to select in graph
            </div>
          </div>
        )}
      </div>
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
