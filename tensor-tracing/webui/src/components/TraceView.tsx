/**
 * View 2: Timeline & Trace Visualization
 *
 * Features:
 * - Timeline slider showing execution progress
 * - Play/pause/step controls for animation
 * - Trace table with virtual scrolling (210 entries)
 * - Click trace entry → highlight node in graph
 * - Sync with graph selection
 */

import { useEffect, useRef, useState } from 'react';
import { List } from 'react-window';
import { useAppStore } from '../stores/useAppStore';
import type { TraceEntry } from '../types/data';

interface TraceViewProps {
  isFullScreen: boolean;
}

export function TraceView({ isFullScreen }: TraceViewProps) {
  const listRef = useRef<any>(null);

  const {
    traceData,
    timeline,
    selectedTrace,
    selectTrace,
    setTimelinePosition,
    setFullScreen,
    correlationIndex,
    graphData,
  } = useAppStore();

  const [filteredEntries, setFilteredEntries] = useState<TraceEntry[]>([]);

  // Filter trace entries based on current filters
  useEffect(() => {
    if (!traceData) return;

    // For now, just use all entries
    // TODO: Apply filters from store
    setFilteredEntries(traceData.entries);
  }, [traceData]);

  // Auto-scroll to selected trace entry
  useEffect(() => {
    if (!selectedTrace || !listRef.current) return;

    const index = filteredEntries.findIndex(e => e.entry_id === selectedTrace.entryId);
    if (index >= 0) {
      listRef.current.scrollToRow({ index, align: 'center' });
    }
  }, [selectedTrace, filteredEntries]);

  if (!traceData) {
    return (
      <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex items-center justify-center">
        <span className="text-gray-500">No trace data loaded</span>
      </div>
    );
  }

  const maxTime = traceData.metadata.duration_ms;

  // Handle trace entry click
  const handleTraceClick = (entry: TraceEntry) => {
    selectTrace({
      entryId: entry.entry_id,
      timestamp_relative_ms: entry.timestamp_relative_ms,
      dst_name: entry.dst_name,
      sources: entry.sources,
    });

    // Jump timeline to this entry's timestamp
    setTimelinePosition(entry.timestamp_relative_ms);
  };

  // Render a single trace row
  const TraceRow = ({ index, style = {} }: { index: number; style?: React.CSSProperties }) => {
    const entry = filteredEntries[index];
    const isSelected = selectedTrace?.entryId === entry.entry_id;
    const isActive = timeline.currentTime >= entry.timestamp_relative_ms &&
                     (index === filteredEntries.length - 1 ||
                      timeline.currentTime < filteredEntries[index + 1]?.timestamp_relative_ms);

    // Calculate total size from all sources
    const totalSize = entry.sources.reduce((sum, src) => sum + src.size_bytes, 0);

    // Check if this trace entry has a corresponding graph node
    const hasGraphNode = correlationIndex && entry.sources.length > 0 && entry.sources.some(source => {
      // Check by address
      if (correlationIndex.addressToNode.has(source.tensor_ptr)) return true;
      // Check by name
      if (graphData) {
        return graphData.nodes.some(n => n.label === source.name);
      }
      return false;
    });

    // In full-screen mode, show detailed multi-line view
    if (isFullScreen) {
      return (
        <div
          style={style}
          onClick={() => handleTraceClick(entry)}
          className={`
            border-b-2 border-gray-800 cursor-pointer py-3 px-4
            ${isSelected ? 'bg-blue-900/50 border-blue-500' : 'hover:bg-gray-800'}
            ${isActive ? 'bg-amber-900/30' : ''}
          `}
          title={hasGraphNode ? 'Click to highlight in graph' : 'No corresponding graph node'}
        >
          {/* Header row: operation info */}
          <div className="flex items-center gap-3 mb-3">
            <span className={`text-xs ${hasGraphNode ? 'text-green-400' : 'text-gray-700'}`}>
              {hasGraphNode ? '●' : '○'}
            </span>
            <span className="text-gray-500 font-mono text-sm">#{entry.entry_id}</span>
            <span className="text-gray-300 font-mono text-sm">{entry.timestamp_relative_ms.toFixed(2)}ms</span>
            <span className="text-white font-bold text-sm">{entry.operation_type}</span>
            <span className="text-gray-400 text-sm">
              {entry.layer_id === null ? 'Layer: N/A' : `Layer: ${entry.layer_id}`}
            </span>
            <span className="text-gray-500 text-sm">Thread: {entry.thread_id}</span>
            <span className="text-purple-400 text-sm">{entry.phase}</span>
          </div>

          {/* Destination Tensor */}
          <div className="ml-6 mb-2">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-yellow-400 font-semibold text-xs">DESTINATION:</span>
              <span className="text-yellow-200 font-mono text-sm">{entry.dst_name}</span>
            </div>
          </div>

          {/* Sources: detailed list */}
          <div className="ml-6 space-y-1">
            <div className="text-blue-400 font-semibold text-xs mb-1">
              SOURCES ({entry.sources.length}):
            </div>
            {entry.sources.length === 0 ? (
              <div className="text-gray-500 text-xs italic ml-4">No source tensors</div>
            ) : (
              entry.sources.map((source, idx) => (
                <div key={idx} className="flex items-center gap-4 text-xs font-mono ml-4">
                  <span className="text-gray-500 w-8">[{idx}]</span>
                  <span className="text-white min-w-[200px]">{source.name}</span>
                  <span className={`px-2 py-0.5 rounded font-semibold ${
                    source.memory_source === 'DISK'
                      ? 'bg-blue-900 text-blue-300'
                      : 'bg-green-900 text-green-300'
                  }`}>
                    {source.memory_source}
                  </span>
                  <span className="text-gray-300 min-w-[60px]">{formatSize(source.size_bytes)}</span>
                  {source.memory_source === 'DISK' && source.disk_offset !== undefined && (
                    <span className="text-blue-400">offset: 0x{source.disk_offset.toString(16)}</span>
                  )}
                  {source.memory_source === 'BUFFER' && source.buffer_id !== undefined && (
                    <span className="text-green-400">buffer: 0x{source.buffer_id.toString(16)}</span>
                  )}
                  {source.layer_id !== null && (
                    <span className="text-purple-400">L{source.layer_id}</span>
                  )}
                  <span className="text-gray-600 text-[10px]">{source.tensor_ptr}</span>
                </div>
              ))
            )}
            <div className="text-gray-500 text-xs pt-1 ml-4">
              Total input size: {formatSize(totalSize)}
            </div>
          </div>
        </div>
      );
    }

    // Compact mode for small view
    const displayName = entry.sources.length > 0
      ? `${entry.sources[0].name}${entry.sources.length > 1 ? ` +${entry.sources.length - 1}` : ''}`
      : entry.dst_name || '<no sources>';

    const getSourceBadge = () => {
      if (entry.sources.length === 0) return null;
      const diskCount = entry.sources.filter(s => s.memory_source === 'DISK').length;
      const bufferCount = entry.sources.filter(s => s.memory_source === 'BUFFER').length;

      if (diskCount > 0 && bufferCount > 0) {
        return <span className="text-purple-400 text-xs">D+B</span>;
      } else if (diskCount > 0) {
        return <span className="text-blue-400 text-xs">DSK</span>;
      } else if (bufferCount > 0) {
        return <span className="text-green-400 text-xs">BUF</span>;
      }
      return null;
    };

    return (
      <div
        style={{ height: '32px', ...style }}
        onClick={() => handleTraceClick(entry)}
        className={`
          flex items-center gap-2 px-3 text-sm border-b border-gray-800 cursor-pointer
          ${isSelected ? 'bg-blue-900/50 border-blue-500' : 'hover:bg-gray-800'}
          ${isActive ? 'bg-amber-900/30' : ''}
        `}
        title={hasGraphNode ? 'Click to highlight in graph' : 'No corresponding graph node'}
      >
        <span className={`w-4 text-xs ${hasGraphNode ? 'text-green-400' : 'text-gray-700'}`}>
          {hasGraphNode ? '●' : '○'}
        </span>
        <span className="text-gray-500 w-12 text-right font-mono text-xs">
          {entry.entry_id}
        </span>
        <span className="text-gray-300 w-20 text-right font-mono text-xs">
          {entry.timestamp_relative_ms.toFixed(2)}ms
        </span>
        <span className="text-white w-32 font-mono text-xs truncate">
          {entry.operation_type}
        </span>
        <span className="text-gray-400 w-12 text-center font-mono text-xs">
          {entry.layer_id === null ? '-' : `L${entry.layer_id}`}
        </span>
        <span className="text-gray-300 flex-1 truncate font-mono text-xs">
          {displayName}
        </span>
        <span className="w-10 text-center">
          {getSourceBadge()}
        </span>
        <span className="text-gray-500 w-20 text-right font-mono text-xs">
          {formatSize(totalSize)}
        </span>
      </div>
    );
  };

  return (
    <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex flex-col">
      {/* Header */}
      <div className="h-12 border-b border-gray-700 flex items-center justify-between px-4">
        <div>
          <span className="text-white font-semibold">Timeline & Trace</span>
          <span className="ml-3 text-gray-400 text-sm">
            {filteredEntries.length} entries
          </span>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-gray-400 text-sm font-mono">
            {timeline.currentTime.toFixed(2)} / {maxTime.toFixed(2)} ms
          </div>
          {!isFullScreen && (
            <button
              onClick={() => setFullScreen('trace')}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
              title="Enter full-screen mode"
            >
              ⛶ Full Screen
            </button>
          )}
        </div>
      </div>

      {/* Timeline Controls */}
      <div className="border-b border-gray-700 p-3 space-y-3">
        {/* Timeline slider */}
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={0}
            max={maxTime}
            step={0.01}
            value={timeline.currentTime}
            onChange={(e) => setTimelinePosition(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-4
                       [&::-webkit-slider-thumb]:h-4
                       [&::-webkit-slider-thumb]:bg-blue-500
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:cursor-pointer"
          />
        </div>
      
      </div>

      {/* Table header */}
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 border-b border-gray-700 text-xs text-gray-400 font-semibold">
        <span className="w-4" title="Graph correlation indicator">🔗</span>
        <span className="w-12 text-right">#</span>
        <span className="w-20 text-right">Time (ms)</span>
        <span className="w-32">Operation</span>
        <span className="w-12 text-center">Layer</span>
        <span className="flex-1">Source Tensors</span>
        <span className="w-10 text-center">Mem</span>
        <span className="w-20 text-right">Size</span>
      </div>

      {/* Trace table with scrolling */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {filteredEntries.map((entry, index) => (
          <TraceRow key={entry.entry_id} index={index} style={{}} />
        ))}
      </div>
    </div>
  );
}

// Helper function to format bytes
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}
