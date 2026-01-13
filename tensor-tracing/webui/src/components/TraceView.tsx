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
    correlationIndex,
    graphData,
  } = useAppStore();

  const [filteredEntries, setFilteredEntries] = useState<TraceEntry[]>([]);
  const [hoveredEntry, setHoveredEntry] = useState<{ entry: TraceEntry; x: number; y: number } | null>(null);
  const [hoveredSources, setHoveredSources] = useState<{ entry: TraceEntry; x: number; y: number } | null>(null);
  const hoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const sourcesHoverTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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

  // Handle hover with delay
  const handleRowHover = (entry: TraceEntry, event: React.MouseEvent) => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);

    hoverTimeoutRef.current = setTimeout(() => {
      const rect = event.currentTarget.getBoundingClientRect();
      setHoveredEntry({
        entry,
        x: rect.left + 20,
        y: rect.top,
      });
    }, 200); // 200ms delay
  };

  const handleRowLeave = () => {
    if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
    setHoveredEntry(null);
  };

  // Handle sources column hover
  const handleSourcesHover = (entry: TraceEntry, event: React.MouseEvent) => {
    if (sourcesHoverTimeoutRef.current) clearTimeout(sourcesHoverTimeoutRef.current);

    sourcesHoverTimeoutRef.current = setTimeout(() => {
      const rect = event.currentTarget.getBoundingClientRect();
      setHoveredSources({
        entry,
        x: rect.left,
        y: rect.bottom + 8,
      });
    }, 150);
  };

  const handleSourcesLeave = () => {
    if (sourcesHoverTimeoutRef.current) clearTimeout(sourcesHoverTimeoutRef.current);
    setHoveredSources(null);
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (hoverTimeoutRef.current) clearTimeout(hoverTimeoutRef.current);
      if (sourcesHoverTimeoutRef.current) clearTimeout(sourcesHoverTimeoutRef.current);
    };
  }, []);

  // Render a single trace row - UNIFIED COMPACT DESIGN
  const TraceRow = ({ index, style = {} }: { index: number; style?: React.CSSProperties }) => {
    const entry = filteredEntries[index];
    const isSelected = selectedTrace?.entryId === entry.entry_id;
    const isActive = timeline.currentTime >= entry.timestamp_relative_ms &&
                     (index === filteredEntries.length - 1 ||
                      timeline.currentTime < filteredEntries[index + 1]?.timestamp_relative_ms);

    // Calculate total size from all sources
    const totalSize = entry.sources.reduce((sum, src) => sum + src.size_bytes, 0);

    // Check if this trace entry has a corresponding graph node (NAME-BASED)
    const hasGraphNode = correlationIndex && entry.sources.length > 0 && entry.sources.some(source => {
      // Check by name using correlation index
      return correlationIndex.nameToGraphNode.has(source.name);
    });

    // Format sources for display: "first_source +N"
    const sourcesDisplay = entry.sources.length === 0
      ? '<no sources>'
      : entry.sources.length === 1
      ? entry.sources[0].name
      : `${entry.sources[0].name} +${entry.sources.length - 1}`;

    // Memory badge
    const diskCount = entry.sources.filter(s => s.memory_source === 'DISK').length;
    const bufferCount = entry.sources.filter(s => s.memory_source === 'BUFFER').length;
    const memBadge = diskCount > 0 && bufferCount > 0 ? 'D+B'
                   : diskCount > 0 ? 'DSK'
                   : bufferCount > 0 ? 'BUF'
                   : null;

    // UNIFIED COMPACT ROW (works for both 50% and 100% width)
    return (
      <div
        style={{ height: '26px', ...style }}
        onClick={() => handleTraceClick(entry)}
        onMouseEnter={(e) => handleRowHover(entry, e)}
        onMouseLeave={handleRowLeave}
        className={`
          flex items-center gap-2 px-3 text-xs font-mono border-b border-gray-800 cursor-pointer transition-colors
          ${isSelected ? 'bg-blue-900/50 border-l-2 border-l-blue-500' : 'hover:bg-gray-800'}
          ${isActive ? 'bg-amber-900/30' : ''}
        `}
        title="Hover for details • Click to select"
      >
        {/* Correlation indicator */}
        <span className={`w-4 ${hasGraphNode ? 'text-green-400' : 'text-gray-700'}`}>
          {hasGraphNode ? '●' : '○'}
        </span>

        {/* Entry ID */}
        <span className="w-10 text-right text-gray-500">
          {entry.entry_id}
        </span>

        {/* Timestamp */}
        <span className="w-16 text-right text-gray-300">
          {entry.timestamp_relative_ms.toFixed(1)}
        </span>

        {/* Operation */}
        <span className="w-24 text-white truncate" title={entry.operation_type}>
          {entry.operation_type}
        </span>

        {/* DESTINATION */}
        <span className="flex-1 min-w-[140px] text-yellow-200 truncate" title={entry.dst_name}>
          {entry.dst_name}
        </span>

        {/* Sources - Hoverable! */}
        <span
          className="flex-1 min-w-[160px] text-blue-200 truncate cursor-help hover:text-blue-100 hover:underline"
          title="Hover to see all sources"
          onMouseEnter={(e) => {
            e.stopPropagation();
            handleSourcesHover(entry, e);
          }}
          onMouseLeave={(e) => {
            e.stopPropagation();
            handleSourcesLeave();
          }}
        >
          {sourcesDisplay}
        </span>

        {/* Layer */}
        <span className="w-10 text-center text-gray-400" title={entry.layer_id === null ? 'No layer' : `Layer ${entry.layer_id}`}>
          {entry.layer_id === null ? '-' : `L${entry.layer_id}`}
        </span>

        {/* Memory badge */}
        <span className="w-10 text-center">
          {memBadge && (
            <span className={`px-1 py-0.5 rounded text-[10px] font-semibold ${
              memBadge === 'DSK' ? 'bg-blue-900 text-blue-300' :
              memBadge === 'BUF' ? 'bg-green-900 text-green-300' :
              'bg-purple-900 text-purple-300'
            }`}>
              {memBadge}
            </span>
          )}
        </span>

        {/* Size */}
        <span className="w-16 text-right text-gray-500">
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
      <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 border-b border-gray-700 text-xs text-gray-400 font-semibold uppercase tracking-wide">
        <span className="w-4" title="Graph correlation">🔗</span>
        <span className="w-10 text-right">#</span>
        <span className="w-16 text-right">Time</span>
        <span className="w-24">Operation</span>
        <span className="flex-1 min-w-[140px]">Destination</span>
        <span className="flex-1 min-w-[160px]">Sources</span>
        <span className="w-10 text-center">Lyr</span>
        <span className="w-10 text-center">Mem</span>
        <span className="w-16 text-right">Size</span>
      </div>

      {/* Trace table with scrolling */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {filteredEntries.map((entry, index) => (
          <TraceRow key={entry.entry_id} index={index} style={{}} />
        ))}
      </div>

      {/* Hover Tooltip - Beautiful Detailed Card */}
      {hoveredEntry && (
        <div
          className="fixed z-50 w-[480px] bg-gray-900/98 border-2 border-gray-600 rounded-lg shadow-2xl p-4 space-y-3 pointer-events-none"
          style={{
            left: `${Math.min(hoveredEntry.x, window.innerWidth - 500)}px`,
            top: `${hoveredEntry.y}px`,
          }}
        >
          {/* Destination Section */}
          <div>
            <div className="text-yellow-400 font-semibold text-xs mb-1">DESTINATION:</div>
            <div className="text-yellow-200 font-mono text-sm">{hoveredEntry.entry.dst_name}</div>
          </div>

          {/* Sources Section */}
          <div>
            <div className="text-blue-400 font-semibold text-xs mb-1">
              SOURCES ({hoveredEntry.entry.sources.length}):
            </div>
            {hoveredEntry.entry.sources.length === 0 ? (
              <div className="text-gray-500 text-xs italic ml-4">No source tensors</div>
            ) : (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {hoveredEntry.entry.sources.map((source, idx) => (
                  <div key={idx} className="ml-4">
                    <div className="text-white font-mono text-xs mb-1">
                      [{idx}] {source.name}
                    </div>
                    <div className="flex items-center gap-3 text-[10px] ml-4 flex-wrap">
                      <span className={`px-2 py-0.5 rounded font-semibold ${
                        source.memory_source === 'DISK'
                          ? 'bg-blue-900 text-blue-300'
                          : 'bg-green-900 text-green-300'
                      }`}>
                        {source.memory_source}
                      </span>
                      <span className="text-gray-300">{formatSize(source.size_bytes)}</span>
                      {source.memory_source === 'DISK' && source.disk_offset !== undefined && (
                        <span className="text-blue-400">offset: 0x{source.disk_offset.toString(16)}</span>
                      )}
                      {source.memory_source === 'BUFFER' && source.buffer_id !== undefined && (
                        <span className="text-green-400">buffer: 0x{source.buffer_id.toString(16)}</span>
                      )}
                      {source.layer_id !== null && (
                        <span className="text-purple-400">L{source.layer_id}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Total Size */}
          <div className="text-gray-500 text-xs border-t border-gray-700 pt-2">
            Total input size: {formatSize(hoveredEntry.entry.sources.reduce((sum, s) => sum + s.size_bytes, 0))}
          </div>
        </div>
      )}

      {/* Sources Hover Tooltip - Beautiful Detailed Card (Same as your screenshot!) */}
      {hoveredSources && (
        <div
          className="fixed z-50 w-[520px] bg-blue-950/98 border-2 border-blue-600 rounded-lg shadow-2xl p-4 space-y-3 pointer-events-none"
          style={{
            left: `${Math.min(hoveredSources.x, window.innerWidth - 540)}px`,
            top: `${hoveredSources.y}px`,
          }}
        >
          {/* Destination Section */}
          <div>
            <div className="text-yellow-400 font-semibold text-xs mb-1">DESTINATION:</div>
            <div className="text-yellow-200 font-mono text-base">{hoveredSources.entry.dst_name}</div>
          </div>

          {/* Sources Section - Full Details */}
          <div>
            <div className="text-blue-400 font-semibold text-xs mb-2">
              SOURCES ({hoveredSources.entry.sources.length}):
            </div>
            {hoveredSources.entry.sources.length === 0 ? (
              <div className="text-gray-500 text-xs italic ml-4">No source tensors</div>
            ) : (
              <div className="space-y-3 max-h-80 overflow-y-auto pr-2">
                {hoveredSources.entry.sources.map((source, idx) => (
                  <div key={idx} className="ml-4">
                    {/* Source name with index */}
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-gray-500 text-xs">[{idx}]</span>
                      <span className="text-white font-mono text-sm">{source.name}</span>
                    </div>

                    {/* Source metadata */}
                    <div className="flex items-center gap-3 ml-6 flex-wrap">
                      <span className={`px-2 py-1 rounded font-semibold text-xs ${
                        source.memory_source === 'DISK'
                          ? 'bg-blue-900 text-blue-300'
                          : 'bg-green-900 text-green-300'
                      }`}>
                        {source.memory_source}
                      </span>
                      <span className="text-white text-xs">{formatSize(source.size_bytes)}</span>
                      {source.memory_source === 'DISK' && source.disk_offset !== undefined && (
                        <span className="text-blue-400 text-xs font-mono">
                          offset: 0x{source.disk_offset.toString(16)}
                        </span>
                      )}
                      {source.memory_source === 'BUFFER' && source.buffer_id !== undefined && (
                        <span className="text-green-400 text-xs font-mono">
                          buffer: 0x{source.buffer_id.toString(16)}
                        </span>
                      )}
                      {source.layer_id !== null && (
                        <span className="text-purple-400 text-xs">L{source.layer_id}</span>
                      )}
                      <span className="text-gray-600 text-[10px] font-mono">{source.tensor_ptr}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Total Size */}
          <div className="text-gray-500 text-xs border-t border-gray-700 pt-2">
            Total input size: {formatSize(hoveredSources.entry.sources.reduce((sum, s) => sum + s.size_bytes, 0))}
          </div>
        </div>
      )}
    </div>
  );
}

// Helper function to format bytes
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}
