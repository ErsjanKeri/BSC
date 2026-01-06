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
  const animationRef = useRef<number | null>(null);

  const {
    traceData,
    timeline,
    selectedTrace,
    selectTrace,
    setTimelinePosition,
    playTimeline,
    pauseTimeline,
    setPlaybackSpeed,
    setFullScreen,
  } = useAppStore();

  const [filteredEntries, setFilteredEntries] = useState<TraceEntry[]>([]);

  // Filter trace entries based on current filters
  useEffect(() => {
    if (!traceData) return;

    // For now, just use all entries
    // TODO: Apply filters from store
    setFilteredEntries(traceData.entries);
  }, [traceData]);

  // Animation loop
  useEffect(() => {
    if (!timeline.isPlaying || !traceData) return;

    const maxTime = traceData.metadata.duration_ms;
    const frameRate = 60; // 60 FPS
    const increment = (1000 / frameRate) * timeline.playbackSpeed;

    const animate = () => {
      setTimelinePosition((prev) => {
        const newTime = prev + increment;
        if (newTime >= maxTime) {
          pauseTimeline();
          return maxTime;
        }
        return newTime;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [timeline.isPlaying, timeline.playbackSpeed, traceData, setTimelinePosition, pauseTimeline]);

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
      tensor_ptr: entry.tensor_ptr,
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

    return (
      <div
        style={{ height: '32px', ...style }}
        onClick={() => handleTraceClick(entry)}
        className={`
          flex items-center gap-3 px-3 text-sm border-b border-gray-800 cursor-pointer
          ${isSelected ? 'bg-blue-900/50 border-blue-500' : 'hover:bg-gray-800'}
          ${isActive ? 'bg-amber-900/30' : ''}
        `}
      >
        <span className="text-gray-500 w-12 text-right font-mono text-xs">
          {entry.entry_id}
        </span>
        <span className="text-gray-300 w-20 text-right font-mono text-xs">
          {entry.timestamp_relative_ms.toFixed(2)}ms
        </span>
        <span className="text-white w-24 font-mono text-xs">
          {entry.operation_type}
        </span>
        <span className="text-gray-400 w-12 text-center font-mono text-xs">
          L{entry.layer_id === 65535 ? '-' : entry.layer_id}
        </span>
        <span className="text-gray-300 flex-1 truncate font-mono text-xs">
          {entry.tensor_name || '<anonymous>'}
        </span>
        <span className="text-gray-500 w-20 text-right font-mono text-xs">
          {formatSize(entry.size_bytes)}
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

        {/* Playback controls */}
        <div className="flex items-center gap-2">
          {/* Play/Pause */}
          <button
            onClick={() => timeline.isPlaying ? pauseTimeline() : playTimeline()}
            className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 text-white rounded flex items-center gap-2"
          >
            {timeline.isPlaying ? '⏸' : '▶'} {timeline.isPlaying ? 'Pause' : 'Play'}
          </button>

          {/* Step backward */}
          <button
            onClick={() => {
              const currentIndex = filteredEntries.findIndex(
                e => e.timestamp_relative_ms > timeline.currentTime
              );
              const prevIndex = Math.max(0, currentIndex - 1);
              if (filteredEntries[prevIndex]) {
                setTimelinePosition(filteredEntries[prevIndex].timestamp_relative_ms);
              }
            }}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded"
            title="Previous entry"
          >
            ⏮
          </button>

          {/* Step forward */}
          <button
            onClick={() => {
              const nextEntry = filteredEntries.find(
                e => e.timestamp_relative_ms > timeline.currentTime
              );
              if (nextEntry) {
                setTimelinePosition(nextEntry.timestamp_relative_ms);
              }
            }}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded"
            title="Next entry"
          >
            ⏭
          </button>

          {/* Reset */}
          <button
            onClick={() => setTimelinePosition(0)}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded"
            title="Reset to start"
          >
            ⏮⏮
          </button>

          {/* Playback speed */}
          <div className="flex items-center gap-2 ml-4">
            <span className="text-gray-400 text-sm">Speed:</span>
            {[0.5, 1, 2, 4].map(speed => (
              <button
                key={speed}
                onClick={() => setPlaybackSpeed(speed)}
                className={`
                  px-2 py-1 text-sm rounded
                  ${timeline.playbackSpeed === speed
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }
                `}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table header */}
      <div className="flex items-center gap-3 px-3 py-2 bg-gray-800 border-b border-gray-700 text-xs text-gray-400 font-semibold">
        <span className="w-12 text-right">#</span>
        <span className="w-20 text-right">Time (ms)</span>
        <span className="w-24">Operation</span>
        <span className="w-12 text-center">Layer</span>
        <span className="flex-1">Tensor Name</span>
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
