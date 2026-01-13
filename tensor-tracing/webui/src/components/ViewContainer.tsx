/**
 * ViewContainer - Draggable wrapper for views
 *
 * Provides drag & drop functionality and close button for each view
 */

import { useState } from 'react';
import type { ViewType } from '../stores/useAppStore';

interface ViewContainerProps {
  viewType: ViewType;
  onClose: () => void;
  onDragStart: (viewType: ViewType) => void;
  onDragEnd: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (targetView: ViewType) => void;
  children: React.ReactNode;
}

export function ViewContainer({
  viewType,
  onClose,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
  children,
}: ViewContainerProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragStart = (e: React.DragEvent) => {
    setIsDragging(true);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', viewType);
    onDragStart(viewType);
  };

  const handleDragEnd = () => {
    setIsDragging(false);
    onDragEnd();
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setIsDragOver(true);
    onDragOver(e);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    onDrop(viewType);
  };

  const viewTitles: Record<ViewType, string> = {
    graph: 'Computation Graph',
    trace: 'Trace & Timeline',
    heatmap: 'Memory Heatmap',
  };

  return (
    <div
      className={`relative h-full ${isDragging ? 'opacity-50' : ''} ${
        isDragOver ? 'ring-2 ring-blue-500' : ''
      }`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Drag handle header - ONLY THIS IS DRAGGABLE */}
      <div
        className="absolute top-0 left-0 right-0 h-8 bg-gray-800/50 border-b border-gray-700 flex items-center justify-between px-3 cursor-move z-10"
        draggable
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
      >
        <span className="text-xs text-gray-400 font-medium">
          ⋮⋮ {viewTitles[viewType]}
        </span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-red-400 text-sm"
          title={`Close ${viewTitles[viewType]}`}
        >
          ✕
        </button>
      </div>

      {/* View content - NOT DRAGGABLE (so sliders work!) */}
      <div className="h-full pt-8">
        {children}
      </div>
    </div>
  );
}
