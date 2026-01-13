/**
 * Heatmap Analysis Utilities
 *
 * Precision-focused analysis of tensor access patterns from trace logs.
 *
 * Key Requirements:
 * - Count exact access frequency per tensor
 * - Separate DISK vs BUFFER accesses
 * - Track both source (read) and destination (write) accesses
 * - Use actual byte offsets from data
 *
 * CRITICAL: Handle 19-character name truncation in trace sources!
 */

import type { TraceData, MemoryMap } from '../types/data';

/**
 * Helper: Find matching tensor in memory map considering 19-char truncation
 *
 * Trace source names are truncated to 19 characters, but memory map has full names.
 * This function finds the correct match.
 *
 * @param traceName - Name from trace (potentially truncated to 19 chars)
 * @param memoryMap - Memory map with full tensor names
 * @returns Full tensor name if found, or traceName if not in memory map
 */
function findMatchingTensorName(traceName: string, memoryMap: MemoryMap): string {
  // Try exact match first (for non-truncated names)
  const exactMatch = memoryMap.tensors.find(t => t.name === traceName);
  if (exactMatch) return exactMatch.name;

  // If traceName is exactly 19 characters, it might be truncated
  if (traceName.length === 19) {
    // Find tensors whose first 19 characters match
    const prefixMatch = memoryMap.tensors.find(t => t.name.startsWith(traceName));
    if (prefixMatch) return prefixMatch.name;
  }

  // Not in memory map - it's a runtime tensor
  return traceName;
}

/**
 * Access statistics for a single tensor
 */
export interface TensorAccessStats {
  tensorName: string;
  totalAccesses: number;
  sourceAccesses: number;    // Times used as a source (read)
  destAccesses: number;       // Times used as destination (write)
  diskAccesses: number;       // Accesses from DISK
  bufferAccesses: number;     // Accesses from BUFFER
  firstAccessTime: number;    // First access timestamp (ms)
  lastAccessTime: number;     // Last access timestamp (ms)
  offset_start?: number;      // For DISK tensors
  offset_end?: number;        // For DISK tensors
  size_bytes: number;
}

/**
 * Calculate access frequency for all tensors from trace data
 *
 * This is the foundation of the heatmap - we count how many times
 * each tensor is accessed throughout the entire inference.
 *
 * @param traceData - All trace entries
 * @param memoryMap - GGUF memory layout
 * @returns Map of tensor name to access statistics
 */
export function calculateTensorAccessFrequency(
  traceData: TraceData,
  memoryMap: MemoryMap
): Map<string, TensorAccessStats> {
  const stats = new Map<string, TensorAccessStats>();

  // Initialize stats for all tensors in memory map
  memoryMap.tensors.forEach(tensor => {
    stats.set(tensor.name, {
      tensorName: tensor.name,
      totalAccesses: 0,
      sourceAccesses: 0,
      destAccesses: 0,
      diskAccesses: 0,
      bufferAccesses: 0,
      firstAccessTime: Infinity,
      lastAccessTime: -Infinity,
      offset_start: tensor.offset_start,
      offset_end: tensor.offset_end,
      size_bytes: tensor.size_bytes,
    });
  });

  // Process all trace entries
  traceData.entries.forEach(entry => {
    const timestamp = entry.timestamp_relative_ms;

    // Process destination tensor (write access)
    // Match considering 19-char truncation
    const dstFullName = findMatchingTensorName(entry.dst_name, memoryMap);
    const dstStats = stats.get(dstFullName);

    if (dstStats) {
      dstStats.totalAccesses++;
      dstStats.destAccesses++;
      dstStats.firstAccessTime = Math.min(dstStats.firstAccessTime, timestamp);
      dstStats.lastAccessTime = Math.max(dstStats.lastAccessTime, timestamp);
    } else {
      // Destination is a runtime tensor not in memory map (e.g., intermediate results)
      stats.set(dstFullName, {
        tensorName: dstFullName,
        totalAccesses: 1,
        sourceAccesses: 0,
        destAccesses: 1,
        diskAccesses: 0,
        bufferAccesses: 0,
        firstAccessTime: timestamp,
        lastAccessTime: timestamp,
        size_bytes: 0, // Unknown size for runtime tensors
      });
    }

    // Process source tensors (read accesses)
    entry.sources.forEach(source => {
      // CRITICAL: Match considering 19-char truncation!
      const srcFullName = findMatchingTensorName(source.name, memoryMap);
      let srcStats = stats.get(srcFullName);

      if (!srcStats) {
        // Create entry for runtime tensor not in memory map
        srcStats = {
          tensorName: srcFullName,
          totalAccesses: 0,
          sourceAccesses: 0,
          destAccesses: 0,
          diskAccesses: 0,
          bufferAccesses: 0,
          firstAccessTime: Infinity,
          lastAccessTime: -Infinity,
          size_bytes: source.size_bytes,
        };
        stats.set(srcFullName, srcStats);
      }

      srcStats.totalAccesses++;
      srcStats.sourceAccesses++;
      srcStats.firstAccessTime = Math.min(srcStats.firstAccessTime, timestamp);
      srcStats.lastAccessTime = Math.max(srcStats.lastAccessTime, timestamp);

      // Track memory source
      if (source.memory_source === 'DISK') {
        srcStats.diskAccesses++;
      } else if (source.memory_source === 'BUFFER') {
        srcStats.bufferAccesses++;
      }
    });
  });

  return stats;
}

/**
 * Get maximum access count across all tensors
 * Used for normalizing heat colors (0 → max)
 */
export function getMaxAccessCount(stats: Map<string, TensorAccessStats>): number {
  let max = 0;
  stats.forEach(stat => {
    if (stat.totalAccesses > max) {
      max = stat.totalAccesses;
    }
  });
  return max;
}

/**
 * Filter stats to only DISK tensors (permanent GGUF tensors from memory map)
 *
 * IMPORTANT: Only returns tensors that exist in the memory map.
 * This excludes:
 * - Runtime tensors with truncated names
 * - Buffer-only intermediate tensors
 * - Tensors with disk_offset set incorrectly in trace
 *
 * @param stats - Access statistics map
 * @param memoryMap - GGUF memory map (source of truth)
 * @returns Only the permanent DISK tensors (should be 201 for TinyLlama)
 */
export function getDiskTensorStats(
  stats: Map<string, TensorAccessStats>,
  memoryMap: MemoryMap
): TensorAccessStats[] {
  const diskStats: TensorAccessStats[] = [];

  // Only include tensors that exist in memory map
  memoryMap.tensors.forEach(tensor => {
    const stat = stats.get(tensor.name);
    if (stat) {
      diskStats.push(stat);
    }
  });

  return diskStats;
}

/**
 * Filter stats to only BUFFER tensors (accessed from buffer, no disk offset)
 */
export function getBufferTensorStats(
  stats: Map<string, TensorAccessStats>
): TensorAccessStats[] {
  const bufferStats: TensorAccessStats[] = [];
  stats.forEach(stat => {
    if (stat.bufferAccesses > 0 && stat.offset_start === undefined) {
      bufferStats.push(stat);
    }
  });
  return bufferStats;
}

/**
 * Convert access count to heat color
 *
 * Scale: Gray (0) → Blue (low) → Yellow (medium) → Red (high)
 *
 * @param accessCount - Number of accesses
 * @param maxCount - Maximum access count (for normalization)
 * @returns RGB color string
 */
export function accessCountToHeatColor(accessCount: number, maxCount: number): string {
  if (accessCount === 0 || maxCount === 0) {
    return 'rgb(55, 65, 81)'; // gray-700
  }

  const intensity = accessCount / maxCount;

  if (intensity < 0.25) {
    // Blue (low)
    const r = Math.floor(59 + (96 * intensity * 4)); // 59 → 155
    const g = Math.floor(130 + (25 * intensity * 4)); // 130 → 155
    const b = 246; // Constant blue
    return `rgb(${r}, ${g}, ${b})`;
  } else if (intensity < 0.5) {
    // Blue → Green
    const t = (intensity - 0.25) * 4;
    const r = Math.floor(155 - (139 * t)); // 155 → 16
    const g = Math.floor(155 + (70 * t)); // 155 → 225
    const b = Math.floor(246 - (162 * t)); // 246 → 84
    return `rgb(${r}, ${g}, ${b})`;
  } else if (intensity < 0.75) {
    // Green → Yellow
    const t = (intensity - 0.5) * 4;
    const r = Math.floor(16 + (234 * t)); // 16 → 250
    const g = Math.floor(225 + (19 * t)); // 225 → 244
    const b = Math.floor(84 - (20 * t)); // 84 → 64
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // Yellow → Red
    const t = (intensity - 0.75) * 4;
    const r = Math.floor(250 - (11 * t)); // 250 → 239
    const g = Math.floor(244 - (176 * t)); // 244 → 68
    const b = Math.floor(64 - (15 * t)); // 64 → 49
    return `rgb(${r}, ${g}, ${b})`;
  }
}

/**
 * Format byte count for display
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';

  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

/**
 * Convert file offset to canvas X position
 *
 * This is CRITICAL for precision - we must map byte offsets to exact pixel positions.
 *
 * @param offset - Byte offset in file
 * @param totalFileSize - Total GGUF file size in bytes
 * @param canvasWidth - Canvas width in pixels
 * @returns X position in pixels
 */
export function offsetToCanvasX(
  offset: number,
  totalFileSize: number,
  canvasWidth: number
): number {
  return (offset / totalFileSize) * canvasWidth;
}

/**
 * Convert canvas X position to file offset
 * Used for click/hover detection
 */
export function canvasXToOffset(
  x: number,
  totalFileSize: number,
  canvasWidth: number
): number {
  return (x / canvasWidth) * totalFileSize;
}
