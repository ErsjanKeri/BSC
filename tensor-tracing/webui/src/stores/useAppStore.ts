/**
 * Zustand Store - Global Application State
 *
 * Centralized state management for all 4 views:
 * - Graph View
 * - Trace View
 * - Memory Heatmap
 * - 3D Transformer
 */

import { create } from 'zustand';
import type {
  GraphData,
  TraceData,
  MemoryMap,
  BufferTimeline,
  CorrelationIndex,
  SelectedNode,
  SelectedTrace,
  TimelineState,
  FilterState,
} from '../types/data';

// ============================================================================
// Helper Functions for Name-Based Correlation
// ============================================================================

/**
 * Normalize tensor name by removing common suffixes.
 *
 * Examples:
 *   "Qcur-0 (view)" → "Qcur-0"
 *   "cache_k_l0 (view) (permuted)" → "cache_k_l0"
 *   "blk.5.attn_q.weight" → "blk.5.attn_q.weight" (unchanged)
 *
 * @param name - Tensor name to normalize
 * @returns Normalized name (base name without suffixes)
 */
function normalizeTensorName(name: string): string {
  if (!name) return '';

  // Remove common suffixes: (view), (reshaped), (permuted), (copy)
  return name
    .replace(/\s*\(view\)/g, '')
    .replace(/\s*\(reshaped\)/g, '')
    .replace(/\s*\(permuted\)/g, '')
    .replace(/\s*\(copy\)/g, '')
    .trim();
}

/**
 * Try to match a tensor name from trace with a graph node.
 * Handles name variations due to views, reshapes, etc.
 *
 * @param traceName - Tensor name from trace
 * @param graphNodes - Array of graph nodes
 * @returns Matching graph node or null
 */
function findMatchingGraphNode(traceName: string, graphNodes: any[]): any {
  // Strategy 1: Exact match
  const exactMatch = graphNodes.find(n => n.label === traceName);
  if (exactMatch) return exactMatch;

  // Strategy 2: Normalized match (strip suffixes)
  const normalizedTrace = normalizeTensorName(traceName);
  const normalizedMatch = graphNodes.find(n =>
    normalizeTensorName(n.label) === normalizedTrace
  );
  if (normalizedMatch) return normalizedMatch;

  // Strategy 3: Prefix match (trace name might be longer with suffixes)
  const prefixMatch = graphNodes.find(n =>
    traceName.startsWith(n.label) || n.label.startsWith(traceName)
  );
  if (prefixMatch) return prefixMatch;

  return null;
}

// View types for layout management
export type ViewType = 'graph' | 'trace' | 'heatmap';

// Heatmap visualization modes
export type HeatmapMode = 'total-accumulated' | 'current-layer';

interface AppStore {
  // ========================================================================
  // Data State
  // ========================================================================
  memoryMap: MemoryMap | null
  bufferTimeline: BufferTimeline | null
  currentTokenId: number
  graphData: GraphData | null
  traceData: TraceData | null
  correlationIndex: CorrelationIndex | null

  // ========================================================================
  // UI State
  // ========================================================================
  selectedNode: SelectedNode | null
  selectedTrace: SelectedTrace | null
  hoveredNode: string | null           // Node ID being hovered
  timeline: TimelineState
  filters: FilterState
  fullScreenView: ViewType | null  // Which view is full-screen

  // Layout state for drag & drop
  visibleViews: ViewType[]         // Which views are currently shown
  viewOrder: ViewType[]            // Order of views for rendering

  // Heatmap visualization mode
  heatmapMode: HeatmapMode         // 'total-accumulated' | 'current-layer'

  // ========================================================================
  // Loading State
  // ========================================================================
  isLoading: boolean
  loadingError: string | null

  // ========================================================================
  // Actions - Data Loading
  // ========================================================================
  loadMemoryMap: () => Promise<void>
  loadBufferStats: () => Promise<void>
  loadTokenData: (tokenId: number) => Promise<void>
  buildCorrelationIndex: () => void

  // ========================================================================
  // Actions - Selection & Interaction
  // ========================================================================
  selectNode: (node: SelectedNode | null) => void
  selectTrace: (trace: SelectedTrace | null) => void
  setHoveredNode: (nodeId: string | null) => void

  // ========================================================================
  // Actions - Timeline Control
  // ========================================================================
  setTimelinePosition: (time: number) => void
  playTimeline: () => void
  pauseTimeline: () => void
  setPlaybackSpeed: (speed: number) => void
  resetTimeline: () => void

  // ========================================================================
  // Actions - Filtering
  // ========================================================================
  setLayerFilter: (layer: number | null) => void
  setCategoryFilter: (category: string | null) => void
  setSearchTerm: (term: string) => void
  clearFilters: () => void

  // ========================================================================
  // Actions - Full-Screen Mode
  // ========================================================================
  setFullScreen: (view: ViewType | null) => void

  // ========================================================================
  // Actions - Layout Management
  // ========================================================================
  toggleView: (view: ViewType) => void       // Show/hide a view
  reorderViews: (newOrder: ViewType[]) => void  // Reorder views via drag & drop

  // ========================================================================
  // Actions - Heatmap Mode
  // ========================================================================
  setHeatmapMode: (mode: HeatmapMode) => void  // Toggle between heatmap modes
}

export const useAppStore = create<AppStore>((set, get) => ({
  // ========================================================================
  // Initial State
  // ========================================================================
  memoryMap: null,
  bufferTimeline: null,
  currentTokenId: 0,
  graphData: null,
  traceData: null,
  correlationIndex: null,

  selectedNode: null,
  selectedTrace: null,
  hoveredNode: null,

  timeline: {
    currentTime: 0,
    isPlaying: false,
    playbackSpeed: 1.0,
  },

  filters: {
    selectedLayer: null,
    selectedCategory: null,
    searchTerm: '',
  },

  fullScreenView: null,

  // Layout state - all 3 views visible by default in order: graph, trace, heatmap
  visibleViews: ['graph', 'trace', 'heatmap'],
  viewOrder: ['graph', 'trace', 'heatmap'],

  // Heatmap mode - default to total accumulated
  heatmapMode: 'total-accumulated',

  isLoading: false,
  loadingError: null,

  // ========================================================================
  // Data Loading Actions
  // ========================================================================

  loadMemoryMap: async () => {
    set({ isLoading: true, loadingError: null });
    try {
      const response = await fetch('/data/memory-map.json');
      if (!response.ok) throw new Error('Failed to load memory map');
      const memoryMap = await response.json();
      set({ memoryMap, isLoading: false });
    } catch (error) {
      set({
        loadingError: `Failed to load memory map: ${error}`,
        isLoading: false,
      });
    }
  },

  loadBufferStats: async () => {
    set({ isLoading: true, loadingError: null });
    try {
      const response = await fetch('/data/buffer-timeline.json');
      if (!response.ok) throw new Error('Failed to load buffer stats');
      const bufferTimeline = await response.json();
      set({ bufferTimeline, isLoading: false });
      console.log('Buffer timeline loaded:', {
        buffers: bufferTimeline.metadata.total_buffers,
        events: bufferTimeline.metadata.total_events,
        peak_mb: bufferTimeline.metadata.peak_occupancy_mb,
      });
    } catch (error) {
      set({
        loadingError: `Failed to load buffer stats: ${error}`,
        isLoading: false,
      });
    }
  },

  loadTokenData: async (tokenId: number) => {
    set({ isLoading: true, loadingError: null });
    try {
      // Load graph data
      const graphResponse = await fetch(`/data/graphs/token-${String(tokenId).padStart(5, '0')}.json`);
      if (!graphResponse.ok) throw new Error(`Failed to load graph for token ${tokenId}`);
      const graphData = await graphResponse.json();

      // Load trace data
      const traceResponse = await fetch(`/data/traces/token-${String(tokenId).padStart(5, '0')}.json`);
      if (!traceResponse.ok) throw new Error(`Failed to load trace for token ${tokenId}`);
      const traceData = await traceResponse.json();

      set({
        currentTokenId: tokenId,
        graphData,
        traceData,
        isLoading: false,
      });

      // Build correlation index after data loads
      get().buildCorrelationIndex();

    } catch (error) {
      set({
        loadingError: `Failed to load token data: ${error}`,
        isLoading: false,
      });
    }
  },

  buildCorrelationIndex: () => {
    const { graphData, traceData, memoryMap } = get();

    if (!graphData || !traceData || !memoryMap) {
      console.warn('Cannot build correlation index: missing data');
      return;
    }

    console.log('Building NAME-BASED correlation index...');

    // Build tensor name → graph node mapping
    const nameToGraphNode = new Map<string, typeof graphData.nodes[0]>();
    graphData.nodes.forEach(node => {
      // Store both exact name and normalized name
      nameToGraphNode.set(node.label, node);
      const normalized = normalizeTensorName(node.label);
      if (normalized !== node.label) {
        // Also map normalized name (for fuzzy matching)
        nameToGraphNode.set(normalized, node);
      }
    });

    // Build tensor name → trace entries mapping (NAME-BASED, not address!)
    // Each source tensor name maps to all trace entries that accessed it
    const nameToTraces = new Map<string, typeof traceData.entries>();
    traceData.entries.forEach(entry => {
      // Map destination tensor
      const dstName = entry.dst_name;
      if (dstName) {
        const existing = nameToTraces.get(dstName) || [];
        nameToTraces.set(dstName, [...existing, entry]);

        // Also map normalized name
        const normalized = normalizeTensorName(dstName);
        if (normalized !== dstName) {
          const existingNorm = nameToTraces.get(normalized) || [];
          nameToTraces.set(normalized, [...existingNorm, entry]);
        }
      }

      // Map each source tensor to this entry
      entry.sources.forEach(source => {
        const srcName = source.name;
        if (srcName) {
          const existing = nameToTraces.get(srcName) || [];
          nameToTraces.set(srcName, [...existing, entry]);

          // Also map normalized name
          const normalized = normalizeTensorName(srcName);
          if (normalized !== srcName) {
            const existingNorm = nameToTraces.get(normalized) || [];
            nameToTraces.set(normalized, [...existingNorm, entry]);
          }
        }
      });
    });

    // Build tensor name → memory tensor mapping (GGUF weights only)
    const nameToMemory = new Map<string, typeof memoryMap.tensors[0]>();
    memoryMap.tensors.forEach(tensor => {
      nameToMemory.set(tensor.name, tensor);
    });

    // Build layer → nodes mapping (unchanged)
    const layerToNodes = new Map<number, typeof graphData.nodes>();
    graphData.nodes.forEach(node => {
      if (node.layer_id !== null) {
        const existing = layerToNodes.get(node.layer_id) || [];
        layerToNodes.set(node.layer_id, [...existing, node]);
      }
    });

    // Build time index for animation (NAME-BASED)
    const timestamps = traceData.entries.map(e => e.timestamp_relative_ms);
    const activeNodeNames = traceData.entries.map(entry => {
      // Get all tensor names involved in this entry
      const names: string[] = [];

      // Add destination
      if (entry.dst_name) {
        names.push(entry.dst_name);
      }

      // Add all sources
      entry.sources.forEach(source => {
        if (source.name) {
          names.push(source.name);
        }
      });

      return names;
    });

    const correlationIndex: CorrelationIndex = {
      nameToGraphNode,
      nameToTraces,
      nameToMemory,
      layerToNodes,
      timeIndex: {
        timestamps,
        activeNodeNames,
      },
    };

    set({ correlationIndex });
    console.log('✓ NAME-BASED correlation index built:', {
      graphNodes: nameToGraphNode.size,
      traceEntries: nameToTraces.size,
      memoryTensors: nameToMemory.size,
      layers: layerToNodes.size,
    });
  },

  // ========================================================================
  // Selection Actions
  // ========================================================================

  selectNode: (node) => {
    set({ selectedNode: node });

    // If a node is selected, also update timeline to first trace of this tensor
    if (node && get().correlationIndex) {
      const index = get().correlationIndex!;

      // Try to find traces for this tensor by name
      const traces = index.nameToTraces.get(node.label);
      if (traces && traces.length > 0) {
        set({
          timeline: {
            ...get().timeline,
            currentTime: traces[0].timestamp_relative_ms,
          },
        });
        console.log(`✓ Found ${traces.length} traces for node '${node.label}'`);
      } else {
        // Try normalized name
        const normalized = normalizeTensorName(node.label);
        const tracesNorm = index.nameToTraces.get(normalized);
        if (tracesNorm && tracesNorm.length > 0) {
          set({
            timeline: {
              ...get().timeline,
              currentTime: tracesNorm[0].timestamp_relative_ms,
            },
          });
          console.log(`✓ Found ${tracesNorm.length} traces for normalized '${normalized}'`);
        } else {
          console.log(`⚠️  No traces found for '${node.label}'`);
        }
      }
    }
  },

  selectTrace: (trace) => {
    set({ selectedTrace: trace });

    // If a trace is selected, try to highlight corresponding node in graph
    if (trace && get().correlationIndex) {
      const index = get().correlationIndex!;
      let foundNode = null;

      // Try to find graph node by NAME (check destination first, then sources)
      // Check destination tensor
      if (trace.dst_name) {
        foundNode = index.nameToGraphNode.get(trace.dst_name);
        if (foundNode) {
          console.log('✓ Trace→Graph correlation (dst exact):', {
            trace_entry: trace.entryId,
            dst_name: trace.dst_name,
            graph_node: foundNode.label,
          });
        }
      }

      // Check sources if destination not found
      if (!foundNode && trace.sources) {
        for (const source of trace.sources) {
          // Try exact name match
          foundNode = index.nameToGraphNode.get(source.name);
          if (foundNode) {
            console.log('✓ Trace→Graph correlation (source exact):', {
              trace_entry: trace.entryId,
              source: source.name,
              graph_node: foundNode.label,
            });
            break;
          }

          // Try normalized match
          const normalized = normalizeTensorName(source.name);
          foundNode = index.nameToGraphNode.get(normalized);
          if (foundNode) {
            console.log('✓ Trace→Graph correlation (source normalized):', {
              trace_entry: trace.entryId,
              source: source.name,
              normalized,
              graph_node: foundNode.label,
            });
            break;
          }
        }
      }

      // If found, select the node
      if (foundNode) {
        set({
          selectedNode: {
            nodeId: foundNode.id,
            address: foundNode.address,
            label: foundNode.label,
            layer_id: foundNode.layer_id,
          },
        });
      }
    }
  },

  setHoveredNode: (nodeId) => {
    set({ hoveredNode: nodeId });
  },

  // ========================================================================
  // Timeline Actions
  // ========================================================================

  setTimelinePosition: (time) => {
    set({
      timeline: {
        ...get().timeline,
        currentTime: time,
      },
    });
  },

  playTimeline: () => {
    set({
      timeline: {
        ...get().timeline,
        isPlaying: true,
      },
    });
  },

  pauseTimeline: () => {
    set({
      timeline: {
        ...get().timeline,
        isPlaying: false,
      },
    });
  },

  setPlaybackSpeed: (speed) => {
    set({
      timeline: {
        ...get().timeline,
        playbackSpeed: speed,
      },
    });
  },

  resetTimeline: () => {
    set({
      timeline: {
        currentTime: 0,
        isPlaying: false,
        playbackSpeed: 1.0,
      },
    });
  },

  // ========================================================================
  // Filter Actions
  // ========================================================================

  setLayerFilter: (layer) => {
    set({
      filters: {
        ...get().filters,
        selectedLayer: layer,
      },
    });
  },

  setCategoryFilter: (category) => {
    set({
      filters: {
        ...get().filters,
        selectedCategory: category,
      },
    });
  },

  setSearchTerm: (term) => {
    set({
      filters: {
        ...get().filters,
        searchTerm: term,
      },
    });
  },

  clearFilters: () => {
    set({
      filters: {
        selectedLayer: null,
        selectedCategory: null,
        searchTerm: '',
      },
    });
  },

  // ========================================================================
  // Full-Screen Actions
  // ========================================================================

  setFullScreen: (view) => {
    set({ fullScreenView: view });
  },

  // ========================================================================
  // Layout Management Actions
  // ========================================================================

  toggleView: (view) => {
    const { visibleViews } = get();

    if (visibleViews.includes(view)) {
      // Remove view
      const newVisible = visibleViews.filter(v => v !== view);
      set({ visibleViews: newVisible });
      console.log(`✓ View '${view}' hidden. Visible: [${newVisible.join(', ')}]`);
    } else {
      // Add view back
      const newVisible = [...visibleViews, view];
      set({ visibleViews: newVisible });
      console.log(`✓ View '${view}' shown. Visible: [${newVisible.join(', ')}]`);
    }
  },

  reorderViews: (newOrder) => {
    set({ viewOrder: newOrder });
    console.log(`✓ Views reordered: [${newOrder.join(', ')}]`);
  },

  // ========================================================================
  // Heatmap Mode Actions
  // ========================================================================

  setHeatmapMode: (mode) => {
    set({ heatmapMode: mode });
    console.log(`✓ Heatmap mode changed to: ${mode}`);
  },
}));
