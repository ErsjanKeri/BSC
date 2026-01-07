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
  fullScreenView: 'graph' | 'trace' | 'heatmap' | 'transformer' | null  // Which view is full-screen

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
  setFullScreen: (view: 'graph' | 'trace' | 'heatmap' | 'transformer' | null) => void
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

    // Build address → node mapping
    const addressToNode = new Map<string, typeof graphData.nodes[0]>();
    graphData.nodes.forEach(node => {
      addressToNode.set(node.address, node);
    });

    // Build address → trace entries mapping (NEW: handle multi-source entries)
    // Each source tensor in an entry gets mapped to that entry
    const addressToTraces = new Map<string, typeof traceData.entries>();
    traceData.entries.forEach(entry => {
      // Map each source tensor to this entry
      entry.sources.forEach(source => {
        const existing = addressToTraces.get(source.tensor_ptr) || [];
        addressToTraces.set(source.tensor_ptr, [...existing, entry]);
      });
    });

    // Build node ID → memory tensor mapping (by tensor name)
    const nodeToMemory = new Map<string, typeof memoryMap.tensors[0]>();
    const tensorNameMap = new Map(memoryMap.tensors.map(t => [t.name, t]));
    graphData.nodes.forEach(node => {
      const memTensor = tensorNameMap.get(node.label);
      if (memTensor) {
        nodeToMemory.set(node.id, memTensor);
      }
    });

    // Build layer → nodes mapping
    const layerToNodes = new Map<number, typeof graphData.nodes>();
    graphData.nodes.forEach(node => {
      if (node.layer_id !== null) {
        const existing = layerToNodes.get(node.layer_id) || [];
        layerToNodes.set(node.layer_id, [...existing, node]);
      }
    });

    // Build time index for animation (NEW: get addresses from sources)
    const timestamps = traceData.entries.map(e => e.timestamp_relative_ms);
    const activeNodes = traceData.entries.map(entry => {
      // Get all nodes involved in this entry (from all sources)
      const nodeIds: string[] = [];
      entry.sources.forEach(source => {
        const node = addressToNode.get(source.tensor_ptr);
        if (node) {
          nodeIds.push(node.id);
        }
      });
      return nodeIds;
    });

    const correlationIndex: CorrelationIndex = {
      addressToNode,
      addressToTraces,
      nodeToMemory,
      layerToNodes,
      timeIndex: {
        timestamps,
        activeNodes,
      },
    };

    set({ correlationIndex });
    console.log('Correlation index built:', {
      nodes: addressToNode.size,
      traces: addressToTraces.size,
      layers: layerToNodes.size,
    });
  },

  // ========================================================================
  // Selection Actions
  // ========================================================================

  selectNode: (node) => {
    set({ selectedNode: node });

    // If a node is selected, also update timeline to first trace of this node
    if (node && get().correlationIndex) {
      const traces = get().correlationIndex!.addressToTraces.get(node.address);
      if (traces && traces.length > 0) {
        set({
          timeline: {
            ...get().timeline,
            currentTime: traces[0].timestamp_relative_ms,
          },
        });
      }
    }
  },

  selectTrace: (trace) => {
    set({ selectedTrace: trace });

    // If a trace is selected, try to highlight corresponding node in graph
    if (trace && trace.sources && trace.sources.length > 0 && get().correlationIndex) {
      const index = get().correlationIndex!;
      let foundNode = null;

      // Strategy 1: Try to find by address (from any source)
      for (const source of trace.sources) {
        const node = index.addressToNode.get(source.tensor_ptr);
        if (node) {
          foundNode = node;
          console.log('✓ Trace→Graph correlation (by address):', {
            trace_entry: trace.entryId,
            source: source.name,
            graph_node: node.label,
            node_id: node.id,
          });
          break;
        }
      }

      // Strategy 2: Try to find by tensor name (fallback)
      if (!foundNode && get().graphData) {
        const graphData = get().graphData!;
        for (const source of trace.sources) {
          // Try exact match
          const matchByName = graphData.nodes.find(n => n.label === source.name);
          if (matchByName) {
            foundNode = matchByName;
            console.log('✓ Trace→Graph correlation (by name):', {
              trace_entry: trace.entryId,
              source: source.name,
              graph_node: matchByName.label,
              node_id: matchByName.id,
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
      } else {
        console.log('⚠ Trace→Graph correlation failed:', {
          trace_entry: trace.entryId,
          sources: trace.sources.map(s => s.name),
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
}));
