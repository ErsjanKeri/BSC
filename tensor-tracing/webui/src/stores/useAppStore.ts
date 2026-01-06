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

    // Build address → trace entries mapping
    const addressToTraces = new Map<string, typeof traceData.entries>();
    traceData.entries.forEach(entry => {
      const existing = addressToTraces.get(entry.tensor_ptr) || [];
      addressToTraces.set(entry.tensor_ptr, [...existing, entry]);
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

    // Build time index for animation
    const timestamps = traceData.entries.map(e => e.timestamp_relative_ms);
    const activeNodes = traceData.entries.map(entry => {
      const node = addressToNode.get(entry.tensor_ptr);
      return node ? [node.id] : [];
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

    // If a trace is selected, highlight corresponding node
    if (trace && get().correlationIndex) {
      const node = get().correlationIndex!.addressToNode.get(trace.tensor_ptr);
      if (node) {
        set({
          selectedNode: {
            nodeId: node.id,
            address: node.address,
            label: node.label,
            layer_id: node.layer_id,
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
}));
