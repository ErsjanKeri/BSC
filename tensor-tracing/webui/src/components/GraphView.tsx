/**
 * View 1: Computation Graph Visualization
 *
 * Interactive graph viewer using Cytoscape.js with:
 * - Color-coded nodes by category
 * - Click to select and inspect nodes
 * - Sync with timeline (highlight active nodes)
 * - Zoom, pan, fit controls
 * - Layer filtering
 */

import { useEffect, useRef, useState } from 'react';
import cytoscape from 'cytoscape';
import type { Core, NodeSingular } from 'cytoscape';
import dagre from 'cytoscape-dagre';
import { useAppStore } from '../stores/useAppStore';

// Register dagre layout extension
cytoscape.use(dagre);

// Category color scheme (matching Tailwind config)
const CATEGORY_COLORS = {
  input: '#3b82f6',      // blue-500
  attention: '#10b981',  // green-500
  ffn: '#f59e0b',        // amber-500
  norm: '#8b5cf6',       // violet-500
  output: '#ef4444',     // red-500
  other: '#6b7280',      // gray-500
};

interface GraphViewProps {
  isFullScreen: boolean;
}

export function GraphView({ isFullScreen }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [selectedNodeInfo, setSelectedNodeInfo] = useState<any>(null);

  // Layer filtering: 'all' or specific layer number (0, 1, 2, ...)
  // Defaults to Layer 0 for small screen, 'all' for full screen
  const [selectedLayer, setSelectedLayer] = useState<number | 'all'>(0);

  const {
    graphData,
    selectedNode,
    selectNode,
    setHoveredNode,
    timeline,
    correlationIndex,
    filters,
    setFullScreen,
  } = useAppStore();

  // Auto-switch layer view based on full-screen mode
  useEffect(() => {
    if (isFullScreen) {
      // Full screen: show all layers
      setSelectedLayer('all');
    } else {
      // Small screen: show Layer 0
      setSelectedLayer(0);
    }
  }, [isFullScreen]);

  /**
   * Get nodes for a specific layer, including connected N/A nodes and previous layer outputs
   *
   * Algorithm:
   * 1. Get all nodes with layer_id === targetLayer
   * 2. Get all N/A nodes (layer_id === null) that have edges to/from layer nodes
   * 3. Get all nodes from layer N-1 that feed into layer N (predecessors)
   * 4. Return combined set
   */
  // Get available layers from graph data
  const availableLayers = graphData
    ? Array.from(new Set(graphData.nodes.map(n => n.layer_id).filter(l => l !== null))).sort((a, b) => a! - b!)
    : [];

  const getFilteredNodesForLayer = (targetLayer: number | 'all') => {
    if (!graphData) return { nodes: [], edges: [] };

    // If 'all', return everything
    if (targetLayer === 'all') {
      return {
        nodes: graphData.nodes,
        edges: graphData.edges,
      };
    }

    // Step 1: Get all nodes from target layer
    const layerNodes = graphData.nodes.filter(n => n.layer_id === targetLayer);
    const layerNodeIds = new Set(layerNodes.map(n => n.id));

    // Step 2: Get N/A nodes connected to this layer
    const connectedNA = graphData.nodes.filter(node => {
      // Only consider N/A nodes
      if (node.layer_id !== null) return false;

      // Check if this N/A node has any edge to/from a layer node
      return graphData.edges.some(edge =>
        (edge.source === node.id && layerNodeIds.has(edge.target)) ||
        (edge.target === node.id && layerNodeIds.has(edge.source))
      );
    });

    // Step 3: Get previous layer outputs that feed into current layer
    // Only if not Layer 0 (no previous layer)
    const previousLayerOutputs = targetLayer > 0
      ? graphData.nodes.filter(node => {
          // Only consider nodes from previous layer
          if (node.layer_id !== targetLayer - 1) return false;

          // Include if this node has an edge TO current layer
          return graphData.edges.some(edge =>
            edge.source === node.id && layerNodeIds.has(edge.target)
          );
        })
      : [];

    // Step 4: Combine all
    const filteredNodes = [...layerNodes, ...connectedNA, ...previousLayerOutputs];
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));

    // Step 5: Get edges - only include edges where BOTH endpoints are visible
    // We can't render edges to non-existent nodes in Cytoscape
    const filteredEdges = graphData.edges.filter(edge =>
      filteredNodeIds.has(edge.source) && filteredNodeIds.has(edge.target)
    );

    return {
      nodes: filteredNodes,
      edges: filteredEdges,
    };
  };

  // Initialize Cytoscape instance
  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    // Destroy existing instance
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    // Apply layer filtering using our new logic
    // selectedLayer determines what to show (Layer 0 by default, or 'all', or specific layer)
    const { nodes: nodesToShow, edges: edgesToShow } = getFilteredNodesForLayer(selectedLayer);

    // Debug logging
    if (selectedLayer !== 'all') {
      const layerNodeCount = nodesToShow.filter(n => n.layer_id === selectedLayer).length;
      const naNodeCount = nodesToShow.filter(n => n.layer_id === null).length;
      const prevLayerNodeCount = selectedLayer > 0 ? nodesToShow.filter(n => n.layer_id === (selectedLayer as number) - 1).length : 0;

      console.log(
        `Layer ${selectedLayer}: ${nodesToShow.length} nodes total ` +
        `(${layerNodeCount} layer nodes, ${naNodeCount} N/A nodes, ${prevLayerNodeCount} from prev layer), ` +
        `${edgesToShow.length} edges`
      );
    } else {
      console.log(`All layers: ${nodesToShow.length} nodes, ${edgesToShow.length} edges`);
    }

      // Create Cytoscape instance
      const cy = cytoscape({
        container: containerRef.current,
        style: [
        // Node styles
        {
          selector: 'node',
          style: {
            'background-color': (ele: NodeSingular) => {
              const category = ele.data('category') as string;
              return CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS] || CATEGORY_COLORS.other;
            },
            'label': 'data(label)',
            'color': '#ffffff',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '10px',
            'width': 40,
            'height': 40,
            'border-width': 2,
            'border-color': (ele: NodeSingular) => {
              // Use alternating border colors for visual layer separation
              const layerId = ele.data('layer_id') as number | null;
              if (layerId !== null && layerId % 2 === 1) {
                return '#fbbf24'; // amber-400 for odd layers
              }
              return '#1f2937'; // gray-800 for even layers
            },
          },
        },

        // Selected node style
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#ffffff',
          },
        },

        // Highlighted node (during timeline animation)
        {
          selector: 'node.highlighted',
          style: {
            'border-width': 4,
            'border-color': '#fbbf24', // amber-400
            'background-opacity': 1,
          },
        },

        // Dimmed node (when filtering)
        {
          selector: 'node.dimmed',
          style: {
            'opacity': 0.3,
          },
        },

        // Edge styles
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#374151', // gray-700
            'target-arrow-color': '#374151',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1,
          },
        },

        // Edge label
        {
          selector: 'edge[label]',
          style: {
            'label': 'data(label)',
            'font-size': '8px',
            'color': '#9ca3af', // gray-400
            'text-background-color': '#111827', // gray-900
            'text-background-opacity': 0.8,
            'text-background-padding': '2px',
          },
        },
      ],

      layout: {
        name: 'dagre',  // Hierarchical top-to-bottom layout
        rankDir: 'TB',  // Top-to-bottom
        ranker: 'tight-tree',  // Compact hierarchical layout
        nodeSep: 50,    // Horizontal spacing between nodes
        rankSep: 100,   // Vertical spacing between ranks (layers)
        padding: 50,
        animate: false,  // Disable animation for faster rendering
      } as any,  // dagre layout options not in base types

      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.2,
    });

    // Load filtered graph data
    if (nodesToShow.length > 0) {
      // Add nodes
      const nodes = nodesToShow.map(node => ({
        data: {
          id: node.id,
          label: node.label,
          address: node.address,
          operation: node.operation,
          shape: node.shape,
          dtype: node.dtype,
          layer_id: node.layer_id,
          category: node.category,
        },
      }));

      // Add edges
      const edges = edgesToShow.map(edge => ({
        data: {
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          label: edge.label,
        },
      }));

      cy.add([...nodes, ...edges]);

      // Apply layout
      const layout = cy.layout({
        name: 'dagre',
        rankDir: 'TB',
        ranker: 'tight-tree',
        nodeSep: 50,
        rankSep: 100,
        padding: 50,
        animate: false,
      } as any);  // dagre layout options not in base types

      layout.run();

      // Fit to viewport with padding, then zoom out a bit for better initial view
      setTimeout(() => {
        cy.fit(undefined, 80);
        cy.zoom(cy.zoom() * 0.8);  // Zoom out 20% for better overview
        cy.center();
      }, 100);
    }

    cyRef.current = cy;

    // Event handlers (attach after successful creation)
    if (cyRef.current) {
      cyRef.current.on('tap', 'node', (event) => {
        const node = event.target;
        const nodeData = node.data();

        // Update selected node in store
        selectNode({
          nodeId: nodeData.id,
          address: nodeData.address,
          label: nodeData.label,
          layer_id: nodeData.layer_id,
        });

        // Update local selected node info for details panel
        setSelectedNodeInfo({
          id: nodeData.id,
          label: nodeData.label,
          address: nodeData.address,
          operation: nodeData.operation,
          shape: nodeData.shape,
          dtype: nodeData.dtype,
          layer_id: nodeData.layer_id,
          category: nodeData.category,
        });
      });

      cyRef.current.on('mouseover', 'node', (event) => {
        const node = event.target;
        setHoveredNode(node.data('id'));
      });

      cyRef.current.on('mouseout', 'node', () => {
        setHoveredNode(null);
      });

      // Deselect on background click
      cyRef.current.on('tap', (event) => {
        if (event.target === cyRef.current) {
          selectNode(null);
          setSelectedNodeInfo(null);
        }
      });
    }

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, [graphData, selectNode, setHoveredNode, isFullScreen, selectedLayer]);

  // Highlight nodes based on timeline position
  useEffect(() => {
    if (!cyRef.current || !correlationIndex) return;

    const cy = cyRef.current;

    // Remove previous highlights
    cy.nodes('.highlighted').removeClass('highlighted');

    // Find active nodes at current timeline position
    const { timeIndex } = correlationIndex;
    const currentTime = timeline.currentTime;

    // Binary search to find the closest timestamp
    let closestIndex = 0;
    for (let i = 0; i < timeIndex.timestamps.length; i++) {
      if (timeIndex.timestamps[i] <= currentTime) {
        closestIndex = i;
      } else {
        break;
      }
    }

    // Highlight active nodes
    const activeNodeIds = timeIndex.activeNodes[closestIndex] || [];
    activeNodeIds.forEach(nodeId => {
      const node = cy.getElementById(nodeId);
      if (node) {
        node.addClass('highlighted');
      }
    });
  }, [timeline.currentTime, correlationIndex]);

  // Apply layer filter
  useEffect(() => {
    if (!cyRef.current) return;

    const cy = cyRef.current;

    // Remove previous dimming
    cy.nodes('.dimmed').removeClass('dimmed');

    // Apply layer filter
    if (filters.selectedLayer !== null) {
      cy.nodes().forEach(node => {
        const layerId = node.data('layer_id');
        if (layerId !== filters.selectedLayer) {
          node.addClass('dimmed');
        }
      });
    }

    // Apply category filter
    if (filters.selectedCategory !== null) {
      cy.nodes().forEach(node => {
        const category = node.data('category');
        if (category !== filters.selectedCategory) {
          node.addClass('dimmed');
        }
      });
    }

    // Apply search filter
    if (filters.searchTerm) {
      const searchLower = filters.searchTerm.toLowerCase();
      cy.nodes().forEach(node => {
        const label = node.data('label').toLowerCase();
        const operation = node.data('operation').toLowerCase();
        if (!label.includes(searchLower) && !operation.includes(searchLower)) {
          node.addClass('dimmed');
        }
      });
    }
  }, [filters]);

  // Programmatically select node from store
  useEffect(() => {
    if (!cyRef.current) return;

    const cy = cyRef.current;

    // Deselect all
    cy.nodes(':selected').unselect();

    // Select node if specified
    if (selectedNode) {
      const node = cy.getElementById(selectedNode.nodeId);
      if (node) {
        node.select();

        // Center on selected node
        cy.animate({
          center: { eles: node },
          zoom: 1.5,
        }, {
          duration: 300,
        });
      }
    }
  }, [selectedNode]);

  return (
    <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex flex-col">
      {/* Header */}
      <div className="h-12 border-b border-gray-700 flex items-center justify-between px-4">
        <div>
          <span className="text-white font-semibold">Computation Graph</span>
          {graphData && (
            <span className="ml-3 text-gray-400 text-sm">
              {selectedLayer === 'all'
                ? `${graphData.metadata.total_nodes} nodes (all layers)`
                : `Layer ${selectedLayer} + connected nodes`
              }
            </span>
          )}
        </div>

        {/* Graph controls */}
        <div className="flex gap-2">
          {/* Layer selector dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-gray-400 text-sm">Layer:</span>
            <select
              value={selectedLayer}
              onChange={(e) => {
                const value = e.target.value;
                setSelectedLayer(value === 'all' ? 'all' : parseInt(value, 10));
              }}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {availableLayers.map(layer => (
                <option key={layer} value={layer!}>
                  Layer {layer}
                </option>
              ))}
              <option value="all">All Layers</option>
            </select>
          </div>

          {!isFullScreen && (
            <button
              onClick={() => setFullScreen('graph')}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
              title="Enter full-screen mode"
            >
              ⛶ Full Screen
            </button>
          )}
     
        </div>
      </div>

      {/* Main graph area */}
      <div className="flex-1 relative">
        <div ref={containerRef} className="w-full h-full" />

        {/* Node details panel (when node selected) */}
        {selectedNodeInfo && (
          <div className="absolute top-4 right-4 w-72 bg-gray-800 border border-gray-600 rounded-lg p-4 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <span className="text-white font-semibold">Node Details</span>
              <button
                onClick={() => {
                  selectNode(null);
                  setSelectedNodeInfo(null);
                }}
                className="text-gray-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="space-y-2 text-sm">
              <div>
                <span className="text-gray-400">Label:</span>{' '}
                <span className="text-white font-mono">{selectedNodeInfo.label}</span>
              </div>
              <div>
                <span className="text-gray-400">Operation:</span>{' '}
                <span className="text-white">{selectedNodeInfo.operation}</span>
              </div>
              <div>
                <span className="text-gray-400">Category:</span>{' '}
                <span
                  className="inline-block px-2 py-0.5 rounded text-white text-xs font-semibold"
                  style={{
                    backgroundColor: CATEGORY_COLORS[selectedNodeInfo.category as keyof typeof CATEGORY_COLORS] || CATEGORY_COLORS.other,
                  }}
                >
                  {selectedNodeInfo.category}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Shape:</span>{' '}
                <span className="text-white font-mono">
                  [{selectedNodeInfo.shape.join(', ')}]
                </span>
              </div>
              <div>
                <span className="text-gray-400">Data Type:</span>{' '}
                <span className="text-white font-mono">{selectedNodeInfo.dtype}</span>
              </div>
              <div>
                <span className="text-gray-400">Layer:</span>{' '}
                <span className="text-white">
                  {selectedNodeInfo.layer_id !== null ? selectedNodeInfo.layer_id : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Address:</span>{' '}
                <span className="text-white font-mono text-xs break-all">
                  {selectedNodeInfo.address}
                </span>
              </div>
            </div>

            {/* Show trace count if available */}
            {correlationIndex && (
              <div className="mt-3 pt-3 border-t border-gray-700">
                <span className="text-gray-400 text-sm">
                  Trace entries:{' '}
                  <span className="text-white font-semibold">
                    {correlationIndex.addressToTraces.get(selectedNodeInfo.address)?.length || 0}
                  </span>
                </span>
              </div>
            )}
          </div>
        )}

        {/* Category legend */}
        <div className="absolute bottom-4 left-4 bg-gray-800 border border-gray-600 rounded-lg p-3">
          <div className="text-white text-xs font-semibold mb-2">Categories</div>
          <div className="space-y-1">
            {Object.entries(CATEGORY_COLORS).map(([category, color]) => (
              <div key={category} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded"
                  style={{ backgroundColor: color }}
                />
                <span className="text-gray-300 text-xs capitalize">{category}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
