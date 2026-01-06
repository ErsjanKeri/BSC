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
  const [showAllNodes, setShowAllNodes] = useState(false);

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

  // Initialize Cytoscape instance
  useEffect(() => {
    if (!containerRef.current || !graphData) return;

    // Destroy existing instance
    if (cyRef.current) {
      cyRef.current.destroy();
    }

    // Filter nodes to show
    // Full-screen: show all nodes automatically
    // Small screen: Layer 0 only unless user toggles
    const nodesToShow = (isFullScreen || showAllNodes)
      ? graphData.nodes
      : graphData.nodes.filter(n => n.layer_id === 0 || n.layer_id === null);

      const edgesToShow = graphData.edges.filter(e => {
        const sourceNode = graphData.nodes.find(n => n.id === e.source);
        const targetNode = graphData.nodes.find(n => n.id === e.target);
        return nodesToShow.includes(sourceNode!) && nodesToShow.includes(targetNode!);
      });

      console.log(`Rendering ${nodesToShow.length} nodes, ${edgesToShow.length} edges`);

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
            'border-color': '#1f2937', // gray-800
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
  }, [graphData, selectNode, setHoveredNode, showAllNodes, isFullScreen]);

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
              {showAllNodes ? `${graphData.metadata.total_nodes} nodes` : `Layer 0 only`}
            </span>
          )}
        </div>

        {/* Graph controls */}
        <div className="flex gap-2">
          {!isFullScreen && (
            <>
              <button
                onClick={() => setFullScreen('graph')}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
                title="Enter full-screen mode"
              >
                ⛶ Full Screen
              </button>
              <button
                onClick={() => setShowAllNodes(!showAllNodes)}
                className={`px-3 py-1 text-white text-sm rounded ${
                  showAllNodes ? 'bg-amber-600 hover:bg-amber-500' : 'bg-gray-700 hover:bg-gray-600'
                }`}
                title="Toggle all layers"
              >
                {showAllNodes ? 'Layer 0 Only' : 'Show All Layers'}
              </button>
            </>
          )}
          <button
            onClick={() => cyRef.current?.fit(undefined, 50)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
            title="Fit graph to viewport"
          >
            Fit
          </button>
          <button
            onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 1.2)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
            title="Zoom in"
          >
            +
          </button>
          <button
            onClick={() => cyRef.current?.zoom(cyRef.current.zoom() * 0.8)}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
            title="Zoom out"
          >
            −
          </button>
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
