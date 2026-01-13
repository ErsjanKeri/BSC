/**
 * View 4: 3D Transformer Architecture Visualization
 *
 * Features:
 * - 3D visualization of transformer layers (stacked boxes)
 * - Interactive rotation, zoom, pan
 * - Layer highlighting during timeline animation
 * - Color-coded by access activity
 */

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import { useAppStore } from '../stores/useAppStore';
import type { Mesh } from 'three';

interface LayerBoxProps {
  layer_id: number;
  position: [number, number, number];
  isActive: boolean;
  isSelected: boolean;
  onClick: () => void;
  accessCount: number;
  maxAccessCount: number;
}

function LayerBox({
  layer_id,
  position,
  isActive,
  isSelected,
  onClick,
  accessCount,
  maxAccessCount,
}: LayerBoxProps) {
  const meshRef = useRef<Mesh>(null);

  // Pulse animation when active
  useFrame((state) => {
    if (meshRef.current && isActive) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.05;
      meshRef.current.scale.set(scale, scale, scale);
    } else if (meshRef.current) {
      meshRef.current.scale.set(1, 1, 1);
    }
  });

  // Color based on access frequency
  const color = useMemo(() => {
    if (isActive) return '#fbbf24'; // amber-400 (active)
    if (isSelected) return '#3b82f6'; // blue-500 (selected)

    const intensity = maxAccessCount > 0 ? accessCount / maxAccessCount : 0;

    if (intensity === 0) return '#374151'; // gray-700 (no access)
    if (intensity < 0.3) return '#3b82f6'; // blue-500 (low)
    if (intensity < 0.7) return '#10b981'; // green-500 (med)
    return '#ef4444'; // red-500 (high)
  }, [isActive, isSelected, accessCount, maxAccessCount]);

  return (
    <group position={position}>
      <mesh ref={meshRef} onClick={onClick}>
        <boxGeometry args={[2, 0.3, 2]} />
        <meshStandardMaterial
          color={color}
          emissive={isActive ? '#fbbf24' : '#000000'}
          emissiveIntensity={isActive ? 0.5 : 0}
        />
      </mesh>

      {/* Layer label */}
      <Text
        position={[0, 0, 1.2]}
        fontSize={0.2}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        Layer {layer_id}
      </Text>

      {/* Access count label */}
      {accessCount > 0 && (
        <Text
          position={[0, -0.25, 1.2]}
          fontSize={0.12}
          color="#9ca3af"
          anchorX="center"
          anchorY="middle"
        >
          {accessCount} accesses
        </Text>
      )}
    </group>
  );
}

function TransformerModel() {
  const { memoryMap, traceData, timeline, correlationIndex } = useAppStore();

  if (!memoryMap) return null;

  const numLayers = memoryMap.metadata.n_layers;

  // Calculate access counts per layer
  const layerAccessCounts = useMemo(() => {
    const counts = new Map<number, number>();

    if (traceData) {
      traceData.entries.forEach(entry => {
        if (entry.layer_id !== null && entry.layer_id !== 65535) {
          counts.set(entry.layer_id, (counts.get(entry.layer_id) || 0) + 1);
        }
      });
    }

    return counts;
  }, [traceData]);

  const maxAccessCount = Math.max(...Array.from(layerAccessCounts.values()), 1);

  // Determine active layers at current timeline position
  const activeLayers = useMemo(() => {
    if (!traceData || !correlationIndex) return new Set<number>();

    const active = new Set<number>();
    const currentTime = timeline.currentTime;

    // Find entries around current time (±10ms window)
    traceData.entries.forEach(entry => {
      if (
        Math.abs(entry.timestamp_relative_ms - currentTime) < 10 &&
        entry.layer_id !== null &&
        entry.layer_id !== 65535
      ) {
        active.add(entry.layer_id);
      }
    });

    return active;
  }, [traceData, correlationIndex, timeline.currentTime]);

  // Generate layer boxes
  const layerSpacing = 0.5;
  const layers = [];

  for (let i = 0; i < numLayers; i++) {
    const y = (i - numLayers / 2) * layerSpacing;
    const accessCount = layerAccessCounts.get(i) || 0;

    layers.push(
      <LayerBox
        key={i}
        layer_id={i}
        position={[0, y, 0]}
        isActive={activeLayers.has(i)}
        isSelected={false}
        onClick={() => {}}
        accessCount={accessCount}
        maxAccessCount={maxAccessCount}
      />
    );
  }

  return <>{layers}</>;
}

interface TransformerViewProps {
  isFullScreen: boolean;
}

export function TransformerView({ isFullScreen }: TransformerViewProps) {
  const { memoryMap, setFullScreen } = useAppStore();

  if (!memoryMap) {
    return (
      <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex items-center justify-center">
        <span className="text-gray-500">No model data loaded</span>
      </div>
    );
  }

  return (
    <div className="w-full h-full bg-gray-900 border border-gray-700 rounded-lg flex flex-col">
      {/* Header */}
      <div className="h-12 border-b border-gray-700 flex items-center justify-between px-4">
        <div>
          <span className="text-white font-semibold">3D Transformer</span>
          <span className="ml-3 text-gray-400 text-sm">
            {memoryMap.metadata.n_layers} layers
          </span>
        </div>

        <div className="flex items-center gap-4">
          {!isFullScreen && (
            <button
              onClick={() => setFullScreen('transformer')}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded"
              title="Enter full-screen mode"
            >
              ⛶ Full Screen
            </button>
          )}

          {/* Controls hint */}
          <div className="text-gray-500 text-xs">
            Click layer to filter • Drag to rotate • Scroll to zoom
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <div className="flex-1">
        <Canvas camera={{ position: [5, 0, 5], fov: 50 }}>
          {/* Lighting */}
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <pointLight position={[-10, -10, -5]} intensity={0.5} />

          {/* Transformer model */}
          <TransformerModel />

          {/* Camera controls */}
          <OrbitControls
            enableZoom={true}
            enablePan={true}
            enableRotate={true}
            minDistance={3}
            maxDistance={20}
          />

          {/* Grid helper */}
          <gridHelper args={[10, 10, '#374151', '#1f2937']} />
        </Canvas>
      </div>

      {/* Legend */}
      <div className="border-t border-gray-700 p-3">
        <div className="flex items-center gap-4 text-xs">
          <span className="text-gray-400 font-semibold">Colors:</span>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#374151' }}></div>
            <span className="text-gray-400">No access</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#3b82f6' }}></div>
            <span className="text-gray-400">Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#10b981' }}></div>
            <span className="text-gray-400">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#ef4444' }}></div>
            <span className="text-gray-400">High</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded" style={{ backgroundColor: '#fbbf24' }}></div>
            <span className="text-gray-400">Active (pulsing)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
