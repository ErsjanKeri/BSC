/**
 * Type declarations for cytoscape-dagre
 * Since @types/cytoscape-dagre doesn't exist, we declare the module manually
 */

declare module 'cytoscape-dagre' {
  import { Ext } from 'cytoscape';
  const dagre: Ext;
  export = dagre;
}
