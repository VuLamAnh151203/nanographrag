import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import asdict

# Import from lightrag components
from lightrag.utils import logger
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.operate_old import kg_direct_recall
from lightrag.types import QueryParam
from lightrag.util_funcs import always_get_an_event_loop

# Import entity mapping
from entity_mapping import find_mapped_entity_description, find_mapped_edge_description

class DualGraphQuery:
    """
    Class for querying two knowledge graphs in different languages,
    mapping entities between them, and returning consolidated results.
    """
    
    def __init__(
        self,
        graph1_instance,
        graph2_instance,
        mapping_file_path: str,
        output_dir: str = "query_results",
        graph1_name: str = "graph1",
        graph2_name: str = "graph2"
    ):
        """
        Initialize dual graph querying with two LightRAG instances and entity mapping.
        
        Args:
            graph1_instance: First LightRAG instance (primary language)
            graph2_instance: Second LightRAG instance (secondary language)
            mapping_file_path: Path to the entity/edge mapping JSON file
            output_dir: Directory to save query results
            graph1_name: Name identifier for the first graph
            graph2_name: Name identifier for the second graph
        """
        self.graph1 = graph1_instance
        self.graph2 = graph2_instance
        self.mapping_file = mapping_file_path
        self.output_dir = output_dir
        self.graph1_name = graph1_name
        self.graph2_name = graph2_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate mapping file exists
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")
        
        logger.info(f"DualGraphQuery initialized with {graph1_name} and {graph2_name}")
    
    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Query both graphs and return consolidated results.
        
        Args:
            query_text: The query string
            mode: Query mode - 'local' (nodes), 'global' (edges), or 'hybrid' (both)
            param: Query parameters (if None, will be created based on mode)
            save_results: Whether to save results to a JSONL file
            
        Returns:
            Dict containing consolidated query results
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.aquery(query_text, mode, param, save_results)
        )
    
    async def aquery(
        self,
        query_text: str,
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Async version of query method.
        """
        # Set up query parameters based on mode if not provided
        if param is None:
            param = QueryParam()
            
            if mode.lower() == "local":
                param.mode = "local"
            elif mode.lower() == "global":
                param.mode = "global"
            else:
                param.mode = "hybrid"  # default
        
        logger.info(f"Querying with mode: {param.mode}")
        
        # Query both graphs
        graph1_results, graph2_results = await asyncio.gather(
            self._query_single_graph(self.graph1, query_text, param),
            self._query_single_graph(self.graph2, query_text, param)
        )
        
        # Map entities from graph2 to graph1
        mapped_graph2_results = self._map_graph2_to_graph1(graph2_results)
        
        # Merge results
        consolidated_results = self._merge_results(
            graph1_results, 
            mapped_graph2_results
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(consolidated_results)
        
        # Create final result object
        result = {
            "query": query_text,
            "mode": param.mode,
            "candidates": consolidated_results,
            "metrics": metrics,
            "graph1_raw_results": graph1_results,
            "graph2_raw_results": graph2_results,
            "timestamp": self._get_timestamp()
        }
        
        # Save results if requested
        if save_results:
            self._save_results(result, query_text)
        
        return result
    
    async def _query_single_graph(
        self, 
        graph_instance, 
        query_text: str, 
        param: QueryParam
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Query a single graph and return results."""
        try:
            candidates, hl_keywords, ll_keywords = await graph_instance.anew_retrieval(
                query=query_text,
                param=param
            )
            return candidates, hl_keywords, ll_keywords
        except Exception as e:
            logger.error(f"Error querying graph: {e}")
            return [], [], []
    
    def _map_graph2_to_graph1(
        self, 
        graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Map entities and edges from graph2 to graph1 using the mapping file."""
        candidates, hl_keywords, ll_keywords = graph2_results
        mapped_candidates = []
        
        for candidate in candidates:
            cand_type = candidate.get("type")
            
            if cand_type == "node":
                # Map entity node
                entity_name = candidate.get("entity_name")
                description = candidate.get("description", "")
                
                # Use entity mapping
                mapped_result = find_mapped_entity_description(
                    self.mapping_file,
                    entity_name,
                    description,
                    source_graph_key_prefix=self.graph2_name,
                    target_graph_key_prefix=self.graph1_name
                )
                
                if mapped_result:
                    mapped_entity, mapped_desc = mapped_result
                    mapped_candidate = candidate.copy()
                    mapped_candidate["entity_name"] = mapped_entity
                    mapped_candidate["description"] = mapped_desc
                    mapped_candidate["original_entity"] = entity_name
                    mapped_candidate["mapping_source"] = self.graph2_name
                    mapped_candidates.append(mapped_candidate)
                else:
                    # Keep original if no mapping found, but mark it
                    candidate["mapping_status"] = "unmapped"
                    mapped_candidates.append(candidate)
                    
            elif cand_type == "edge":
                # Map edge
                src_entity = candidate.get("src_id")
                tgt_entity = candidate.get("tgt_id")
                description = candidate.get("description", "")
                
                # Use edge mapping
                mapped_result = find_mapped_edge_description(
                    self.mapping_file,
                    src_entity,
                    tgt_entity,
                    description,
                    source_graph_key_prefix=self.graph2_name,
                    target_graph_key_prefix=self.graph1_name
                )
                
                if mapped_result:
                    mapped_src, mapped_tgt, mapped_desc = mapped_result
                    mapped_candidate = candidate.copy()
                    mapped_candidate["src_id"] = mapped_src
                    mapped_candidate["tgt_id"] = mapped_tgt
                    mapped_candidate["description"] = mapped_desc
                    mapped_candidate["original_src"] = src_entity
                    mapped_candidate["original_tgt"] = tgt_entity
                    mapped_candidate["mapping_source"] = self.graph2_name
                    mapped_candidates.append(mapped_candidate)
                else:
                    # Keep original if no mapping found, but mark it
                    candidate["mapping_status"] = "unmapped"
                    mapped_candidates.append(candidate)
            else:
                # For other types, keep as is
                mapped_candidates.append(candidate)
        
        return mapped_candidates, hl_keywords, ll_keywords
    
    def _merge_results(
        self,
        graph1_results: Tuple[List[Dict[str, Any]], List[str], List[str]],
        graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]]
    ) -> List[Dict[str, Any]]:
        """Merge results from both graphs, sorting by score."""
        candidates1, _, _ = graph1_results
        candidates2, _, _ = graph2_results
        
        # Combine candidates
        all_candidates = candidates1 + candidates2
        
        # Sort by score in descending order
        sorted_candidates = sorted(
            all_candidates, 
            key=lambda x: x.get("relevance_score", 0.0), 
            reverse=True
        )
        
        return sorted_candidates
    
    def _calculate_metrics(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate recall metrics for nodes and chunks."""
        node_count = 0
        edge_count = 0
        chunk_count = len(set(c.get("chunk_id") for c in candidates if c.get("chunk_id")))
        
        for candidate in candidates:
            if candidate.get("type") == "node":
                node_count += 1
            elif candidate.get("type") == "edge":
                edge_count += 1
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "chunk_count": chunk_count,
            "total_candidates": len(candidates)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_results(self, result: Dict[str, Any], query_text: str) -> str:
        """Save results to a JSONL file and return the path."""
        # Create a sanitized filename from the query
        sanitized_query = "".join(c if c.isalnum() else "_" for c in query_text[:30])
        timestamp = result["timestamp"].replace(":", "-").replace(".", "-")
        filename = f"query_{sanitized_query}_{timestamp}.jsonl"
        filepath = os.path.join(self.output_dir, filename)
        
        # Write to JSONL
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def batch_query(
        self,
        queries: List[str],
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        batch_output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Run multiple queries in batch mode and return consolidated results.
        
        Args:
            queries: List of query strings to process
            mode: Query mode - 'local' (nodes), 'global' (edges), or 'hybrid' (both)
            param: Query parameters (if None, will be created based on mode)
            save_results: Whether to save individual results to JSONL files
            batch_output_file: Optional filename to save all results in single JSONL file
            
        Returns:
            List of result dictionaries, one for each query
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.abatch_query(queries, mode, param, save_results, batch_output_file)
        )
    
    async def abatch_query(
        self,
        queries: List[str],
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        batch_output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Async version of batch_query method.
        """
        all_results = []
        logger.info(f"Starting batch query processing for {len(queries)} queries")
        
        for i, query_text in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query_text[:50]}...")
            result = await self.aquery(
                query_text=query_text,
                mode=mode,
                param=param,
                save_results=save_results
            )
            all_results.append(result)
        
        # Save all results to a single batch file if requested
        if batch_output_file:
            batch_filepath = os.path.join(self.output_dir, batch_output_file)
            with open(batch_filepath, "w", encoding="utf-8") as f:
                for result in all_results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            logger.info(f"Batch results saved to {batch_filepath}")
        
        logger.info(f"Completed processing {len(queries)} queries")
        return all_results

# Command-line execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query dual knowledge graphs with entity mapping")
    parser.add_argument("--query", help="Query text (for single query mode)")
    parser.add_argument("--queries_file", help="File containing one query per line (for batch mode)")
    parser.add_argument("--mode", default="hybrid", choices=["local", "global", "hybrid"], 
                       help="Query mode - local (nodes), global (edges), or hybrid (both)")
    parser.add_argument("--graph1_path", required=True, help="Path to first graph's data directory")
    parser.add_argument("--graph2_path", required=True, help="Path to second graph's data directory")
    parser.add_argument("--mapping_file", required=True, help="Path to entity/edge mapping file")
    parser.add_argument("--output_dir", default="query_results", help="Directory to save results")
    parser.add_argument("--graph1_name", default="graph1", help="Name for first graph")
    parser.add_argument("--graph2_name", default="graph2", help="Name for second graph")
    parser.add_argument("--batch_output", help="Filename for combined batch results (for batch mode)")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model to use for both graphs")
    
    args = parser.parse_args()
    
    # Ensure either query or queries_file is provided
    if not args.query and not args.queries_file:
        parser.error("Either --query or --queries_file must be provided")
    
    # Import LightRAG and other required components
    from lightrag.lightrag import LightRAG
    from lightrag.embedding.sentence_transformers_embedding import SentenceTransformersEmbedding
    
    # Initialize embedding function for both graphs 
    embedding_func = SentenceTransformersEmbedding(model_name=args.embedding_model)
    
    # Load both graphs - initialize directly instead of using from_saved
    graph1 = LightRAG(
        working_dir=args.graph1_path,
        embedding_func=embedding_func
    )
    graph2 = LightRAG(
        working_dir=args.graph2_path,
        embedding_func=embedding_func
    )
    
    # Create dual query instance
    dual_query = DualGraphQuery(
        graph1_instance=graph1,
        graph2_instance=graph2,
        mapping_file_path=args.mapping_file,
        output_dir=args.output_dir,
        graph1_name=args.graph1_name,
        graph2_name=args.graph2_name
    )
    
    # Determine if running in single query or batch mode
    if args.query:
        # Single query mode
        results = dual_query.query(
            query_text=args.query,
            mode=args.mode,
            save_results=True
        )
        
        # Print summary
        print(f"Query: {args.query}")
        print(f"Mode: {args.mode}")
        print(f"Results: {results['metrics']['total_candidates']} candidates found")
        print(f"Nodes: {results['metrics']['node_count']}")
        print(f"Edges: {results['metrics']['edge_count']}")
        print(f"Chunks: {results['metrics']['chunk_count']}")
        
    else:
        # Batch mode
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
            
        print(f"Processing {len(queries)} queries in batch mode")
        
        all_results = dual_query.batch_query(
            queries=queries,
            mode=args.mode,
            save_results=True,
            batch_output_file=args.batch_output
        )
        
        # Print summary
        total_candidates = sum(r['metrics']['total_candidates'] for r in all_results)
        total_nodes = sum(r['metrics']['node_count'] for r in all_results)
        total_edges = sum(r['metrics']['edge_count'] for r in all_results)
        total_chunks = sum(r['metrics']['chunk_count'] for r in all_results)
        
        print(f"Batch completed: {len(queries)} queries processed")
        print(f"Total candidates: {total_candidates}")
        print(f"Total nodes: {total_nodes}")
        print(f"Total edges: {total_edges}")
        print(f"Total chunks: {total_chunks}")
        
    print(f"Results saved to {args.output_dir}") 