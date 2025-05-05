import json
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class BatchGraphMapper:
    def __init__(self, mapping_file_path: str, graph1_name: str = "graph1", graph2_name: str = "graph2"):
        """
        Initialize the batch mapper with mapping indices.
        
        Args:
            mapping_file_path: Path to the JSON mapping file
            graph1_name: Name of graph1 (source)
            graph2_name: Name of graph2 (target)
        """
        self.mapping_file = mapping_file_path
        self.graph1_name = graph1_name
        self.graph2_name = graph2_name
        
        # Build indices once during initialization
        self.entity_index = self._build_entity_mapping_index(mapping_file_path)
        self.edge_index = self._build_edge_mapping_index(mapping_file_path)
        
        logger.info(f"BatchGraphMapper initialized with {len(self.entity_index)} entity mappings and {len(self.edge_index)} edge mappings")
    
    def _build_entity_mapping_index(self, mapping_file_path: str) -> Dict:
        """
        Build an indexed lookup structure for entity mappings, keeping only the highest
        similarity score mapping for each entity.
        """
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                all_mappings = json.load(f)
        except Exception as e:
            logger.error(f"Error reading or parsing mapping file {mapping_file_path}: {e}")
            return {}
        
        # Create an index for faster lookups
        entity_index = {}
        
        # Group by key first, to find highest score per key
        grouped_mappings = {}
        for mapping in all_mappings:
            if mapping.get("type") == "entity":
                source_entity = mapping.get("graph1_entity_name")
                source_desc = mapping.get("graph1_trigger_description")
                
                if source_entity is not None and source_desc is not None:
                    # Use a composite key of entity name and description
                    key = (str(source_entity).strip().lower(), str(source_desc).strip())
                    
                    # Keep track of mappings with scores for this key
                    if key not in grouped_mappings:
                        grouped_mappings[key] = []
                        
                    grouped_mappings[key].append({
                        "mapped_entity": mapping.get("graph2_entity_name"),
                        "mapped_desc": mapping.get("graph2_trigger_description"),
                        "mapped_chunk_id": mapping.get("graph2_trigger_chunk_id"),
                        "similarity_score": mapping.get("similarity_score", 0.0)
                    })
        
        # Now keep only the highest score mapping for each key
        for key, mappings in grouped_mappings.items():
            # Sort by similarity score in descending order
            mappings.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            # Keep only the highest scoring mapping
            entity_index[key] = mappings[0]
        
        logger.info(f"Built entity mapping index with {len(entity_index)} entries")
        return entity_index

    def _build_edge_mapping_index(self, mapping_file_path: str) -> Dict:
        """
        Build an indexed lookup structure for edge mappings, keeping only the highest
        similarity score mapping for each edge.
        """
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                all_mappings = json.load(f)
        except Exception as e:
            logger.error(f"Error reading or parsing mapping file {mapping_file_path}: {e}")
            return {}
        
        # Create an index for faster lookups
        edge_index = {}
        
        # Group by key first, to find highest score per key
        grouped_mappings = {}
        for mapping in all_mappings:
            if mapping.get("type") == "edge":
                source_src = mapping.get("graph1_src_entity")
                source_tgt = mapping.get("graph1_tgt_entity")
                source_desc = mapping.get("graph1_trigger_description")
                
                if source_src is not None and source_tgt is not None and source_desc is not None:
                    # Use a composite key of source, target and description
                    key = (
                        str(source_src).strip().lower(),
                        str(source_tgt).strip().lower(),
                        str(source_desc).strip()
                    )
                    
                    # Keep track of mappings with scores for this key
                    if key not in grouped_mappings:
                        grouped_mappings[key] = []
                        
                    grouped_mappings[key].append({
                        "mapped_src": mapping.get("graph2_src_entity"),
                        "mapped_tgt": mapping.get("graph2_tgt_entity"),
                        "mapped_desc": mapping.get("graph2_trigger_description"),
                        "mapped_chunk_id": mapping.get("graph2_trigger_chunk_id"),
                        "similarity_score": mapping.get("similarity_score", 0.0)
                    })
        
        # Now keep only the highest score mapping for each key
        for key, mappings in grouped_mappings.items():
            # Sort by similarity score in descending order
            mappings.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            # Keep only the highest scoring mapping
            edge_index[key] = mappings[0]
        
        logger.info(f"Built edge mapping index with {len(edge_index)} entries")
        return edge_index
    
    def find_mapped_entity_description(self, input_entity_name: str, input_description: str) -> Optional[Tuple[str, str, str]]:
        """
        Uses indexed lookup to find mapped entity and description.
        """
        normalized_entity = str(input_entity_name).strip().lower()
        normalized_desc = str(input_description).strip()
        
        # Lookup in our pre-built index
        key = (normalized_entity, normalized_desc)
        mapping = self.entity_index.get(key)
        
        if mapping and mapping["mapped_entity"] is not None and mapping["mapped_desc"] is not None and mapping["mapped_chunk_id"] is not None:
            result = (
                str(mapping["mapped_entity"]), 
                str(mapping["mapped_desc"]), 
                str(mapping["mapped_chunk_id"])
            )
            return result
        
        return None

    def find_mapped_edge_description(self, input_src_entity: str, input_tgt_entity: str, input_description: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Uses indexed lookup to find mapped edge details.
        """
        normalized_src = str(input_src_entity).strip().lower()
        normalized_tgt = str(input_tgt_entity).strip().lower()
        normalized_desc = str(input_description).strip()
        
        # Lookup in our pre-built index
        key = (normalized_src, normalized_tgt, normalized_desc)
        mapping = self.edge_index.get(key)
        
        if mapping and mapping["mapped_src"] is not None and mapping["mapped_tgt"] is not None and \
           mapping["mapped_desc"] is not None and mapping["mapped_chunk_id"] is not None:
            result = (
                str(mapping["mapped_src"]), 
                str(mapping["mapped_tgt"]), 
                str(mapping["mapped_desc"]), 
                str(mapping["mapped_chunk_id"])
            )
            return result
        
        return None

    def map_graph2_to_graph1(
        self,
        graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """
        Map entities and edges from graph2 to graph1 using the mapping file.
        """
        candidates, hl_keywords, ll_keywords = graph2_results
        mapped_candidates = []
        
        for candidate in candidates:
            cand_type = candidate.get("retrieval_type")
            
            if cand_type == "node":
                # Map entity node
                entity_name = candidate.get("entity_name")
                description = candidate.get("description", "")
                
                # Use entity mapping
                mapped_result = self.find_mapped_entity_description(entity_name, description)
                
                if mapped_result:
                    # print("mapp cross ne")
                    mapped_entity, mapped_desc, mapped_chunkid = mapped_result
                    mapped_candidate = candidate.copy()
                    mapped_candidate["entity_name"] = mapped_entity
                    mapped_candidate["description"] = mapped_desc
                    mapped_candidate["original_entity"] = entity_name
                    mapped_candidate['retrieved_chunk_id'] = mapped_chunkid
                    mapped_candidate["mapping_status"] = "mapped_success"
                    mapped_candidates.append(mapped_candidate)
                else:
                    # Keep original if no mapping found, but mark it
                    candidate["mapping_status"] = "unmapped"
                    # mapped_candidates.append(candidate)
                    
            elif cand_type == "edge":
                # Map edge
                src_entity = candidate.get("src_id")
                tgt_entity = candidate.get("tgt_id")
                description = candidate.get("description", "")
                
                # Use edge mapping
                mapped_result = self.find_mapped_edge_description(src_entity, tgt_entity, description)
                
                if mapped_result:
                    # print("mapp cross ne")
                    mapped_src, mapped_tgt, mapped_desc, mapped_chunkid = mapped_result
                    mapped_candidate = candidate.copy()
                    mapped_candidate["src_id"] = mapped_src
                    mapped_candidate["tgt_id"] = mapped_tgt
                    mapped_candidate["description"] = mapped_desc
                    mapped_candidate["original_src"] = src_entity
                    mapped_candidate["original_tgt"] = tgt_entity
                    mapped_candidate['retrieved_chunk_id'] = mapped_chunkid
                    mapped_candidate["mapping_status"] = "mapped_success"
                    mapped_candidates.append(mapped_candidate)
                else:
                    # Keep original if no mapping found, but mark it
                    candidate["mapping_status"] = "unmapped"
                    # mapped_candidates.append(candidate)
            else:
                # For other types, keep as is
                mapped_candidates.append(candidate)
        
        return mapped_candidates, hl_keywords, ll_keywords

# For backward compatibility with the find_mapped_entity_description and find_mapped_edge_description functions
_mappers = {}

def find_mapped_entity_description(
    mapping_file_path: str,
    input_entity_name: str,
    input_description: str,
    source_graph_key_prefix: str = "graph1",
    target_graph_key_prefix: str = "graph2"
) -> Optional[Tuple[str, str, str]]:
    """
    Cached wrapper for the original function to improve performance.
    """
    # Create or retrieve a cached mapper
    global _mappers
    mapper_key = f"{mapping_file_path}_{source_graph_key_prefix}_{target_graph_key_prefix}"
    
    if mapper_key not in _mappers:
        _mappers[mapper_key] = BatchGraphMapper(
            mapping_file_path=mapping_file_path,
            graph1_name=source_graph_key_prefix,
            graph2_name=target_graph_key_prefix
        )
        
    mapper = _mappers[mapper_key]
    return mapper.find_mapped_entity_description(input_entity_name, input_description)

def find_mapped_edge_description(
    mapping_file_path: str,
    input_src_entity: str,
    input_tgt_entity: str,
    input_description: str,
    source_graph_key_prefix: str = "graph1",
    target_graph_key_prefix: str = "graph2"
) -> Optional[Tuple[str, str, str, str]]:
    """
    Cached wrapper for the original function to improve performance.
    """
    # Create or retrieve a cached mapper
    global _mappers
    mapper_key = f"{mapping_file_path}_{source_graph_key_prefix}_{target_graph_key_prefix}"
    
    if mapper_key not in _mappers:
        _mappers[mapper_key] = BatchGraphMapper(
            mapping_file_path=mapping_file_path,
            graph1_name=source_graph_key_prefix,
            graph2_name=target_graph_key_prefix
        )
        
    mapper = _mappers[mapper_key]
    return mapper.find_mapped_edge_description(input_src_entity, input_tgt_entity, input_description)