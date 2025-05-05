import os
import json
import asyncio
import logging
import argparse
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np

# Import necessary components from lightrag
# try:
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.utils import logger
# except ImportError as e:
#     print(f"Error importing lightrag components: {e}")
#     print("Please ensure lightrag and entity_mapping are available.")
#     # Define logger as a basic logger if import fails, to allow script structure
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)
#     # Define NanoVectorDBStorage as a dummy if import fails
#     class NanoVectorDBStorage:
#         def __init__(self, *args, **kwargs):
#             logger.error("Failed to import NanoVectorDBStorage. Using dummy.")
#         async def find_similar_vdb_pairs(self, *args, **kwargs):
#             logger.error("find_similar_vdb_pairs not available due to import error.")
#             return []


class DummyEmbedding:
    """Dummy embedding function for inference - may be needed by NanoVectorDBStorage."""
    def __init__(self, embedding_dim: int = 1024):
        self.embedding_dim = embedding_dim

    async def __call__(self, texts: list[str]) -> np.ndarray:
        """Return random embeddings for inference."""
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)

# --- Core Mapping Logic ---

async def map_elements_between_graphs(
    graph1_name: str,
    graph2_name: str,
    graph1_entity_vdb_path: str,
    graph2_entity_vdb_path: str,
    graph1_edge_vdb_path: str,
    graph2_edge_vdb_path: str,
    output_dir: str,
    threshold: float = 0.8,
    vdb_config: Optional[Dict[str, Any]] = None, # Made optional
    embedding_dim: int = 768,
    max_threshold: float = 1.0
) -> str:
    """
    Map entities AND edges between two knowledge graphs based on VDB similarity.
    Saves pairs of VDB entries that match, including their triggering chunk IDs
    and descriptions, to a JSON file.

    Args:
        graph1_name: Name for graph 1 (e.g., 'en').
        graph2_name: Name for graph 2 (e.g., 'vi').
        graph1_entity_vdb_path: Path to graph 1's entity VDB file.
        graph2_entity_vdb_path: Path to graph 2's entity VDB file.
        graph1_edge_vdb_path: Path to graph 1's edge VDB file.
        graph2_edge_vdb_path: Path to graph 2's edge VDB file.
        output_dir: Directory to save mapping results.
        threshold: Similarity threshold for matching (0.0 to 1.0).
        vdb_config: Optional configuration for NanoVectorDBStorage.
        embedding_dim: Dimension of embeddings used in VDBs.

    Returns:
        The path to the generated JSON mapping file.

    Raises:
        FileNotFoundError: If any input VDB file is not found.
        Exception: For other processing errors.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"element_mapping_{graph1_name}_{graph2_name}_thresh{threshold}.json"
        output_path = os.path.join(output_dir, output_filename)

        logger.info(f"Starting element mapping. Output will be saved to: {output_path}")

        # Check if input files exist before proceeding
        for path in [graph1_entity_vdb_path, graph2_entity_vdb_path, graph1_edge_vdb_path, graph2_edge_vdb_path]:
             if not os.path.exists(path):
                  raise FileNotFoundError(f"Input VDB file not found: {path}")

        # Initialize VDB Storage Helper
        embedding_func = DummyEmbedding(embedding_dim=embedding_dim)
        vdb_storage = NanoVectorDBStorage(
            global_config=vdb_config or {},
            namespace="element_mapping",
            embedding_func=embedding_func
        )

        all_mapping_records = []

        # --- Entity Mapping ---
        logger.info(f"Finding similar ENTITY pairs (Threshold: {threshold})...")
        entity_vdb_pairs = await vdb_storage.find_similar_vdb_pairs(
            graph1_entity_vdb_path,
            graph2_entity_vdb_path,
            threshold,
            max_threshold=max_threshold
        )
        logger.info(f"Processing {len(entity_vdb_pairs)} potential entity pairs...")

        for entry1, entry2, score in entity_vdb_pairs:
            record = process_entity_pair_simple(entry1, entry2, score)
            if record:
                all_mapping_records.append(record)
        processed_entity_count = len(all_mapping_records)
        logger.info(f"Added {processed_entity_count} entity mapping records.")

        # --- Edge Mapping ---
        logger.info(f"Finding similar EDGE pairs (Threshold: {threshold})...")
        edge_vdb_pairs = await vdb_storage.find_similar_vdb_pairs(
            graph1_edge_vdb_path,
            graph2_edge_vdb_path,
            threshold,
            max_threshold=max_threshold
        )
        logger.info(f"Processing {len(edge_vdb_pairs)} potential edge pairs...")

        edge_records_added = 0
        for entry1, entry2, score in edge_vdb_pairs:
             record = process_edge_pair_simple(entry1, entry2, score)
             if record:
                 all_mapping_records.append(record)
                 edge_records_added += 1

        logger.info(f"Added {edge_records_added} edge mapping records.")


        # --- Save Combined Mappings ---
        logger.info(f"Saving {len(all_mapping_records)} total mappings to {output_path}")
        all_mapping_records.sort(key=lambda x: x['similarity_score'], reverse=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_mapping_records, f, ensure_ascii=False, indent=2)
        except Exception as e:
             logger.error(f"Failed to write mapping results to JSON file {output_path}: {e}")
             raise # Re-raise saving error

        logger.info(f"Element mapping completed successfully.")
        return output_path

    except FileNotFoundError as e:
        logger.error(f"Prerequisite VDB file not found: {e}")
        raise # Re-raise file not found error
    except Exception as e:
        logger.error(f"Error in element mapping pipeline: {e}", exc_info=True)
        raise # Re-raise other errors

def process_entity_pair_simple(entry1: Dict[str, Any], entry2: Dict[str, Any], score: float) -> Optional[Dict[str, Any]]:
    """Creates a mapping record for an entity pair using only VDB entry data."""
    try:
        entity_name_1 = entry1.get("entity_name")
        chunk_id_1 = entry1.get("chunk_id")
        desc_1 = entry1.get("description", "N/A")

        entity_name_2 = entry2.get("entity_name")
        chunk_id_2 = entry2.get("chunk_id")
        desc_2 = entry2.get("description", "N/A")

        if not all([entity_name_1, chunk_id_1, entity_name_2, chunk_id_2]):
            logger.debug(f"Skipping entity pair due to missing core fields. E1: {entry1}, E2: {entry2}")
            return None

        mapping_record = {
            "type": "entity",
            "similarity_score": float(score),
            "graph1_entity_name": entity_name_1,
            "graph1_trigger_chunk_id": chunk_id_1,
            "graph1_trigger_description": desc_1,
            "graph2_entity_name": entity_name_2,
            "graph2_trigger_chunk_id": chunk_id_2,
            "graph2_trigger_description": desc_2,
        }
        return mapping_record

    except Exception as e:
        entity1_repr = entry1.get('entity_name', entry1.get('__id__', 'Unknown'))
        entity2_repr = entry2.get('entity_name', entry2.get('__id__', 'Unknown'))
        logger.error(f"Error processing simple entity pair ({entity1_repr}, {entity2_repr}): {e}", exc_info=False) # Less verbose logging in loop
        return None

def process_edge_pair_simple(entry1: Dict[str, Any], entry2: Dict[str, Any], score: float) -> Optional[Dict[str, Any]]:
    """Creates a mapping record for an edge pair using only VDB entry data."""
    try:
        src_1 = entry1.get("src_id")
        tgt_1 = entry1.get("tgt_id")
        chunk_id_1 = entry1.get("chunk_id")
        desc_1 = entry1.get("description", "N/A")

        src_2 = entry2.get("src_id")
        tgt_2 = entry2.get("tgt_id")
        chunk_id_2 = entry2.get("chunk_id")
        desc_2 = entry2.get("description", "N/A")

        if not all([src_1, tgt_1, chunk_id_1, src_2, tgt_2, chunk_id_2]):
            logger.debug(f"Skipping edge pair due to missing core fields. E1: {entry1}, E2: {entry2}")
            return None

        mapping_record = {
            "type": "edge",
            "similarity_score": float(score),
            "graph1_src_entity": src_1,
            "graph1_tgt_entity": tgt_1,
            "graph1_trigger_chunk_id": chunk_id_1,
            "graph1_trigger_description": desc_1,
            "graph2_src_entity": src_2,
            "graph2_tgt_entity": tgt_2,
            "graph2_trigger_chunk_id": chunk_id_2,
            "graph2_trigger_description": desc_2,
        }
        return mapping_record

    except Exception as e:
        edge1_id = f"({entry1.get('src_id', 'N/A')}, {entry1.get('tgt_id', 'N/A')})"
        edge2_id = f"({entry2.get('src_id', 'N/A')}, {entry2.get('tgt_id', 'N/A')})"
        logger.error(f"Error processing simple edge pair ({edge1_id}, {edge2_id}): {e}", exc_info=False) # Less verbose logging in loop
        return None

# --- Utility Functions for Reading Mappings ---

def find_mapped_entity_description(
    mapping_file_path: str,
    input_entity_name: str,
    input_description: str,
    source_graph_key_prefix: str = "graph1",
    target_graph_key_prefix: str = "graph2"
) -> Optional[Tuple[str, str]]:
    """
    Finds the corresponding entity and description in the target graph based on
    input from the source graph using the pre-computed mapping file.

    Args:
        mapping_file_path: Path to the JSON mapping file.
        input_entity_name: The entity name from the source graph.
        input_description: The specific description from the source graph's VDB entry.
        source_graph_key_prefix: Prefix for keys related to the source graph ("graph1" or "graph2").
        target_graph_key_prefix: Prefix for keys related to the target graph ("graph2" or "graph1").

    Returns:
        A tuple containing (mapped_entity_name, mapped_description) from the target graph,
        or None if no match is found.
    """
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            all_mappings = json.load(f)
    except Exception as e:
        logger.error(f"Error reading or parsing mapping file {mapping_file_path}: {e}")
        return None

    normalized_input_entity = str(input_entity_name).strip().lower()
    normalized_input_desc = str(input_description).strip()

    source_entity_key = "graph1_entity_name"
    source_desc_key = "graph1_trigger_description"
    target_entity_key = "graph2_entity_name"
    target_desc_key = "graph2_trigger_description"
    chunkid = "graph2_trigger_chunk_id"
    
    for mapping in all_mappings:
        if mapping.get("type") == "entity":
            source_entity = mapping.get(source_entity_key)
            source_desc = mapping.get(source_desc_key)

            if source_entity is not None and source_desc is not None:
                normalized_source_entity = str(source_entity).strip().lower()
                normalized_source_desc = str(source_desc).strip()

                if (normalized_source_entity == normalized_input_entity and
                    normalized_source_desc == normalized_input_desc):

                    mapped_entity = mapping.get(target_entity_key)
                    mapped_desc = mapping.get(target_desc_key)
                    mapped_chunkid = mapping.get(chunkid)
                    if mapped_entity is not None and mapped_desc is not None:
                        logger.debug(f"Found entity match: '{input_entity_name}' -> '{mapped_entity}'")
                        return str(mapped_entity), str(mapped_desc), str(mapped_chunkid)
                    else:
                        logger.warning(f"Source matched but target keys missing: {mapping}")

    logger.debug(f"No exact entity match found for '{input_entity_name}' / '{input_description}'.")
    return None


def find_mapped_edge_description(
    mapping_file_path: str,
    input_src_entity: str,
    input_tgt_entity: str,
    input_description: str,
    source_graph_key_prefix: str = "graph1",
    target_graph_key_prefix: str = "graph2"
) -> Optional[Tuple[str, str, str]]:
    """
    Finds the corresponding edge details (src, tgt, description) in the target graph.

    Args:
        mapping_file_path: Path to the JSON mapping file.
        input_src_entity: The source entity name from the source graph.
        input_tgt_entity: The target entity name from the source graph.
        input_description: The specific edge description from the source graph's VDB entry.
        source_graph_key_prefix: Prefix for source graph keys.
        target_graph_key_prefix: Prefix for target graph keys.

    Returns:
        A tuple (mapped_src_entity, mapped_tgt_entity, mapped_description) or None.
    """
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            all_mappings = json.load(f)
    except Exception as e:
        logger.error(f"Error reading or parsing mapping file {mapping_file_path}: {e}")
        return None

    normalized_input_src = str(input_src_entity).strip().lower()
    normalized_input_tgt = str(input_tgt_entity).strip().lower()
    normalized_input_desc = str(input_description).strip()

    source_src_key = "graph1_src_entity"
    source_tgt_key = "graph1_tgt_entity"
    source_desc_key = "graph1_trigger_description"
    
    target_src_key = "graph2_src_entity"
    target_tgt_key = "graph2_tgt_entity"
    target_desc_key = "graph2_trigger_description"

    chunkid = "graph2_trigger_chunk_id"

    for mapping in all_mappings:
        if mapping.get("type") == "edge":
            source_src = mapping.get(source_src_key)
            source_tgt = mapping.get(source_tgt_key)
            source_desc = mapping.get(source_desc_key)

            if source_src is not None and source_tgt is not None and source_desc is not None:
                normalized_source_src = str(source_src).strip().lower()
                normalized_source_tgt = str(source_tgt).strip().lower()
                normalized_source_desc = str(source_desc).strip()

                if (normalized_source_src == normalized_input_src and
                    normalized_source_tgt == normalized_input_tgt and
                    normalized_source_desc == normalized_input_desc):

                    mapped_src = mapping.get(target_src_key)
                    mapped_tgt = mapping.get(target_tgt_key)
                    mapped_desc = mapping.get(target_desc_key)
                    mapped_chunkid = mapping.get(chunkid)
                    if mapped_src is not None and mapped_tgt is not None and mapped_desc is not None:
                        logger.debug(f"Found edge match: ({input_src_entity} -> {input_tgt_entity}) -> ({mapped_src} -> {mapped_tgt})")
                        return str(mapped_src), str(mapped_tgt), str(mapped_desc), str(mapped_chunkid)
                    else:
                        logger.warning(f"Source matched but target keys missing: {mapping}")

    logger.debug(f"No exact edge match found for ({input_src_entity} -> {input_tgt_entity}) / '{input_description}'.")
    return None


# --- Main Execution Block ---

def main():
    """Parses arguments and runs the element mapping."""
    parser = argparse.ArgumentParser(description="Map entities and edges between two KGs using VDB similarity")
    # Graph Naming
    parser.add_argument("--graph1_name", required=True, help="Name for graph 1 (e.g., 'en')")
    parser.add_argument("--graph2_name", required=True, help="Name for graph 2 (e.g., 'vi')")
    # VDB Paths
    parser.add_argument("--g1_entity_vdb", required=True, help="Path to graph 1's entity VDB file")
    parser.add_argument("--g2_entity_vdb", required=True, help="Path to graph 2's entity VDB file")
    parser.add_argument("--g1_edge_vdb", required=True, help="Path to graph 1's edge VDB file")
    parser.add_argument("--g2_edge_vdb", required=True, help="Path to graph 2's edge VDB file")
    # Output and Parameters
    parser.add_argument("--output", required=True, help="Directory to save mapping results")
    parser.add_argument("--threshold", type=float, default=0.8, help="Similarity threshold")
    parser.add_argument("--embedding_dim", type=int, default=1024, help="Dimension of embeddings used in VDBs")
    parser.add_argument("--max_threshold", type=float, default=None, help="Similarity threshold")

    args = parser.parse_args()

    # Config for NanoVectorDBStorage (optional, might be needed internally)
    vdb_storage_config = {
        "working_dir": args.output, # May not be strictly needed
        "embedding_batch_num": 32, # Likely not used here
        "vector_db_storage_cls_kwargs": {
            # Threshold is passed directly to find_similar_vdb_pairs
            "cosine_better_than_threshold": args.threshold
        }
    }

    # Run mapping
    try:
        # Use asyncio.run() for the top-level async function
        output_file = asyncio.run(
            map_elements_between_graphs(
                graph1_name=args.graph1_name,
                graph2_name=args.graph2_name,
                graph1_entity_vdb_path=args.g1_entity_vdb,
                graph2_entity_vdb_path=args.g2_entity_vdb,
                graph1_edge_vdb_path=args.g1_edge_vdb,
                graph2_edge_vdb_path=args.g2_edge_vdb,
                output_dir=args.output,
                threshold=args.threshold,
                vdb_config=vdb_storage_config,
                embedding_dim=args.embedding_dim,
                max_threshold=args.max_threshold
            )
        )
        logger.info(f"Mapping process finished. Results are in: {output_file}")
    except Exception as e:
        logger.error(f"Mapping script failed: {e}", exc_info=True)
        import sys
        sys.exit(1)

if __name__ == "__main__":
    # Configure logging for the script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main() 