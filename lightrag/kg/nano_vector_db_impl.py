# import asyncio
# import os
# from typing import Any, final
# from dataclasses import dataclass
# import numpy as np

# import time

# from lightrag.utils import (
#     logger,
#     compute_mdhash_id,
# )
# import pipmaster as pm
# from lightrag.base import (
#     BaseVectorStorage,
# )

# if not pm.is_installed("nano-vectordb"):
#     pm.install("nano-vectordb")

# try:
#     from nano_vectordb import NanoVectorDB
# except ImportError as e:
#     raise ImportError(
#         "`nano-vectordb` library is not installed. Please install it via pip: `pip install nano-vectordb`."
#     ) from e


# @final
# @dataclass
# class NanoVectorDBStorage(BaseVectorStorage):
#     def __post_init__(self):
#         # Initialize lock only for file operations
#         self._save_lock = asyncio.Lock()
#         # Use global config value if specified, otherwise use default
#         kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
#         cosine_threshold = kwargs.get("cosine_better_than_threshold")
#         if cosine_threshold is None:
#             raise ValueError(
#                 "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
#             )
#         self.cosine_better_than_threshold = cosine_threshold

#         self._client_file_name = os.path.join(
#             self.global_config["working_dir"], f"vdb_{self.namespace}.json"
#         )
#         self._max_batch_size = self.global_config["embedding_batch_num"]
#         self._client = NanoVectorDB(
#             self.embedding_func.embedding_dim, storage_file=self._client_file_name
#         )

#     async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
#         logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
#         if not len(data):
#             logger.warning("You insert an empty data to vector DB")
#             return []

#         current_time = time.time()
#         list_data = [
#             {
#                 "__id__": k,
#                 "__created_at__": current_time,
#                 **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
#             }
#             for k, v in data.items()
#         ]
#         contents = [v["content"] for v in data.values()]
#         batches = [
#             contents[i : i + self._max_batch_size]
#             for i in range(0, len(contents), self._max_batch_size)
#         ]

#         embedding_tasks = [self.embedding_func(batch) for batch in batches]
#         embeddings_list = await asyncio.gather(*embedding_tasks)

#         embeddings = np.concatenate(embeddings_list)
#         if len(embeddings) == len(list_data):
#             for i, d in enumerate(list_data):
#                 d["__vector__"] = embeddings[i]
#             results = self._client.upsert(datas=list_data)
#             return results
#         else:
#             # sometimes the embedding is not returned correctly. just log it.
#             logger.error(
#                 f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
#             )

#     async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
#         embedding = await self.embedding_func([query])
#         embedding = embedding[0]
#         results = self._client.query(
#             query=embedding,
#             top_k=top_k,
#             better_than_threshold=self.cosine_better_than_threshold,
#         )
#         results = [
#             {
#                 **dp,
#                 "id": dp["__id__"],
#                 "distance": dp["__metrics__"],
#                 "created_at": dp.get("__created_at__"),
#             }
#             for dp in results
#         ]
#         return results

#     @property
#     def client_storage(self):
#         return getattr(self._client, "_NanoVectorDB__storage")

#     async def delete(self, ids: list[str]):
#         """Delete vectors with specified IDs

#         Args:
#             ids: List of vector IDs to be deleted
#         """
#         try:
#             self._client.delete(ids)
#             logger.info(
#                 f"Successfully deleted {len(ids)} vectors from {self.namespace}"
#             )
#         except Exception as e:
#             logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

#     async def delete_entity(self, entity_name: str) -> None:
#         try:
#             entity_id = compute_mdhash_id(entity_name, prefix="ent-")
#             logger.debug(
#                 f"Attempting to delete entity {entity_name} with ID {entity_id}"
#             )
#             # Check if the entity exists
#             if self._client.get([entity_id]):
#                 await self.delete([entity_id])
#                 logger.debug(f"Successfully deleted entity {entity_name}")
#             else:
#                 logger.debug(f"Entity {entity_name} not found in storage")
#         except Exception as e:
#             logger.error(f"Error deleting entity {entity_name}: {e}")

#     async def delete_entity_relation(self, entity_name: str) -> None:
#         try:
#             relations = [
#                 dp
#                 for dp in self.client_storage["data"]
#                 if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
#             ]
#             logger.debug(f"Found {len(relations)} relations for entity {entity_name}")
#             ids_to_delete = [relation["__id__"] for relation in relations]

#             if ids_to_delete:
#                 await self.delete(ids_to_delete)
#                 logger.debug(
#                     f"Deleted {len(ids_to_delete)} relations for {entity_name}"
#                 )
#             else:
#                 logger.debug(f"No relations found for entity {entity_name}")
#         except Exception as e:
#             logger.error(f"Error deleting relations for {entity_name}: {e}")

#     async def index_done_callback(self) -> None:
#         async with self._save_lock:
#             self._client.save()
            
import asyncio
import os
from typing import Any, final, Dict, List,TypedDict, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import json
import base64
from tqdm import trange
import time

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
import pipmaster as pm
from lightrag.base import (
    BaseVectorStorage,
)

if not pm.is_installed("nano-vectordb"):
    pm.install("nano-vectordb")

try:
    from nano_vectordb import NanoVectorDB
except ImportError as e:
    raise ImportError(
        "`nano-vectordb` library is not installed. Please install it via pip: `pip install nano-vectordb`."
    ) from e

Data = TypedDict("Data", {"__id__": str, "__vector__": np.ndarray})
ConditionLambda = Callable[[Data], bool]

def buffer_string_to_array(base64_str: str, dtype=np.float32) -> np.ndarray:
    """Convert base64 encoded string to numpy array."""
    return np.frombuffer(base64.b64decode(base64_str), dtype=dtype)

def normalize(a: np.ndarray) -> np.ndarray:
    """Normalize array to unit length."""
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

def load_matrix_from_json(file_name: str) -> np.ndarray:
    """Load a matrix from a JSON file.
    
    Args:
        file_name: Path to the JSON file containing the matrix
        
    Returns:
        numpy.ndarray: The loaded matrix
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)
    
    if "matrix" not in data:
        raise ValueError("JSON file must contain a 'matrix' field")
    
    return buffer_string_to_array(data["matrix"]).reshape(-1, data["embedding_dim"])

def find_similar_pairs(matrix1: np.ndarray, matrix2: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int, float]]:
    """Find similar pairs between two matrices using cosine similarity.
    
    Args:
        matrix1: First matrix of shape (n, d)
        matrix2: Second matrix of shape (m, d)
        threshold: Similarity threshold (default: 0.8)
        
    Returns:
        list[tuple[int, int, float]]: List of (index1, index2, similarity) tuples
    """
    # Normalize matrices
    matrix1_norm = normalize(matrix1)
    matrix2_norm = normalize(matrix2)
    
    # Compute cosine similarity
    similarity = np.dot(matrix1_norm, matrix2_norm.T)
    
    # Find pairs above threshold
    pairs = []
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            if similarity[i, j] >= threshold:
                pairs.append((i, j, similarity[i, j]))
    
    return pairs

def load_entity_vdb(file_path: str) -> Dict[str, Any]:
    """Load entity VDB from file.
    
    Args:
        file_path: Path to the VDB file
        
    Returns:
        Dict containing VDB data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    return data

def find_entity_pairs(vdb1: Dict[str, Any], vdb2: Dict[str, Any], threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Find similar entity pairs between two VDBs.
    
    Args:
        vdb1: First VDB data
        vdb2: Second VDB data
        threshold: Similarity threshold
        
    Returns:
        List of similar entity pairs with metadata
    """
    # Extract matrices and entity data
    matrix1 = buffer_string_to_array(vdb1["matrix"]).reshape(-1, vdb1["embedding_dim"])
    matrix2 = buffer_string_to_array(vdb2["matrix"]).reshape(-1, vdb2["embedding_dim"])
    
    # Normalize matrices
    matrix1_norm = normalize(matrix1)
    matrix2_norm = normalize(matrix2)
    
    # Compute cosine similarity
    import time
    start_time  = time.time()
    print(matrix1_norm.shape)
    print(matrix1_norm.shape)
    similarity = np.dot(matrix1_norm, matrix2_norm.T)
    print("Done similarity...")
    
    # Find pairs above threshold
    # pairs = []
    # for i in trange(similarity.shape[0]):
    #     for j in trange(similarity.shape[1]):
    #         if similarity[i, j] >= threshold:
    #             entity1 = vdb1["data"][i]
    #             entity2 = vdb2["data"][j]
    #             pairs.append({
    #                 "entity1": {
    #                     "id": entity1["__id__"],
    #                     "name": entity1.get("entity_name", ""),
    #                     "type": entity1.get("type", ""),
    #                     "index": i
    #                 },
    #                 "entity2": {
    #                     "id": entity2["__id__"],
    #                     "name": entity2.get("entity_name", ""),
    #                     "type": entity2.get("type", ""),
    #                     "index": j
    #                 },
    #                 "similarity": float(similarity[i, j])
    #             })
    
    # return pairs
    matches = np.where(similarity >= threshold)
    print("Time for find match: ", time.time() - start_time)
    pairs = []
    for i, j in zip(matches[0], matches[1]):
        entity1 = vdb1["data"][i]
        entity2 = vdb2["data"][j]
        pairs.append({
            "entity1": {
                "id": entity1["__id__"],
                "name": entity1.get("entity_name", ""),
                "type": entity1.get("type", ""),
                "index": i
            },
            "entity2": {
                "id": entity2["__id__"],
                "name": entity2.get("entity_name", ""),
                "type": entity2.get("type", ""),
                "index": j
            },
            "similarity": float(similarity[i, j])
        })
    print("Time for save pair: ", time.time() - start_time)
    return pairs

def save_pairs_to_json(pairs: List[Dict[str, Any]], output_file: str) -> None:
    """Save entity pairs to JSON file in format entity_name: [list of related entities].
    
    Args:
        pairs: List of entity pairs
        output_file: Path to output JSON file
    """
    # Create a dictionary to store entity mappings
    entity_mapping = {}
    
    # Process each pair and build the entity mapping
    for pair in pairs:
        entity1_name = pair["entity1"]["name"]
        entity2_name = pair["entity2"]["name"]
        
        # Add entity2 to entity1's list
        if entity1_name not in entity_mapping:
            entity_mapping[entity1_name] = []
        if entity2_name != entity1_name:  # Remove self-references
            entity_mapping[entity1_name].append(entity2_name)
        
        # Add entity1 to entity2's list
        if entity2_name not in entity_mapping:
            entity_mapping[entity2_name] = []
        if entity1_name != entity2_name:  # Remove self-references
            entity_mapping[entity2_name].append(entity1_name)
    
    # Save the mapping to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump( entity_mapping, f, ensure_ascii=False, indent=2)

def load_vdb_data(file_path: str) -> Dict[str, Any]:
    """Load VDB data from file, handling potential missing file."""
    if not os.path.exists(file_path):
        logger.error(f"VDB file not found: {file_path}")
        raise FileNotFoundError(f"File {file_path} not found")

    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        # Basic validation
        if "matrix" not in data or "data" not in data or "embedding_dim" not in data:
            raise ValueError(f"VDB file {file_path} is missing required fields ('matrix', 'data', 'embedding_dim')")
        # Ensure data is a list
        if not isinstance(data.get("data"), list):
             raise ValueError(f"Field 'data' in VDB file {file_path} must be a list.")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise ValueError(f"Invalid JSON format in {file_path}") from e
    except Exception as e:
        logger.error(f"Error loading VDB file {file_path}: {e}")
        raise

def find_vdb_entry_pairs(vdb1: Dict[str, Any], vdb2: Dict[str, Any], threshold: float = 0.8, max_threshold: float = None) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    Find similar entry pairs between two VDBs based on cosine similarity.

    Args:
        vdb1: First loaded VDB data dictionary.
        vdb2: Second loaded VDB data dictionary.
        threshold: Similarity threshold.

    Returns:
        List of tuples: [(vdb1_entry, vdb2_entry, similarity_score)]
        where vdb1_entry and vdb2_entry are the full dictionary entries from vdb1['data'] and vdb2['data'].
    """
    try:
        # Extract matrices and check dimensions
        if vdb1["embedding_dim"] != vdb2["embedding_dim"]:
            raise ValueError("Embedding dimensions of the two VDBs do not match.")
        embedding_dim = vdb1["embedding_dim"]

        matrix1 = buffer_string_to_array(vdb1["matrix"]).reshape(-1, embedding_dim)
        matrix2 = buffer_string_to_array(vdb2["matrix"]).reshape(-1, embedding_dim)

        # Verify matrix rows match data length
        if matrix1.shape[0] != len(vdb1["data"]):
            raise ValueError(f"Matrix 1 row count ({matrix1.shape[0]}) does not match data length ({len(vdb1['data'])})")
        if matrix2.shape[0] != len(vdb2["data"]):
            raise ValueError(f"Matrix 2 row count ({matrix2.shape[0]}) does not match data length ({len(vdb2['data'])})")

        # Normalize matrices
        matrix1_norm = normalize(matrix1)
        matrix2_norm = normalize(matrix2)

        logger.info(f"Computing similarity between matrices of shape {matrix1_norm.shape} and {matrix2_norm.shape}")
        start_time = time.time()
        similarity = np.dot(matrix1_norm, matrix2_norm.T)
        logger.info(f"Similarity computation took {time.time() - start_time:.2f} seconds.")

        # Find pairs above threshold using np.where for efficiency
        if max_threshold is None:
            matches = np.where(similarity >= threshold)
        else:
            matches = np.where((similarity >= threshold) & (similarity < max_threshold))
        logger.info(f"Found {len(matches[0])} potential pairs above threshold {threshold}.")

        pairs = []
        start_time = time.time()
        # Use zip for efficient iteration over match indices
        for i, j in zip(matches[0], matches[1]):
            try:
                # Access the full data entries using the indices
                entry1 = vdb1["data"][i]
                entry2 = vdb2["data"][j]
                score = float(similarity[i, j])
                pairs.append((entry1, entry2, score))
            except IndexError:
                 logger.warning(f"Index out of bounds when accessing VDB data at indices ({i}, {j}). Skipping pair.")
            except Exception as e:
                 logger.error(f"Error processing pair at indices ({i}, {j}): {e}")

        logger.info(f"Pair processing took {time.time() - start_time:.2f} seconds. Returning {len(pairs)} pairs.")
        return pairs

    except KeyError as e:
        logger.error(f"Missing key in VDB data structure: {e}")
        raise ValueError("Invalid VDB data structure") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred in find_vdb_entry_pairs: {e}", exc_info=True)
        raise

@final
@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    def __post_init__(self):
        # Initialize lock only for file operations
        self._save_lock = asyncio.Lock()
        # Use global config value if specified, otherwise use default
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []

        current_time = time.time()
        list_data = [
            {
                "__id__": k,
                "__created_at__": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(self, query: str, top_k: int, filter_lambda: ConditionLambda = None) -> list[dict[str, Any]]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
            filter_lambda = filter_lambda
        )
        results = [
            {
                **dp,
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]
        return results

    @property
    def client_storage(self):
        return getattr(self._client, "_NanoVectorDB__storage")

    async def delete(self, ids: list[str]):
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            self._client.delete(ids)
            logger.info(
                f"Successfully deleted {len(ids)} vectors from {self.namespace}"
            )
        except Exception as e:
            logger.error(f"Error while deleting vectors from {self.namespace}: {e}")

    async def delete_entity(self, entity_name: str) -> None:
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"Attempting to delete entity {entity_name} with ID {entity_id}"
            )
            # Check if the entity exists
            if self._client.get([entity_id]):
                await self.delete([entity_id])
                logger.debug(f"Successfully deleted entity {entity_name}")
            else:
                logger.debug(f"Entity {entity_name} not found in storage")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        try:
            relations = [
                dp
                for dp in self.client_storage["data"]
                if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
            ]
            logger.debug(f"Found {len(relations)} relations for entity {entity_name}")
            ids_to_delete = [relation["__id__"] for relation in relations]

            if ids_to_delete:
                await self.delete(ids_to_delete)
                logger.debug(
                    f"Deleted {len(ids_to_delete)} relations for {entity_name}"
                )
            else:
                logger.debug(f"No relations found for entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting relations for {entity_name}: {e}")

    async def index_done_callback(self) -> None:
        async with self._save_lock:
            self._client.save()

    async def find_and_store_similar_pairs(self, file1: str, file2: str, threshold: float = 0.8) -> None:
        """Load two matrices from JSON files, find similar pairs, and store them.
        
        Args:
            file1: Path to first JSON file
            file2: Path to second JSON file
            threshold: Similarity threshold (default: 0.8)
        """
        try:
            # Load matrices
            matrix1 = load_matrix_from_json(file1)
            matrix2 = load_matrix_from_json(file2)
            
            # Find similar pairs
            pairs = find_similar_pairs(matrix1, matrix2, threshold)
            
            # Store pairs in vector DB
            pair_data = {}
            for i, j, similarity in pairs:
                pair_id = f"pair_{i}_{j}"
                pair_data[pair_id] = {
                    "content": f"Similar pair {i}-{j}",
                    "similarity": similarity,
                    "index1": i,
                    "index2": j
                }
            
            # Store pairs in vector DB
            # await self.upsert(pair_data)
            # logger.info(f"Stored {len(pairs)} similar pairs")
            
        except Exception as e:
            logger.error(f"Error finding and storing similar pairs: {e}")
            raise

    async def find_similar_vdb_pairs(self, vdb1_path: str, vdb2_path: str, threshold: float = 0.8, max_threshold: float = None) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Loads two VDB files and finds pairs of entries with similarity above a threshold.

        Args:
            vdb1_path: Path to the first VDB file (.json).
            vdb2_path: Path to the second VDB file (.json).
            threshold: Similarity threshold.

        Returns:
            List of tuples: [(vdb1_entry, vdb2_entry, similarity_score)]
        """
        try:
            logger.info(f"Loading VDB file 1: {vdb1_path}")
            vdb1 = load_vdb_data(vdb1_path)
            logger.info(f"Loading VDB file 2: {vdb2_path}")
            vdb2 = load_vdb_data(vdb2_path)

            logger.info("Finding similar VDB entry pairs...")
            # This is a computationally intensive task, run it in a separate thread
            # if it blocks the event loop for too long, although numpy operations
            # often release the GIL. For simplicity, calling directly here.
            # Consider asyncio.to_thread if performance becomes an issue.
            pairs = find_vdb_entry_pairs(vdb1, vdb2, threshold, max_threshold)

            logger.info(f"Found {len(pairs)} similar pairs between VDBs above threshold {threshold}.")
            return pairs

        except FileNotFoundError as e:
            logger.error(f"VDB file not found: {e}")
            return [] # Return empty list if a file is missing
        except ValueError as e:
             logger.error(f"Error processing VDB data: {e}")
             return [] # Return empty list on data errors
        except Exception as e:
            logger.error(f"An unexpected error occurred in find_similar_vdb_pairs: {e}", exc_info=True)
            return [] # Return empty list on unexpected errors
