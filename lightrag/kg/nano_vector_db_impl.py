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
from typing import Any, final, Dict, List
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

    async def query(self, query: str, top_k: int) -> list[dict[str, Any]]:
        embedding = await self.embedding_func([query])
        embedding = embedding[0]
        results = self._client.query(
            query=embedding,
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
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

    async def find_and_save_entity_pairs(self, vdb1_path: str, vdb2_path: str, output_path: str, threshold: float = 0.8) -> None:
        """Find similar entity pairs between two VDBs and save to JSON file.
        
        Args:
            vdb1_path: Path to first VDB file
            vdb2_path: Path to second VDB file
            output_path: Path to save pairs JSON file
            threshold: Similarity threshold (default: 0.8)
        """
        try:
            # Load VDBs
            vdb1 = load_entity_vdb(vdb1_path)
            vdb2 = load_entity_vdb(vdb2_path)
            
            # Find similar pairs
            pairs = find_entity_pairs(vdb1, vdb2, threshold)
            
            # Save pairs to JSON file
            save_pairs_to_json(pairs, output_path)
            logger.info(f"Found and saved {len(pairs)} similar entity pairs to {output_path}")
            
        except Exception as e:
            logger.error(f"Error finding and saving entity pairs: {e}")
            raise
