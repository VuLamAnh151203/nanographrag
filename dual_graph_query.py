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
from lightrag.base import QueryParam
from lightrag.lightrag_old import always_get_an_event_loop

# Import entity mapping
# from entity_mapping import find_mapped_entity_description, find_mapped_edge_description
# Import entity mapping
from optimized_entity_mapping import find_mapped_entity_description, find_mapped_edge_description, BatchGraphMapper


import os
import sys
import time
import json
import random
import asyncio
from typing import List


from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import detect_language


with open("/home/hungpv/projects/list_key_open_router/openrouter_full.json", 'r', encoding='utf-8') as f:
    OPENROUTER_API_KEYS = json.load(f)
OPENROUTER_API_KEYS = OPENROUTER_API_KEYS[700:]
print(f"Tổng số API keys: {len(OPENROUTER_API_KEYS)}")

# Thêm lock để xử lý đồng thời
class APIManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()
        self.permanently_banned_keys = set()  # Thêm set để lưu các key bị cấm vĩnh viễn do lỗi 429
        self.last_switch_time = {}
        self.lock = asyncio.Lock()  # Thêm lock để xử lý đồng thời
        
    async def get_current_api_key(self):
        async with self.lock:
            return self.api_keys[self.current_key_index]
    
    async def switch_to_next_key(self, mark_current_as_failed=False):
        async with self.lock:
            # Chỉ đánh dấu key là failed nếu có yêu cầu
            if mark_current_as_failed:
                self.failed_keys.add(self.current_key_index)
                self.last_switch_time[self.current_key_index] = time.time()
            
            # Lưu index hiện tại để không chọn lại key đó
            previous_index = self.current_key_index
            
            # Cập nhật danh sách available keys
            available_keys = []
            for idx in range(len(self.api_keys)):
                # Không chọn lại key hiện tại, các key đã thất bại tạm thời, và các key bị cấm vĩnh viễn
                if (idx != previous_index and 
                    idx not in self.failed_keys and 
                    idx not in self.permanently_banned_keys):
                    available_keys.append(idx)
                # Phục hồi key đã qua thời gian chờ (chỉ cho key thất bại tạm thời, không cho key bị cấm vĩnh viễn)
                elif idx in self.last_switch_time and idx != previous_index and idx not in self.permanently_banned_keys:
                    if time.time() - self.last_switch_time[idx] > 600:  # 10 phút
                        if idx in self.failed_keys:
                            self.failed_keys.remove(idx)
                        available_keys.append(idx)
            
            # Nếu không còn key nào khả dụng, dừng chương trình
            if not available_keys:
                print("CẢNH BÁO: Không còn API key nào khả dụng!")
                print(f"- Tổng số key: {len(self.api_keys)}")
                print(f"- Key thất bại tạm thời: {len(self.failed_keys)}")
                print(f"- Key bị cấm vĩnh viễn (429): {len(self.permanently_banned_keys)}")
                raise RuntimeError("Không còn API key khả dụng. Chương trình buộc phải dừng lại.")
            
            # Chọn một key ngẫu nhiên từ danh sách khả dụng
            self.current_key_index = random.choice(available_keys)
            print(f"Đã chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
            return self.api_keys[self.current_key_index]
    
    async def mark_key_failed(self, key_index):
        async with self.lock:
            self.failed_keys.add(key_index)
            self.last_switch_time[key_index] = time.time()
            
    async def mark_key_rate_limited(self, key_index):
        """Đánh dấu key bị cấm vĩnh viễn do lỗi rate limit (429)"""
        async with self.lock:
            self.permanently_banned_keys.add(key_index)
            # Xóa khỏi danh sách key thất bại tạm thời nếu có
            if key_index in self.failed_keys:
                self.failed_keys.remove(key_index)
            print(f"API key {self.api_keys[key_index][:5]} đã bị đánh dấu là cấm vĩnh viễn do lỗi 429")

# Khởi tạo API Manager
api_manager = APIManager(OPENROUTER_API_KEYS)

WORKING_DIR = "./new_zalo_graph_single_en_without_embedding"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Sử dụng biến global cho API key hiện tại
current_api_key = OPENROUTER_API_KEYS[0]  # Khởi tạo với key đầu tiên

# Semaphore để giới hạn số lượng request đồng thời
api_semaphore = asyncio.Semaphore(20)  # Giới hạn 5 requests đồng thời

# Cập nhật hàm llm_model_func để xử lý lỗi 429
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    global current_api_key
    max_retries = min(len(OPENROUTER_API_KEYS), 30)  # Giới hạn số lần retry
    retry_count = 0
    
    # Định nghĩa các chuỗi để nhận diện lỗi rate limit
    rate_limit_indicators = [
        "429",
        "too many requests", 
        "rate limit", 
        "quota exceeded", 
        "resource_exhausted",
        "provider returned error"  # OpenRouter thường trả về lỗi này khi có lỗi 429
    ]
    
    # Sử dụng semaphore để giới hạn số lượng request đồng thời
    async with api_semaphore:
        # Luôn chuyển sang key mới trước khi gọi API
        current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
        os.environ["LLM_BINDING_API_KEY"] = current_api_key
        
        while retry_count < max_retries:
            try:
                response = await openai_complete_if_cache(
                    "google/gemini-2.0-flash-exp:free",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=current_api_key, 
                    base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
                    **kwargs
                )
                
                # Kiểm tra xem response có chứa error không (đặc biệt là lỗi 429)
                if hasattr(response, 'error') and response.error is not None:
                    error_info = response.error
                    error_str = str(error_info)
                    print(f"Lỗi trong response: {error_str}")
                    
                    # Xác định xem có phải lỗi 429 không
                    is_rate_limit = False
                    if isinstance(error_info, dict) and 'code' in error_info and error_info['code'] == 429:
                        is_rate_limit = True
                    else:
                        error_lower = error_str.lower()
                        is_rate_limit = any(indicator in error_lower for indicator in rate_limit_indicators)
                    
                    if is_rate_limit:
                        # Nếu là lỗi 429, đánh dấu key này bị cấm vĩnh viễn
                        current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
                        await api_manager.mark_key_rate_limited(current_key_index)
                        
                        # Thử với key khác
                        retry_count += 1
                        if retry_count < max_retries:
                            await asyncio.sleep(0.5 * retry_count)
                            print(f"Thử lại lần {retry_count}/{max_retries}...")
                            current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
                            os.environ["LLM_BINDING_API_KEY"] = current_api_key
                            continue
                        else:
                            raise RuntimeError(f"Không thể hoàn thành yêu cầu sau {max_retries} lần thử")
                
                # Không đánh dấu key là failed khi thành công
                return response
                
            except Exception as e:
                error_str = str(e).lower()
                print(f"Lỗi với API key {current_api_key[:5]}: {str(e)}")
                
                # Kiểm tra xem có phải lỗi rate limit không
                is_rate_limit = any(indicator in error_str for indicator in rate_limit_indicators)
                
                # Lấy index của key hiện tại
                current_key_index = OPENROUTER_API_KEYS.index(current_api_key)
                
                if is_rate_limit:
                    # Nếu là lỗi rate limit, đánh dấu key này bị cấm vĩnh viễn
                    print(f"Phát hiện lỗi rate limit! API key {current_api_key[:5]} bị đánh dấu cấm vĩnh viễn.")
                    await api_manager.mark_key_rate_limited(current_key_index)
                else:
                    # Nếu không phải lỗi rate limit, đánh dấu key này thất bại tạm thời
                    await api_manager.mark_key_failed(current_key_index)
                
                retry_count += 1
                if retry_count < max_retries:
                    # Thêm một khoảng chờ nhỏ trước khi thử lại
                    await asyncio.sleep(0.5 * retry_count)  # Tăng dần thời gian chờ
                    print(f"Thử lại lần {retry_count}/{max_retries}...")
                    try:
                        current_api_key = await api_manager.switch_to_next_key(mark_current_as_failed=False)
                        os.environ["LLM_BINDING_API_KEY"] = current_api_key
                    except RuntimeError as e:
                        # Nếu không còn key nào khả dụng, dừng chương trình
                        print(str(e))
                        sys.exit(1)
                else:
                    print("Đã vượt quá số lần thử lại.")
                    raise RuntimeError(f"Không thể hoàn thành yêu cầu sau {max_retries} lần thử: {str(e)}")
    
    raise RuntimeError("Tất cả API keys đều đã thất bại")

print("Loading model...")
print("google/gemini-2.0-flash-exp:free")

async def embedding_func(texts, model):
    return model.encode(texts)["dense_vecs"]

# Chuyển lại sang hàm đồng bộ
def init_rag(working_dir, embedding_path, embedding_func_name, devices):
    print("Initializing LightRAG...")
    print(f"Using LLM model: google/gemini-2.0-pro-exp-02-05:free")
    
    # Khởi tạo đối tượng LightRAG
    from FlagEmbedding import BGEM3FlagModel

    # model = BGEM3FlagModel('/home/hungpv/projects/train_embedding/nanographrag/flag_embedding_train/test_encoder_only_m3_bge-m3_sd/checkpoint-90804',  
    #                    use_fp16=False, devices = ["cuda:0"],pooling_method = "mean")
    model = BGEM3FlagModel(embedding_path,  
                       use_fp16=False, devices = devices,pooling_method = "mean")
    
    embedding_func_name = embedding_func_name
    # embedding_func_name = "bge-m3-only-entity-name-only-edge-description"
# /home/hungpv/projects/TN/LIGHTRAG/zalo_wiki_graph_single_vi_v2/vdb_chunks_fine-tune-embedding.json

    return LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=512,
            func=lambda texts: embedding_func(texts, model),
        ),
        embedding_func_name = embedding_func_name
    )



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
        graph2_name: str = "graph2",
        mapping_sym_file: Optional[str] = None
    ):
        """Initialize dual graph querying with two LightRAG instances and entity mapping."""
        self.graph1 = graph1_instance
        self.graph2 = graph2_instance
        self.mapping_file = mapping_file_path
        self.output_dir = output_dir
        self.graph1_name = graph1_name
        self.graph2_name = graph2_name
        self.mapping_sym_file = mapping_sym_file
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate mapping file exists
        if not os.path.exists(mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found: {mapping_file_path}")

        # Initialize BatchGraphMapper for faster mapping
        self.batch_mapper = BatchGraphMapper(
            mapping_file_path=mapping_file_path,
            graph1_name=self.graph1_name,
            graph2_name=self.graph2_name
        )
        
        # Initialize BatchGraphMapper for symmetrical mapping if needed
        self.sym_mapper = None
        if mapping_sym_file and os.path.exists(mapping_sym_file):
            self.sym_mapper = BatchGraphMapper(
                mapping_file_path=mapping_sym_file,
                graph1_name=self.graph1_name,
                graph2_name=self.graph1_name
            )
        
        logger.info(f"DualGraphQuery initialized with {graph1_name} and {graph2_name}")
    
    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        sym: bool = False,
        max_threshold: Optional[float] = None,
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
            self.aquery(query_text, mode, param, save_results, sym=sym, max_threshold=max_threshold)
        )
    
    async def aquery(
        self,
        query_text: str,
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        sym: bool = False,
        index: Optional[int] = None,
        max_threshold: Optional[float] = None,
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
        mapped_graph1_results= None
        if sym:
            
            mapped_graph1_results = await  self._map_graph1_to_graph1(graph1_results, query_text=query_text, max_threshold=max_threshold)
        
        # Merge results
        consolidated_results = self._merge_results(
            graph1_results, 
            mapped_graph2_results,
            mapped_graph1_results,
        )
        
        # Calculate metrics
        
        
        # Create final result object
        result = {
            "query": query_text,
            "mode": param.mode,
            "candidates": consolidated_results,
            "graph2_raw_results": graph2_results,
        }
        
        # Save results if requested
        if save_results:
            self._save_results(result, query_text, index)
        
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
    
    # def _map_graph2_to_graph1(
    #     self, 
    #     graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]]
    # ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    #     """Map entities and edges from graph2 to graph1 using the mapping file."""
    #     candidates, hl_keywords, ll_keywords = graph2_results
    #     mapped_candidates = []
        
    #     for candidate in candidates:
    #         cand_type = candidate.get("retrieval_type")
            
    #         if cand_type == "node":
    #             # Map entity node
    #             entity_name = candidate.get("entity_name")
    #             description = candidate.get("description", "")
                
    #             # Use entity mapping
    #             mapped_result = find_mapped_entity_description(
    #                 self.mapping_file,
    #                 entity_name,
    #                 description,
    #                 source_graph_key_prefix=self.graph2_name,
    #                 target_graph_key_prefix=self.graph1_name
    #             )
                
    #             if mapped_result:
    #                 print("mapp cross ne")
    #                 mapped_entity, mapped_desc, mapped_chunkid = mapped_result
    #                 mapped_candidate = candidate.copy()
    #                 mapped_candidate["entity_name"] = mapped_entity
    #                 mapped_candidate["description"] = mapped_desc
    #                 mapped_candidate["original_entity"] = entity_name
    #                 mapped_candidate['retrieved_chunk_id'] = mapped_chunkid
    #                 mapped_candidate["mapping_status"] = "mapped_success"
    #                 # mapped_candidate["mapping_source"] = self.graph2_name
                    
    #                 mapped_candidates.append(mapped_candidate)
    #             else:
    #                 # Keep original if no mapping found, but mark it
    #                 candidate["mapping_status"] = "unmapped"
    #                 # mapped_candidates.append(candidate)
                    
    #         elif cand_type == "edge":
    #             # Map edge
    #             src_entity = candidate.get("src_id")
    #             tgt_entity = candidate.get("tgt_id")
    #             description = candidate.get("description", "")
                
    #             # Use edge mapping
    #             mapped_result = find_mapped_edge_description(
    #                 self.mapping_file,
    #                 src_entity,
    #                 tgt_entity,
    #                 description,
    #                 source_graph_key_prefix=self.graph2_name,
    #                 target_graph_key_prefix=self.graph1_name
    #             )
                
    #             if mapped_result:
    #                 print("mapp cross ne")
    #                 mapped_src, mapped_tgt, mapped_desc, mapped_chunkid = mapped_result
    #                 mapped_candidate = candidate.copy()
    #                 mapped_candidate["src_id"] = mapped_src
    #                 mapped_candidate["tgt_id"] = mapped_tgt
    #                 mapped_candidate["description"] = mapped_desc
    #                 mapped_candidate["original_src"] = src_entity
    #                 mapped_candidate["original_tgt"] = tgt_entity
    #                 mapped_candidate['retrieved_chunk_id'] = mapped_chunkid
    #                 mapped_candidate["mapping_status"] = "mapped_success"
    #                 # mapped_candidate["mapping_source"] = self.graph2_name
    #                 mapped_candidates.append(mapped_candidate)
    #             else:
    #                 # Keep original if no mapping found, but mark it
    #                 candidate["mapping_status"] = "unmapped"
    #                 # mapped_candidates.append(candidate)
    #         else:
    #             # For other types, keep as is
    #             mapped_candidates.append(candidate)
        
    #     return mapped_candidates, hl_keywords, ll_keywords

    def _map_graph2_to_graph1(
        self, 
        graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]]
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Map entities and edges from graph2 to graph1 using the mapping file."""
        # Use the batch mapper for better performance
        return self.batch_mapper.map_graph2_to_graph1(graph2_results)

    # async  def _map_graph1_to_graph1(
    #     self, 
    #     graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]],
    #     query_text: str = None,
    #     max_threshold: Optional[float] = None
    # ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    #     """Map entities and edges from graph2 to graph1 using the mapping file."""
    #     candidates, hl_keywords, ll_keywords = graph2_results
    #     mapped_candidates = []
        
    #     for candidate in candidates:
    #         cand_type = candidate.get("retrieval_type")
            
    #         if cand_type == "node":
    #             # Map entity node
    #             entity_name = candidate.get("entity_name")
    #             description = candidate.get("description", "")
                
    #             # Use entity mapping
    #             mapped_result = find_mapped_entity_description(
    #                 self.mapping_sym_file,
    #                 entity_name,
    #                 description,
    #                 source_graph_key_prefix=self.graph2_name,
    #                 target_graph_key_prefix=self.graph1_name
    #             )
                
    #             if mapped_result:
    #                 mapped_entity, mapped_desc, _ = mapped_result
    #                 chunk_id = await  self.graph1.a_most_relevant_text_chunks_from_nodes(query = query_text, list_nodes = [mapped_entity], threshold=max_threshold)
    #                 chunk_id = chunk_id[mapped_entity] if chunk_id else None
    #                 if chunk_id:
    #                     print("mapp sym ne")
    #                     mapped_candidate = candidate.copy()
    #                     mapped_candidate["entity_name"] = mapped_entity
    #                     mapped_candidate["description"] = mapped_desc
    #                     mapped_candidate["original_entity"] = entity_name
    #                     mapped_candidate['retrieved_chunk_id'] = chunk_id
    #                     mapped_candidate["mapping_status"] = "mapped_sym_success"
    #                     mapped_candidates.append(mapped_candidate)
    #             else:
    #                 # Keep original if no mapping found, but mark it
    #                 candidate["mapping_status"] = "unmapped"
    #                 # mapped_candidates.append(candidate)
                    
    #         elif cand_type == "edge":
    #             # Map edge
    #             src_entity = candidate.get("src_id")
    #             tgt_entity = candidate.get("tgt_id")
    #             description = candidate.get("description", "")
                
    #             # Use edge mapping
    #             mapped_result = find_mapped_edge_description(
    #                 self.mapping_file,
    #                 src_entity,
    #                 tgt_entity,
    #                 description,
    #                 source_graph_key_prefix=self.graph2_name,
    #                 target_graph_key_prefix=self.graph1_name
    #             )
                
    #             if mapped_result:
    #                 mapped_src, mapped_tgt, mapped_desc,_ = mapped_result
    #                 chunk_id = await  self.graph1.a_most_relevant_text_chunks_from_edges(query = query_text, list_edges = [(mapped_src,mapped_tgt)], threshold=max_threshold)
    #                 mapped_chunkid = chunk_id[(mapped_src,mapped_tgt)] if chunk_id else None
    #                 if mapped_chunkid:
    #                     print("mapp sym ne")
    #                     mapped_candidate = candidate.copy()
    #                     mapped_candidate["src_id"] = mapped_src
    #                     mapped_candidate["tgt_id"] = mapped_tgt
    #                     mapped_candidate["description"] = mapped_desc
    #                     mapped_candidate["original_src"] = src_entity
    #                     mapped_candidate["original_tgt"] = tgt_entity
    #                     mapped_candidate['retrieved_chunk_id'] = mapped_chunkid
    #                     mapped_candidate["mapping_status"] = "mapped_sym_success"
    #                     mapped_candidates.append(mapped_candidate)
    #             else:
    #                 # Keep original if no mapping found, but mark it
    #                 candidate["mapping_status"] = "unmapped"
    #                 # mapped_candidates.append(candidate)
    #         else:
    #             # For other types, keep as is
    #             mapped_candidates.append(candidate)
        
    #     return mapped_candidates, hl_keywords, ll_keywords

    async def _map_graph1_to_graph1(
        self, 
        graph1_results: Tuple[List[Dict[str, Any]], List[str], List[str]],
        query_text: str = None,
        max_threshold: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Map entities and edges from graph1 to graph1 using the sym mapping file."""
        candidates, hl_keywords, ll_keywords = graph1_results
        mapped_candidates = []
        
        # If no symmetrical mapper is available, return empty results
        if not self.sym_mapper:
            return mapped_candidates, hl_keywords, ll_keywords
        
        # First, collect all entities and edges that need to be mapped
        entities_to_map = []
        entity_indices = []
        edges_to_map = []
        edge_indices = []
        
        for idx, candidate in enumerate(candidates):
            cand_type = candidate.get("retrieval_type")
            
            if cand_type == "node":
                entity_name = candidate.get("entity_name")
                description = candidate.get("description", "")
                entities_to_map.append((entity_name, description, idx))
                entity_indices.append(idx)
            elif cand_type == "edge":
                src_entity = candidate.get("src_id")
                tgt_entity = candidate.get("tgt_id")
                description = candidate.get("description", "")
                edges_to_map.append((src_entity, tgt_entity, description, idx))
                edge_indices.append(idx)
        
        # Process entities in batch
        if entities_to_map:
            # First, find all mapped entities
            mapped_entities = {}
            for entity_name, description, idx in entities_to_map:
                mapped_result = self.sym_mapper.find_mapped_entity_description(entity_name, description)
                if mapped_result:
                    mapped_entity, mapped_desc, _ = mapped_result
                    mapped_entities[idx] = (mapped_entity, mapped_desc)
            
            # Then, collect all entity names that need chunk lookup
            entity_names_to_lookup = [ent_info[0] for idx, ent_info in mapped_entities.items()]
            
            # Batch lookup of chunks (this is an async operation that might be slow)
            chunk_lookups = {}
            if entity_names_to_lookup:
                chunk_results = await self.graph1.a_most_relevant_text_chunks_from_nodes(
                    query=query_text, 
                    list_nodes=entity_names_to_lookup, 
                    threshold=max_threshold
                )
                
                # Map back to the original indices
                for idx, (mapped_entity, mapped_desc) in mapped_entities.items():
                    if mapped_entity in chunk_results:
                        chunk_lookups[idx] = (mapped_entity, mapped_desc, chunk_results[mapped_entity])
            
            # Create mapped candidates
            for idx, lookup_result in chunk_lookups.items():
                mapped_entity, mapped_desc, chunk_id = lookup_result
                original_candidate = candidates[idx]
                entity_name = original_candidate.get("entity_name")
                
                # print("mapp sym ne")
                mapped_candidate = original_candidate.copy()
                mapped_candidate["entity_name"] = mapped_entity
                mapped_candidate["description"] = mapped_desc
                mapped_candidate["original_entity"] = entity_name
                mapped_candidate['retrieved_chunk_id'] = chunk_id
                mapped_candidate["mapping_status"] = "mapped_sym_success"
                mapped_candidates.append(mapped_candidate)
        
        # Process edges in batch
        if edges_to_map:
            # First, find all mapped edges
            mapped_edges = {}
            for src_entity, tgt_entity, description, idx in edges_to_map:
                mapped_result = self.sym_mapper.find_mapped_edge_description(
                    src_entity, tgt_entity, description)
                if mapped_result:
                    mapped_src, mapped_tgt, mapped_desc, _ = mapped_result
                    mapped_edges[idx] = (mapped_src, mapped_tgt, mapped_desc)
            
            # Then, collect all edge pairs that need chunk lookup
            edge_pairs_to_lookup = [(edge_info[0], edge_info[1]) for idx, edge_info in mapped_edges.items()]
            print(edge_pairs_to_lookup)
            # Batch lookup of chunks
            chunk_lookups = {}
            if edge_pairs_to_lookup:
                chunk_results = await self.graph1.a_most_relevant_text_chunks_from_edges(
                    query=query_text, 
                    list_edges=edge_pairs_to_lookup, 
                    threshold=max_threshold
                )
                
                # Map back to the original indices
                for idx, (mapped_src, mapped_tgt, mapped_desc) in mapped_edges.items():
                    edge_key = (mapped_src, mapped_tgt)
                    if edge_key in chunk_results:
                        chunk_lookups[idx] = (mapped_src, mapped_tgt, mapped_desc, chunk_results[edge_key])
            
            # Create mapped candidates
            for idx, lookup_result in chunk_lookups.items():
                mapped_src, mapped_tgt, mapped_desc, chunk_id = lookup_result
                original_candidate = candidates[idx]
                src_entity = original_candidate.get("src_id")
                tgt_entity = original_candidate.get("tgt_id")
                
                # print("mapp sym ne")
                mapped_candidate = original_candidate.copy()
                mapped_candidate["src_id"] = mapped_src
                mapped_candidate["tgt_id"] = mapped_tgt
                mapped_candidate["description"] = mapped_desc
                mapped_candidate["original_src"] = src_entity
                mapped_candidate["original_tgt"] = tgt_entity
                mapped_candidate['retrieved_chunk_id'] = chunk_id
                mapped_candidate["mapping_status"] = "mapped_sym_success"
                mapped_candidates.append(mapped_candidate)
        
        return mapped_candidates, hl_keywords, ll_keywords

    
    def _merge_results(
        self,
        graph1_results: Tuple[List[Dict[str, Any]], List[str], List[str]],
        graph2_results: Tuple[List[Dict[str, Any]], List[str], List[str]],
        mapped_graph1_results: Optional[Tuple[List[Dict[str, Any]], List[str], List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """Merge results from both graphs, sorting by score."""
        candidates1, _, _ = graph1_results
        candidates2, _, _ = graph2_results
        
        
        # Combine candidates
        all_candidates = candidates1 + candidates2
        
        if mapped_graph1_results:
            mapped_candidates1, _, _ = mapped_graph1_results
            all_candidates += mapped_candidates1
        
        # Sort by score in descending order
        sorted_candidates = sorted(
            all_candidates, 
            key=lambda x: x.get("score", 0.0), 
            reverse=True
        )
        
        return sorted_candidates
    
    
    def _save_results(self, result: Dict[str, Any], query_text: str, index: None) -> str:
        """Save results to a JSONL file and return the path."""
        # Create a sanitized filename from the query
        # sanitized_query = "".join(c if c.isalnum() else "_" for c in query_text[:30])
        # timestamp = result["timestamp"].replace(":", "-").replace(".", "-")
        if index is not None:
            filename = f"query_{str(index)}.jsonl"
        else:
            filename = f"query_{query_text[:30].replace(' ', '_')}.jsonl"
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
        batch_output_file: Optional[str] = None,
        sym: bool = False, 
        max_threshold: Optional[float] = None
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
            self.abatch_query(queries, mode, param, save_results, batch_output_file,sym=sym, max_threshold=max_threshold)
        )
    
    async def abatch_query(
        self,
        queries: List[str],
        mode: str = "hybrid",
        param: Optional[QueryParam] = None,
        save_results: bool = True,
        batch_output_file: Optional[str] = None,
        sym: bool = False,
        max_threshold: Optional[float] = None
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
                save_results=save_results,
                sym=sym,
                index=i+1,
                max_threshold=max_threshold
            )
            all_results.append(result)
        
            # Save all results to a single batch file if requested
            if batch_output_file:
                batch_filepath = os.path.join(self.output_dir, batch_output_file)
                with open(batch_filepath, "w", encoding="utf-8") as f:
                    for result in all_results:
                        f.write(json.dumps(result, ensure_ascii=False, indent=4) + "\n")
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
    parser.add_argument("--mapping_sym_file", default=None, help="Path to entity/edge mapping file")
    parser.add_argument("--output_dir", default="query_results", help="Directory to save results")
    parser.add_argument("--graph1_name", default="graph1", help="Name for first graph")
    parser.add_argument("--graph2_name", default="graph2", help="Name for second graph")
    parser.add_argument("--batch_output", help="Filename for combined batch results (for batch mode)")
    parser.add_argument("--embedding_func_name", default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model to use for both graphs")
    parser.add_argument("--embedding_path", default="graph1", help="Name for first graph")
    parser.add_argument('--devices', nargs='+', help='List of CUDA devices')
    parser.add_argument("--threshold_sym", type=float, default=None)
    
    args = parser.parse_args()
    
    # Ensure either query or queries_file is provided
    if not args.query and not args.queries_file:
        parser.error("Either --query or --queries_file must be provided")
    

    graph1 = init_rag(
        working_dir=args.graph1_path,
        embedding_path=args.embedding_path,
        embedding_func_name=args.embedding_func_name, 
        devices=args.devices,
        # embedding_func=embedding_func
    )
    graph2 = init_rag(
        working_dir=args.graph2_path,
        embedding_path=args.embedding_path,
        embedding_func_name=args.embedding_func_name, 
        devices=args.devices,
        # embedding_func=embedding_func
    )
    
    # Create dual query instance
    dual_query = DualGraphQuery(
        graph1_instance=graph1,
        graph2_instance=graph2,
        mapping_file_path=args.mapping_file,
        output_dir=args.output_dir,
        graph1_name=args.graph1_name,
        graph2_name=args.graph2_name,
        mapping_sym_file=args.mapping_sym_file
    )
    
    # Determine if running in single query or batch mode
    if args.query:
        # Single query mode
        results = dual_query.query(
            query_text=args.query,
            mode=args.mode,
            save_results=True,
            sym=args.mapping_sym_file is not None,
            max_threshold=args.threshold_sym
        )
        
        
    else:
        # Batch mode
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
            
        print(f"Processing {len(queries)} queries in batch mode")
        
        all_results = dual_query.batch_query(
            queries=queries,
            mode=args.mode,
            save_results=True,
            batch_output_file=args.batch_output,
            sym=args.mapping_sym_file is not None,
            max_threshold=args.threshold_sym
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
    
    
    

#############################
