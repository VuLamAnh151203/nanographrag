import os
import sys
import time
import json
import random
from typing import List


from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import detect_language
with open("/home/hungpv/projects/list_key_open_router/openrouter_full.json", 'r', encoding='utf-8') as f:
    OPENROUTER_API_KEYS = json.load(f)

class APIManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 100
        self.failed_keys = set()
        self.last_switch_time = {}  
        
    def get_current_api_key(self):
        return self.api_keys[self.current_key_index]
    
    def switch_to_next_key(self):
        
        self.failed_keys.add(self.current_key_index)
        self.last_switch_time[self.current_key_index] = time.time()
        
        available_keys = []
        for idx in range(len(self.api_keys)):
            if idx not in self.failed_keys:
                available_keys.append(idx)
            elif idx in self.last_switch_time:
                # Nếu đã qua 10 phút kể từ lần cuối sử dụng key này
                if time.time() - self.last_switch_time[idx] > 600:
                    self.failed_keys.remove(idx)
                    available_keys.append(idx)
        
        if not available_keys:
            raise RuntimeError("Tất cả API keys đều đã thất bại. Vui lòng thử lại sau.")
        
        self.current_key_index = random.choice(available_keys)
        print(f"Đã chuyển sang API key: {self.api_keys[self.current_key_index][:5]}...")
        return self.get_current_api_key()
    
    def reset_key(self, key_index):
        if key_index in self.failed_keys:
            self.failed_keys.remove(key_index)

# Khởi tạo API Manager
api_manager = APIManager(OPENROUTER_API_KEYS)

WORKING_DIR = "zalo_wiki"

current_api_key = api_manager.get_current_api_key()
    
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    global current_api_key
    max_retries = len(OPENROUTER_API_KEYS)
    retry_count = 0
    
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
            
            key_index = OPENROUTER_API_KEYS.index(current_api_key)
            current_api_key = api_manager.switch_to_next_key()
            os.environ["LLM_BINDING_API_KEY"] = current_api_key
            api_manager.reset_key(key_index)
            return response
            
        except Exception as e:
            print(f"Lỗi với API key: {str(e)}")
            retry_count += 1
            
            if retry_count < max_retries:
                print("Đang chuyển sang API key tiếp theo...")
                current_api_key = api_manager.switch_to_next_key()
                os.environ["LLM_BINDING_API_KEY"] = current_api_key
            else:
                print("Tất cả API keys đều đã thất bại.")
                raise e
    
    raise RuntimeError("Tất cả API keys đều đã thất bại")

print("Loading model...")
print("google/gemini-2.0-flash-exp:free")

os.environ["LLM_BINDING_API_KEY"] = current_api_key

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=5000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(
                "BAAI/bge-m3"
            ),
            embed_model=AutoModel.from_pretrained(
                "BAAI/bge-m3"
            ),
        ),
    ),
    addon_params={
        "insert_batch_size": 10,
        "language": "Vietnamese"
    }
)

def insert_with_retry(data_lists, target_language="English"):
    global current_api_key
    max_retries = len(OPENROUTER_API_KEYS)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
  
            os.environ["LLM_BINDING_API_KEY"] = current_api_key
            rag.insert_duo(data_lists, target_language="English")
            print("Chèn dữ liệu thành công!")
            return
        except Exception as e:
            error_str = str(e).lower()
            if "api" in error_str or "key" in error_str or "rate" in error_str or "limit" in error_str:
                print(f"Lỗi API khi chèn dữ liệu: {str(e)}")
                retry_count += 1
                
                if retry_count < max_retries:
                    print("Đang chuyển sang API key tiếp theo...")
                    current_api_key = api_manager.switch_to_next_key()
                    os.environ["LLM_BINDING_API_KEY"] = current_api_key
                else:
                    print("Tất cả API keys đều đã thất bại khi chèn dữ liệu.")
                    raise e
            else:
                raise e

def main():
    try:
        
        with open("/home/hungpv/projects/TN/LIGHTRAG/zalo_wiki_data/zalo_wiki_2019_duo_doc_gg.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        insert_with_retry(data[:10])
        
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {str(e)}")

if __name__ == "__main__":
    main()
