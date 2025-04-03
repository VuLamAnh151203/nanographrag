import json
import numpy as np
from typing import Optional, List, Any, Callable

import aioboto3
from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from together import AsyncTogether
import os
from sentence_transformers import SentenceTransformer

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
import asyncio

global_openai_async_client = None
global_azure_openai_async_client = None
global_amazon_bedrock_async_client = None
from aiolimiter import AsyncLimiter

from dotenv import load_dotenv

# Load the .env file
load_dotenv()

rate_limiter = AsyncLimiter(1, time_period=1)   # 1 QPS
minute_limiter = AsyncLimiter(60, time_period=60) 

def get_together_client_instance():
    return AsyncTogether(api_key = os.getenv("TOGETHER_AI"))

def get_openai_async_client_instance(api_key):
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI(api_key)
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_amazon_bedrock_async_client_instance():
    global global_amazon_bedrock_async_client
    if global_amazon_bedrock_async_client is None:
        global_amazon_bedrock_async_client = aioboto3.Session()
    return global_amazon_bedrock_async_client


async def together_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    together_client = get_together_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    
    # Apply rate limiting
    async with rate_limiter, minute_limiter:
        response = await together_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content

async def llama_3_1_8B_Turbo_together( 
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await together_complete_if_cache(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
# async def openai_complete_if_cache(
#     model, prompt, system_prompt=None, history_messages=[],api_key: str | None = None, **kwargs
# ) -> str:
#     openai_async_client = get_openai_async_client_instance(api_key)
#     hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
#     messages = []
#     if system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.extend(history_messages)
#     messages.append({"role": "user", "content": prompt})
#     if hashing_kv is not None:
#         args_hash = compute_args_hash(model, messages)
#         if_cache_return = await hashing_kv.get_by_id(args_hash)
#         if if_cache_return is not None:
#             return if_cache_return["return"]

#     response = await openai_async_client.chat.completions.create(
#         model=model, messages=messages, **kwargs
#     )

#     if hashing_kv is not None:
#         await hashing_kv.upsert(
#             {args_hash: {"return": response.choices[0].message.content, "model": model}}
#         )
#         await hashing_kv.index_done_callback()
#     return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    messages.extend(history_messages)
    messages.append({"role": "user", "content": [{"text": prompt}]})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    inference_config = {
        "temperature": 0,
        "maxTokens": 4096 if "max_tokens" not in kwargs else kwargs["max_tokens"],
    }

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        if system_prompt:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
                system=[{"text": system_prompt}]
            )
        else:
            response = await bedrock_runtime.converse(
                modelId=model, messages=messages, inferenceConfig=inference_config,
            )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response["output"]["message"]["content"][0]["text"], "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response["output"]["message"]["content"][0]["text"]


def create_amazon_bedrock_complete_function(model_id: str) -> Callable:
    """
    Factory function to dynamically create completion functions for Amazon Bedrock

    Args:
        model_id (str): Amazon Bedrock model identifier (e.g., "us.anthropic.claude-3-sonnet-20240229-v1:0")

    Returns:
        Callable: Generated completion function
    """
    async def bedrock_complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List[Any] = [],
        **kwargs
    ) -> str:
        return await amazon_bedrock_complete_if_cache(
            model_id,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    # Set function name for easier debugging
    bedrock_complete.__name__ = f"{model_id}_complete"
    
    return bedrock_complete


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def amazon_bedrock_embedding(texts: list[str]) -> np.ndarray:
    amazon_bedrock_async_client = get_amazon_bedrock_async_client_instance()

    async with amazon_bedrock_async_client.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    ) as bedrock_runtime:
        embeddings = []
        for text in texts:
            body = json.dumps(
                {
                    "inputText": text,
                    "dimensions": 1024,
                }
            )
            response = await bedrock_runtime.invoke_model(
                modelId="amazon.titan-embed-text-v2:0", body=body,
            )
            response_body = await response.get("body").read()
            embeddings.append(json.loads(response_body))
    return np.array([dp["embedding"] for dp in embeddings])


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

from FlagEmbedding import BGEM3FlagModel
bge_m3 = BGEM3FlagModel(os.getenv("EMBEDDING_MODEL_PATH"),  
                       use_fp16=True, device = "cuda:0")
# bge_m3 = SentenceTransformer(model_name_or_path=os.getenv("EMBEDDING_MODEL_PATH"), device="cpu")
@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def bge_m3_embedding(texts: list[str]) -> np.ndarray:
    # bge_m3 = SentenceTransformer(model_name_or_path="/home/vulamanh/Documents/GRAPHRAG_DATN/src/models/bge_m3", device="cpu")
    # return bge_m3.encode(texts, normalize_embeddings=True,show_progress_bar=False)
    return bge_m3.encode(texts, max_length = 8192)["dense_vecs"]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": deployment_name,
                }
            }
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await azure_openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

# with open(os.getenv("API_KEYS_PATH"), 'r', encoding='utf-8') as f:
#     OPENROUTER_API_KEYS = json.load(f)

import time
import random
class APIManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.failed_keys = set()
        self.last_switch_time = {}  # Để theo dõi thời gian giữa các lần chuyển đổi
        
    def get_current_api_key(self):
        return self.api_keys[self.current_key_index]
    
    def switch_to_next_key(self):
        # Đánh dấu key hiện tại là đã thất bại
        self.failed_keys.add(self.current_key_index)
        self.last_switch_time[self.current_key_index] = time.time()
        
        # Tìm key tiếp theo chưa thất bại hoặc đã qua thời gian chờ
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
            raise RuntimeError("All API keys have failed. Please try again later.")
        
        # Chọn một key ngẫu nhiên từ các key có sẵn
        self.current_key_index = random.choice(available_keys)
        return self.get_current_api_key()
    
    def reset_key(self, key_index):
        if key_index in self.failed_keys:
            self.failed_keys.remove(key_index)


# Khởi tạo API Manager
# api_manager = APIManager(OPENROUTER_API_KEYS)

# Giữ nguyên hàm bất đồng bộ này
# async def gemini_2_0(
#     prompt, system_prompt=None, history_messages=[], **kwargs
# ) -> str:
#     max_retries = len(OPENROUTER_API_KEYS)
#     retry_count = 0
    
#     while retry_count < max_retries:
#         try:
#             current_api_key = api_manager.get_current_api_key()
#             # os["OPENAI_API_KEY"] = current_api_key
#             print(f"Using API key: {current_api_key[:5]}...")
            
#             response = await openai_complete_if_cache(
#                 os.getenv("LLM_OPEN_ROUTER_MODEL"),
#                 prompt,
#                 system_prompt=system_prompt,
#                 history_messages=history_messages,
#                 api_key = current_api_key,
#                 base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
#                 **kwargs
#             )
            
#             # Nếu thành công, đánh dấu key này hoạt động tốt
#             api_manager.reset_key(api_manager.current_key_index)
#             return response
            
#         except Exception as e:
#             print(f"Error with API key {api_manager.current_key_index}: {str(e)}")
#             retry_count += 1
            
#             if retry_count < max_retries:
#                 print(f"Switching to next API key...")
#                 api_manager.switch_to_next_key()
#             else:
#                 print("All API keys have failed. Raising error.")
#                 raise e
    
#     raise RuntimeError("All API keys have failed")

async def qwen(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
 
    max_retries = 10
    retry_count = 0
    model = os.getenv("LLM_MODEL", "qwen/qwen-2.5-7b-instruct")
    while retry_count < max_retries:
        try:
            response = await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=os.getenv("LLM_BINDING_API_KEY"), 
                base_url=os.getenv("LLM_BINDING_HOST", "https://openrouter.ai/api/v1"),
                **kwargs
            )
            
            return response
            
        except Exception as e:
            print(f"Lỗi tại lần: {str(retry_count)}/{str(max_retries)}")
            retry_count += 1
            
    
    raise RuntimeError("Tất cả API keys đều đã thất bại")

print(f"Loading model...{os.getenv('LLM_MODEL')}")


async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> str:
    if history_messages is None:
        history_messages = []
    if not api_key:
        api_key = os.environ["OPENAI_API_KEY"]

    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
        "Content-Type": "application/json",
    }

    # # Set openai logger level to INFO when VERBOSE_DEBUG is off
    # if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
    #     logging.getLogger("openai").setLevel(logging.INFO)

    openai_async_client = (
        AsyncOpenAI(default_headers=default_headers, api_key=api_key)
        if base_url is None
        else AsyncOpenAI(
            base_url=base_url, default_headers=default_headers, api_key=api_key
        )
    )
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # logger.debug("===== Sending Query to LLM =====")
    # logger.debug(f"Model: {model}   Base URL: {base_url}")
    # logger.debug(f"Additional kwargs: {kwargs}")
    # verbose_debug(f"Query: {prompt}")
    # verbose_debug(f"System prompt: {system_prompt}")

    try:
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
        raise
    except RateLimitError as e:
        print(f"OpenAI API Rate Limit Error: {e}")
        raise
    # except APITimeoutError as e:
    #     print(f"OpenAI API Timeout Error: {e}")
    #     raise
    except Exception as e:
        print(
            f"OpenAI API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}"
        )
        raise

    # if hasattr(response, "__aiter__"):

    #     async def inner():
    #         try:
    #             async for chunk in response:
    #                 content = chunk.choices[0].delta.content
    #                 if content is None:
    #                     continue
    #                 if r"\u" in content:
    #                     content = safe_unicode_decode(content.encode("utf-8"))
    #                 yield content
    #         except Exception as e:
    #             logger.error(f"Error in stream response: {str(e)}")
    #             raise

    #     return inner()

    # else:
    #     if (
    #         not response
    #         or not response.choices
    #         or not hasattr(response.choices[0], "message")
    #         or not hasattr(response.choices[0].message, "content")
    #     ):
    #         logger.error("Invalid response from OpenAI API")
    #         raise InvalidResponseError("Invalid response from OpenAI API")

    #     content = response.choices[0].message.content

    #     if not content or content.strip() == "":
    #         logger.error("Received empty content from OpenAI API")
    #         raise InvalidResponseError("Received empty content from OpenAI API")

    #     if r"\u" in content:
    #         content = safe_unicode_decode(content.encode("utf-8"))
    #     return content

    content = response.choices[0].message.content

    return content
