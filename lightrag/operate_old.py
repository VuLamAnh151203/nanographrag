from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator
from collections import Counter, defaultdict
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    statistic_data,
    get_conversation_turns,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt_old import GRAPH_FIELD_SEP, PROMPTS, get_prompt
import time


def chunking_by_token_size(
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
    tiktoken_model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                new_chunks.append((len(_tokens), chunk))
        else:
            for chunk in raw_chunks:
                _tokens = encode_string_by_tiktoken(chunk, model_name=tiktoken_model)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = decode_tokens_by_tiktoken(
                            _tokens[start : start + max_token_size],
                            model_name=tiktoken_model,
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = decode_tokens_by_tiktoken(
                tokens[start : start + max_token_size], model_name=tiktoken_model
            )
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    """Handle entity relation summary
    For each entity or relation, input is the combined description of already existing description and new description.
    If too long, use LLM to summarize.
    """
    use_llm_func: callable = global_config["llm_model_func"]
    llm_max_tokens = global_config["llm_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        language=language,
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])

    edge_keywords = clean_str(record_attributes[4])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        metadata={"created_at": time.time()},
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """Get existing nodes from knowledge graph use name,if exists, merge data, else create, then upsert."""
    already_entity_types = []
    already_source_ids = []
    already_description = []
    # already_mapping = {}

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entity_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

        # already_mapping = {**already_node["already_mapping"]}

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entity_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    # description = GRAPH_FIELD_SEP.join(
    #     sorted(set([dp["description"] for dp in nodes_data] + already_description))
    # )
    description = GRAPH_FIELD_SEP.join(
        [dp["description"] for dp in nodes_data] + already_description
    )
    # source_id = GRAPH_FIELD_SEP.join(
    #     set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    # )

    source_id = GRAPH_FIELD_SEP.join(
        [dp["source_id"] for dp in nodes_data] + already_source_ids
    )

    # description = await _handle_entity_relation_summary(
    #     entity_name, description, global_config
    # )

    # for dp in nodes_data:
    #     already_mapping[dp["description"]] = dp["source_id"] 

    
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        # Handle the case where get_edge returns None or missing fields
        if already_edge:
            # Get weight with default 0.0 if missing
            already_weights.append(already_edge.get("weight", 0.0))

            # Get source_id with empty string default if missing or None
            if already_edge.get("source_id") is not None:
                already_source_ids.extend(
                    split_string_by_multi_markers(
                        already_edge["source_id"], [GRAPH_FIELD_SEP]
                    )
                )

            # Get description with empty string default if missing or None
            if already_edge.get("description") is not None:
                already_description.append(already_edge["description"])

            # Get keywords with empty string default if missing or None
            if already_edge.get("keywords") is not None:
                already_keywords.extend(
                    split_string_by_multi_markers(
                        already_edge["keywords"], [GRAPH_FIELD_SEP]
                    )
                )

    # Process edges_data with None checks
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    # description = GRAPH_FIELD_SEP.join(
    #     sorted(
    #         set(
    #             [dp["description"] for dp in edges_data if dp.get("description")]
    #             + already_description
    #         )
    #     )
    # )
    description = GRAPH_FIELD_SEP.join(
                [dp["description"] for dp in edges_data if dp.get("description")]
                + already_description
    )
    # keywords = GRAPH_FIELD_SEP.join(
    #     sorted(
    #         set(
    #             [dp["keywords"] for dp in edges_data if dp.get("keywords")]
    #             + already_keywords
    #         )
    #     )
    # )

    keywords = GRAPH_FIELD_SEP.join(
                [dp["keywords"] for dp in edges_data if dp.get("keywords")]
                + already_keywords
    )
    # source_id = GRAPH_FIELD_SEP.join(
    #     set(
    #         [dp["source_id"] for dp in edges_data if dp.get("source_id")]
    #         + already_source_ids
    #     )
    # )

    source_id = GRAPH_FIELD_SEP.join(
            [dp["source_id"] for dp in edges_data if dp.get("source_id")]
            + already_source_ids
    )

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # description = await _handle_entity_relation_summary(
    #     f"({src_id}, {tgt_id})", description, global_config
    # )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
        ),
    )

    edge_data = dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
    )

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict[str, str],
    llm_response_cache: BaseKVStorage | None = None,
) -> BaseGraphStorage | None:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    enable_llm_cache_for_entity_extract: bool = global_config[
        "enable_llm_cache_for_entity_extract"
    ]

    ordered_chunks = list(chunks.items())
    # add language and example number params to prompt
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    # entity_types = global_config["addon_params"].get(
    #     "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    # )
    _type = get_prompt("DEFAULT_ENTITY_TYPES", language)
    entity_types = global_config["addon_params"].get(
        "entity_types", _type
    )
    example_number = global_config["addon_params"].get("example_number", None)
    _examples = get_prompt("entity_extraction_examples", language)

    # if example_number and example_number < len(PROMPTS["entity_extraction_examples"]):
    #     examples = "\n".join(
    #         PROMPTS["entity_extraction_examples"][: int(example_number)]
    #     )
    # else:
    #     examples = "\n".join(PROMPTS["entity_extraction_examples"])

    if example_number and example_number < len(_examples):
        examples = "\n".join(
            _examples[: int(example_number)]
        )
    else:
        examples = "\n".join(_examples)

    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        language=language,
    )
    # add example's format
    examples = examples.format(**example_context_base)

    # entity_extract_prompt = PROMPTS["entity_extraction"]
    entity_extract_prompt = get_prompt("entity_extraction", language)

    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    # continue_prompt = PROMPTS["entiti_continue_extraction"]
    # if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
    continue_prompt = get_prompt("entiti_continue_extraction", language)
    if_loop_prompt = get_prompt("entity_extraction", language)

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _user_llm_func_with_cache(
        input_text: str, history_messages: list[dict[str, str]] = None
    ) -> str:
        if enable_llm_cache_for_entity_extract and llm_response_cache:
            if history_messages:
                history = json.dumps(history_messages, ensure_ascii=False)
                _prompt = history + "\n" + input_text
            else:
                _prompt = input_text

            arg_hash = compute_args_hash(_prompt)
            cached_return, _1, _2, _3 = await handle_cache(
                llm_response_cache,
                arg_hash,
                _prompt,
                "default",
                cache_type="extract",
                force_llm_cache=True,
            )
            if cached_return:
                logger.debug(f"Found cache for {arg_hash}")
                statistic_data["llm_cache"] += 1
                return cached_return
            statistic_data["llm_call"] += 1
            if history_messages:
                res: str = await use_llm_func(
                    input_text, history_messages=history_messages
                )
            else:
                res: str = await use_llm_func(input_text)
            await save_to_cache(
                llm_response_cache,
                CacheData(
                    args_hash=arg_hash,
                    content=res,
                    prompt=_prompt,
                    cache_type="extract",
                ),
            )
            return res

        if history_messages:
            return await use_llm_func(input_text, history_messages=history_messages)
        else:
            return await use_llm_func(input_text)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """ "Prpocess a single chunk
        Args:
            chunk_key_dp (tuple[str, TextChunkSchema]):
                ("chunck-xxxxxx", {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int})
        """
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        # hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        hint_prompt = entity_extract_prompt.format(
            **context_base, input_text="{input_text}"
        ).format(**context_base, input_text=content)

        final_result = await _user_llm_func_with_cache(hint_prompt)
        # final_result = await use_llm_func(hint_prompt)
        
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _user_llm_func_with_cache(
                continue_prompt, history_messages=history
            )
            # glean_result = await _user_llm_func_with_cache(
            #     continue_prompt, history_messages=history
            # )

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await _user_llm_func_with_cache(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.debug(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
        )
        return dict(maybe_nodes), dict(maybe_edges)

    tasks = [_process_single_content(c) for c in ordered_chunks]
    results = await asyncio.gather(*tasks)

    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)

    # print(maybe_nodes)
    # print(maybe_edges)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )

    all_relationships_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )

    if not len(all_entities_data) and not len(all_relationships_data):
        logger.warning(
            "Didn't extract any entities and relationships, maybe your LLM is not working"
        )
        return None

    if not len(all_entities_data):
        logger.warning("Didn't extract any entities")
    if not len(all_relationships_data):
        logger.warning("Didn't extract any relationships")

    if entity_vdb is not None:
    #     data_for_vdb = {}
    #     for dp in all_entities_data:
    #         list_des = list(dp["description"].split("<SEP>"))
    #         for des in list_des:
    #             data_for_vdb[compute_mdhash_id(dp["entity_name"] + des, prefix="ent-")] = {
    #                 "content": dp["entity_name"] + " " + des,
    #                 "entity_name": dp["entity_name"],
    #             }
        # await entities_vdb.upsert(data_for_vdb)
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    if relationships_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                "src_id": dp["src_id"],
                "tgt_id": dp["tgt_id"],
                "content": dp["keywords"]
                + dp["src_id"]
                + dp["tgt_id"]
                + dp["description"],
                "metadata": {
                    "created_at": dp.get("metadata", {}).get("created_at", time.time())
                },
            }
            for dp in all_relationships_data
        }
        await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst

def log_query_result(query_key, keywords, filename='results.jsonl'):
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump({query_key: keywords}, f, ensure_ascii=False)
        f.write('\n')

# def is_query_cached(query_key, filename='results.jsonl'):
#     with open(filename, 'r', encoding='utf-8') as f:
#         for line in f:
#             try:
#                 record = json.loads(line)
#                 if query_key in record:
#                     return record[query_key]
#             except json.JSONDecodeError:
#                 continue  # Skip malformed lines
#     return None

def is_query_cached(query_key, query_dict):
    return query_dict.get(query_key, None)

async def kg_retrieval(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response
    
    # cache_keywords = is_query_cached(query,global_config["cache_queries"])
    cache_keywords = None
    if cache_keywords:
        logger.info(f"Get cache for queries:{query}")
        hl_keywords, ll_keywords = cache_keywords["hl_keywords"], cache_keywords["ll_keywords"]
    else:
        # Extract keywords using extract_keywords_only function which already supports conversation history
        hl_keywords, ll_keywords = await extract_keywords_only(
            query, query_param, global_config, hashing_kv
        )
        # log_query_result(query, {"hl_keywords" : hl_keywords, "ll_keywords" : ll_keywords}, global_config["cache_queries_file"])

    # store it in the json file 




    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    # if hl_keywords == [] and ll_keywords == []:
    #     logger.warning("low_level_keywords and high_level_keywords is empty")
    #     return []
    # if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
    #     logger.warning(
    #         "low_level_keywords is empty, switching from %s mode to global mode",
    #         query_param.mode,
    #     )
    #     query_param.mode = "global"
    # if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
    #     logger.warning(
    #         "high_level_keywords is empty, switching from %s mode to local mode",
    #         query_param.mode,
    #     )
    #     query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    # chunk_list = await _build_retrieval_context(
    #     ll_keywords_str,
    #     hl_keywords_str,
    #     knowledge_graph_inst,
    #     entities_vdb,
    #     relationships_vdb,
    #     text_chunks_db,
    #     query_param,
    # )
    if query_param.ll_keyword_only:
        if ll_keywords_str: 
            chunk_list = await _build_retrieval_context(
                ll_keywords_str,
                ll_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
        else:
            chunk_list = await _build_retrieval_context(
                hl_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )
    else:
        chunk_list = await _build_retrieval_context(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
        )

    return chunk_list

async def _build_retrieval_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):        
    if not ll_keywords:
        ll_keywords = hl_keywords

    if query_param.mode == "local":
        entities_context, relations_context, text_units_context, entity_chunks_mapping = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context, relation_chunks_mapping = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:   # hybrid mode
        if query_param.retrieval_mode == "list_des": 
            ll_data, hl_data = await asyncio.gather(
                _get_node_data_list_des(
                    ll_keywords,
                    knowledge_graph_inst,
                    entities_vdb,
                    text_chunks_db,
                    query_param,
                ),
                _get_edge_data_list_des(
                    hl_keywords,
                    knowledge_graph_inst,
                    relationships_vdb,
                    text_chunks_db,
                    query_param,
                ),
            )
            (
                ll_entities_context,
                ll_relations_context,
                ll_text_units_context,
                ll_text_units_section_list,
                all_text_units_each_entity
            ) = ll_data

            (   use_entities,
                hl_entities_context,
                hl_relations_context,
                hl_text_units_context,
                hl_text_units_section_list,
                text_unit_per_edges
            ) = hl_data

            ll_text_chunks = [(x[1],x[2],x[3],x[4]) for x in ll_text_units_section_list[1:]]
            hl_text_chunks = [(x[1],x[2],x[3],x[4]) for x in hl_text_units_section_list[1:]]
            # ll_text_chunks = [(x[1],x[2],x[3]) for x in ll_text_units_section_list[1:]]
            # hl_text_chunks = [(x[1],x[2],x[3]) for x in hl_text_units_section_list[1:]]
            # ll_text_chunks = [x[1] for x in ll_text_units_section_list[1:]]
            # hl_text_chunks = [x[1] for x in hl_text_units_section_list[1:]]
        elif query_param.retrieval_mode == "original":
            ll_data, hl_data = await asyncio.gather(
                _get_node_data(
                        ll_keywords,
                        knowledge_graph_inst,
                        entities_vdb,
                        text_chunks_db,
                        query_param,
                ),
                _get_edge_data(
                        hl_keywords,
                        knowledge_graph_inst,
                        relationships_vdb,
                        text_chunks_db,
                        query_param,
                    ),
            )
            (
                    ll_entities_context,
                    ll_relations_context,
                    ll_text_units_context,
                    ll_text_units_section_list,
                    all_text_units_each_entity
            ) = ll_data

            (   use_entities,
                    hl_entities_context,
                    hl_relations_context,
                    hl_text_units_context,
                    hl_text_units_section_list,
                    text_unit_per_edges
            ) = hl_data

            ll_text_chunks = [x[1] for x in ll_text_units_section_list[1:]]
            hl_text_chunks = [x[1] for x in hl_text_units_section_list[1:]]


    return ll_text_chunks,hl_text_chunks,all_text_units_each_entity,text_unit_per_edges


async def kg_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Extract keywords using extract_keywords_only function which already supports conversation history
    hl_keywords, ll_keywords = await extract_keywords_only(
        query, query_param, global_config, hashing_kv
    )

    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # Handle empty keywords
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # Build context
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response


async def extract_keywords_only(
    text: str,
    param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> tuple[list[str], list[str]]:
    """
    Extract high-level and low-level keywords from the given 'text' using the LLM.
    This method does NOT build the final RAG context or provide a final answer.
    It ONLY extracts keywords (hl_keywords, ll_keywords).
    """ 

    # 1. Handle cache if needed - add cache type for keywords


    args_hash = compute_args_hash(param.mode, text, cache_type="keywords")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, text, param.mode, cache_type="keywords"
    )

    
    if cached_response is not None:
        try:
            keywords_data = json.loads(cached_response)
            return keywords_data["high_level_keywords"], keywords_data[
                "low_level_keywords"
            ]
        except (json.JSONDecodeError, KeyError):
            logger.warning(
                "Invalid cache format for keywords, proceeding with extraction"
            )

    # 2. Build the examples
    example_number = global_config["addon_params"].get("example_number", None)
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    _examples = get_prompt("keywords_extraction_examples", language)

    if example_number and example_number < len(_examples):
        examples = "\n".join(
            _examples[: int(example_number)]
        )
    else:
        examples = "\n".join(_examples)


    # 3. Process conversation history
    history_context = ""
    if param.conversation_history:
        history_context = get_conversation_turns(
            param.conversation_history, param.history_turns
        )

    # 4. Build the keyword-extraction prompt
    kw_prompt = get_prompt("keywords_extraction", language).format(
        query=text, examples=examples, language=language, history=history_context
    )


    len_of_prompts = len(encode_string_by_tiktoken(kw_prompt))
    logger.debug(f"[kg_query]Prompt Tokens: {len_of_prompts}")

    # 5. Call the LLM for keyword extraction
    use_model_func = global_config["llm_model_func"]
    result = await use_model_func(kw_prompt, keyword_extraction=True)

    # 6. Parse out JSON from the LLM response
    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        logger.error("No JSON-like structure found in the LLM respond.")
        return [], []
    try:
        keywords_data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return [], []

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    # 7. Cache only the processed keywords with cache type
    if hl_keywords or ll_keywords:
        cache_data = {
            "high_level_keywords": hl_keywords,
            "low_level_keywords": ll_keywords,
        }
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=json.dumps(cache_data),
                prompt=text,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=param.mode,
                cache_type="keywords",
            ),
        )
    return hl_keywords, ll_keywords


async def mix_kg_vector_query(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    """
    Hybrid retrieval implementation combining knowledge graph and vector search.

    This function performs a hybrid search by:
    1. Extracting semantic information from knowledge graph
    2. Retrieving relevant text chunks through vector similarity
    3. Combining both results for comprehensive answer generation
    """
    # 1. Cache handling
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash("mix", query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, "mix", cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 2. Execute knowledge graph and vector searches in parallel
    async def get_kg_context():
        try:
            # Extract keywords using extract_keywords_only function which already supports conversation history
            hl_keywords, ll_keywords = await extract_keywords_only(
                query, query_param, global_config, hashing_kv
            )

            if not hl_keywords and not ll_keywords:
                logger.warning("Both high-level and low-level keywords are empty")
                return None

            # Convert keyword lists to strings
            ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
            hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

            # Set query mode based on available keywords
            if not ll_keywords_str and not hl_keywords_str:
                return None
            elif not ll_keywords_str:
                query_param.mode = "global"
            elif not hl_keywords_str:
                query_param.mode = "local"
            else:
                query_param.mode = "hybrid"

            # Build knowledge graph context
            context = await _build_query_context(
                ll_keywords_str,
                hl_keywords_str,
                knowledge_graph_inst,
                entities_vdb,
                relationships_vdb,
                text_chunks_db,
                query_param,
            )

            return context

        except Exception as e:
            logger.error(f"Error in get_kg_context: {str(e)}")
            return None

    async def get_vector_context():
        # Consider conversation history in vector search
        augmented_query = query
        if history_context:
            augmented_query = f"{history_context}\n{query}"

        try:
            # Reduce top_k for vector search in hybrid mode since we have structured information from KG
            mix_topk = min(10, query_param.top_k)
            results = await chunks_vdb.query(augmented_query, top_k=mix_topk)
            if not results:
                return None

            chunks_ids = [r["id"] for r in results]
            chunks = await text_chunks_db.get_by_ids(chunks_ids)

            valid_chunks = []
            for chunk, result in zip(chunks, results):
                if chunk is not None and "content" in chunk:
                    # Merge chunk content and time metadata
                    chunk_with_time = {
                        "content": chunk["content"],
                        "created_at": result.get("created_at", None),
                    }
                    valid_chunks.append(chunk_with_time)

            if not valid_chunks:
                return None

            maybe_trun_chunks = truncate_list_by_token_size(
                valid_chunks,
                key=lambda x: x["content"],
                max_token_size=query_param.max_token_for_text_unit,
            )

            if not maybe_trun_chunks:
                return None

            # Include time information in content
            formatted_chunks = []
            for c in maybe_trun_chunks:
                chunk_text = c["content"]
                if c["created_at"]:
                    chunk_text = f"[Created at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['created_at']))}]\n{chunk_text}"
                formatted_chunks.append(chunk_text)

            logger.debug(
                f"Truncate chunks from {len(chunks)} to {len(formatted_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
            )
            return "\n--New Chunk--\n".join(formatted_chunks)
        except Exception as e:
            logger.error(f"Error in get_vector_context: {e}")
            return None

    # 3. Execute both retrievals in parallel
    kg_context, vector_context = await asyncio.gather(
        get_kg_context(), get_vector_context()
    )

    # 4. Merge contexts
    if kg_context is None and vector_context is None:
        return PROMPTS["fail_response"]

    if query_param.only_need_context:
        return {"kg_context": kg_context, "vector_context": vector_context}

    # 5. Construct hybrid prompt
    sys_prompt = (
        system_prompt
        if system_prompt
        else PROMPTS["mix_rag_response"].format(
            kg_context=kg_context
            if kg_context
            else "No relevant knowledge graph information found",
            vector_context=vector_context
            if vector_context
            else "No relevant text information found",
            response_type=query_param.response_type,
            history=history_context,
        )
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[mix_kg_vector_query]Prompt Tokens: {len_of_prompts}")

    # 6. Generate response
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # 清理响应内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

        # 7. Save cache - 只有在收集完整响应后才缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode="mix",
                cache_type="query",
            ),
        )

    return response


async def _build_query_context(
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    if query_param.mode == "local":
        entities_context, relations_context, text_units_context = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    elif query_param.mode == "global":
        entities_context, relations_context, text_units_context = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )
    else:  # hybrid mode
        ll_data, hl_data = await asyncio.gather(
            _get_node_data(
                ll_keywords,
                knowledge_graph_inst,
                entities_vdb,
                text_chunks_db,
                query_param,
            ),
            _get_edge_data(
                hl_keywords,
                knowledge_graph_inst,
                relationships_vdb,
                text_chunks_db,
                query_param,
            ),
        )

        (
            ll_entities_context,
            ll_relations_context,
            ll_text_units_context,
        ) = ll_data

        (
            hl_entities_context,
            hl_relations_context,
            hl_text_units_context,
        ) = hl_data

        entities_context, relations_context, text_units_context = combine_contexts(
            [hl_entities_context, ll_entities_context],
            [hl_relations_context, ll_relations_context],
            [hl_text_units_context, ll_text_units_context],
        )
    # not necessary to use LLM to generate a response
    if not entities_context.strip() and not relations_context.strip():
        return None

    result = f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """.strip()
    return result


async def _get_node_data(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""

    # result_deduplicate = []
    # node_name = []
    # for result in results:
    #     if result["entity_name"] not in node_name:
    #         result_deduplicate.append(result)
    #         node_name.append(result["entity_name"])

    # results = result_deduplicate
    node_name = [r["entity_name"] for r in results]
    for n in node_name[:20]:
        logger.info(f"Extract nodes name:{n}")

    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )
    

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk

    # node_name = [x["entity_name"] for x in node_datas]
    # for n in node_name:
    #     logger.info(f"Extract nodes name:{n}")
    (use_text_units,all_text_units_each_entity), use_relations = await asyncio.gather(
        _find_most_related_text_unit_from_entities(
            node_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
        _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        ),
    )

    len_node_datas = len(node_datas)
    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )

    # import json 
    # with open("/home/hungpv/projects/TN/data/data_zalo_QA/filtered_corpus.json", "r") as f:
    #     corpus = json.load(f)

    # reverse_corpus = {v:k for k,v in corpus.items()}
    # chunk_id = [reverse_corpus[chunk] for chunk in use_text_units]
    # logger.info(f"Extract nodes:{chunk_id[:15]}")

    # build prompt
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context,text_units_section_list,all_text_units_each_entity

async def _get_node_data_list_des(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    # get similar entities
    logger.info(
        f"Query nodes: {query}, top_k: {query_param.top_k}, cosine: {entities_vdb.cosine_better_than_threshold}"
    )
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return "", "", ""
    
    if query_param.unique_entity_edge:
        print("Take uniques entities")
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        node_name = []
        for result in results:
            if result["entity_name"] in node_name:
                continue
            if result["chunk_id"] not in node_name:
                chunk_id_list.append(result["chunk_id"])
                node_name.append(result["entity_name"])
                result_chosen.append(result)
            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break

    else:
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        for result in results:
            if result["chunk_id"] not in chunk_id_list:
                chunk_id_list.append(result["chunk_id"])
                result_chosen.append(result)
            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break

    # results = result_deduplicate
    
    
    # # take the chunk id from the results
    distances = [float(r["distance"]) for r in result_chosen]
    chunk_ids = [dp["chunk_id"] for dp in result_chosen]
    description = [dp["description"] for dp in result_chosen]
    entity_name = [dp["entity_name"] for dp in result_chosen]
    chunk_datas = await text_chunks_db.get_by_ids(chunk_ids)
    use_text_units = [(chunk_data, distance,des,e) for chunk_data, distance, des,e in zip(chunk_datas, distances, description,entity_name)]

    # deduplicate the entity
    result_deduplicate = []
    node_name = []
    for result in results:
        if result["entity_name"] not in node_name:
            result_deduplicate.append(result)
            node_name.append(result["entity_name"])

    results = result_deduplicate

    for n in node_name[:20]:
        logger.info(f"Extract nodes name:{n}")

    # get entity information
    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
        ),
        asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
        ),
    )
    

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodprocessedes are missing, maybe the storage is damaged")

    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    # get entitytext chunk

    # (_,all_text_units_each_entity), use_relations = await asyncio.gather(
    #         _find_most_related_text_unit_from_entities(
    #             node_datas, query_param, text_chunks_db, knowledge_graph_inst
    #         ),
    #         _find_most_related_edges_from_entities(
    #             node_datas, query_param, knowledge_graph_inst
    #         ),
    #     )

    if query_param.retrieval_nodes:
        (_,all_text_units_each_entity), use_relations = await asyncio.gather(
            _find_most_related_text_unit_from_entities(
                node_datas, query_param, text_chunks_db, knowledge_graph_inst
            ),
            _find_most_related_edges_from_entities(
                node_datas, query_param, knowledge_graph_inst
            ),
        )
    else:
        all_text_units_each_entity = {}
        use_relations = await _find_most_related_edges_from_entities(
                node_datas, query_param, knowledge_graph_inst
            )

    len_node_datas = len(node_datas)
    # node_datas = truncate_list_by_token_size(
    #     node_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_local_context,
    # )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} chunks"
    )


    # build prompt
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(use_relations):
        created_at = e.get("created_at", "UNKNOWN")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content", "score"]]
    for i, (t,s,d,e) in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"], s,d,e])
    text_units_context = list_of_list_to_csv([[x[0], x[1]] for x in text_units_section_list])

    # text_units_section_list = [["id", "content", "score"]]
    # for i, (t,s,e) in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"], s,e])
    # text_units_context = list_of_list_to_csv([[x[0], x[1]] for x in text_units_section_list])

    # text_units_section_list = [["id", "content"]]
    # for i, t in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"]])
    # text_units_context = list_of_list_to_csv(text_units_section_list)
    return entities_context, relations_context, text_units_context,text_units_section_list,all_text_units_each_entity

async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]

    node_name = [x["entity_name"] for x in node_datas]
    # print(f"Node order 1: {node_name[:10]}")
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    tasks = []
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                tasks.append((c_id, index, this_edges))

    results = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id, _, _ in tasks]
    )

    for (c_id, index, this_edges), data in zip(tasks, results):
        all_text_units_lookup[c_id] = {
            "data": data,
            "order": index,
            "relation_counts": 0,
        }

        if this_edges:
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    # all_text_units = truncate_list_by_token_size(
    #     all_text_units,
    #     key=lambda x: x["data"]["content"],
    #     max_token_size=query_param.max_token_for_text_unit,
    # )

    logger.info(
        f"Truncate chunks from {len(all_text_units_lookup)} to {len(all_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units_each_entity = {}
    # list_entity = []
    # for text_unit in all_text_units:
    #     entity_name = node_datas[text_unit["order"]]["entity_name"]
    #     if entity_name not in list_entity:
    #         list_entity.append(entity_name)
    #     all_text_units_each_entity[entity_name] = all_text_units_each_entity.get(entity_name, []) + [text_unit["data"]["content"]]
    
    # print(f"Node oder 2: {list_entity[:10]}")

    for i in range(len(node_name)):
        results_text_chunks = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id in text_units[i]]
    )
        all_text_units_each_entity[node_name[i]] = all_text_units_each_entity.get(node_name[i], []) + [text_chunk["content"] for text_chunk in results_text_chunks]

    # for n in node_name[:20]:
    #     logger.info(f"Extract nodes name:{n} - # Chunks: {len(all_text_units_each_entity[n])}")

    total_chunks = 0
    for k , v in all_text_units_each_entity.items(): 
        total_chunks += len(v)

    print(total_chunks) 

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units, all_text_units_each_entity


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack, all_edges_degree = await asyncio.gather(
        asyncio.gather(*[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]),
        asyncio.gather(
            *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
        ),
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    logger.debug(
        f"Truncate relations from {len(all_edges)} to {len(all_edges_data)} (max tokens:{query_param.max_token_for_global_context})"
    )

    return all_edges_data


async def _get_edge_data(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]

    for edge_data in edge_datas[:15]:
        logger.info(f"<{edge_data['src_id']},{edge_data['tgt_id']}>")


    # edge_datas = sorted(
    #     edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    len_edge_datas = len(edge_datas)
    # edge_datas = truncate_list_by_token_size(
    #     edge_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_global_context,
    # )
    logger.debug(
        f"Truncate relations from {len_edge_datas} to {len(edge_datas)} (max tokens:{query_param.max_token_for_global_context})"
    )

    use_entities, (use_text_units,text_unit_per_edges) = await asyncio.gather(
        _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        ),
        _find_related_text_unit_from_relationships(
            edge_datas, query_param, text_chunks_db, knowledge_graph_inst
        ),
    )

    
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    # import json 
    # with open("/home/hungpv/projects/TN/data/data_zalo_QA/filtered_corpus.json", "r") as f:
    #     corpus = json.load(f)

    # reverse_corpus = {v:k for k,v in corpus.items()}
    # chunk_id = [reverse_corpus[chunk["cotent"]] for chunk in use_text_units]
    # logger.info(f"Extract nodes:{chunk_id[:20]}")

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return use_entities, entities_context, relations_context, text_units_context,text_units_section_list,text_unit_per_edges

async def _get_edge_data_list_des(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
):
    logger.info(
        f"Query edges: {keywords}, top_k: {query_param.top_k}, cosine: {relationships_vdb.cosine_better_than_threshold}"
    )
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    # take the chunk id from the results

    if query_param.unique_entity_edge:
    #################################### unique edge ##################################
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        edge_name = []
        for result in results:
            if (result["src_id"], result["tgt_id"]) in edge_name:
                continue
            if result["chunk_id"] not in chunk_id_list:
                result_chosen.append(result)
                edge_name.append((result["src_id"], result["tgt_id"]))
                chunk_id_list.append(result["chunk_id"])

            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break

    else:
        #################################### non-unique edge ##################################
        top_k_candidates = 100
        result_chosen = []
        chunk_id_list = []
        for result in results:
            if result["chunk_id"] not in chunk_id_list:
                result_chosen.append(result)
                chunk_id_list.append(result["chunk_id"])

            if len(chunk_id_list) >= top_k_candidates:
                print("Take top 100 uniques chunks")
                break


    # take the chunk id from the results
    distances = [float(r["distance"]) for r in result_chosen]
    chunk_ids = [dp["chunk_id"] for dp in result_chosen]
    description = [dp["description"] for dp in result_chosen]
    edge_name_text = [" <|> ".join([dp["src_id"], dp["tgt_id"]]) for dp in result_chosen] 
    chunk_datas = await text_chunks_db.get_by_ids(chunk_ids)
    use_text_units = [(chunk_data, distance, des,e) for chunk_data, distance,des,e in zip(chunk_datas, distances, description,edge_name_text)]

    
    # Filter những edges bị trùng
    result_deduplicate = []
    edge_name = []
    for result in results:
        if (result["src_id"], result["tgt_id"]) not in edge_name:
            result_deduplicate.append(result)
            edge_name.append((result["src_id"], result["tgt_id"]))

    results = result_deduplicate


    if not len(results):
        return "", "", ""

    edge_datas, edge_degree = await asyncio.gather(
        asyncio.gather(
            *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"])
                for r in results
            ]
        ),
    )





    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")

    edge_datas = [
        {
            "src_id": k["src_id"],
            "tgt_id": k["tgt_id"],
            "rank": d,
            "created_at": k.get("__created_at__", None),  # 从 KV 存储中获取时间元数据
            **v,
        }
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]

    # for r in results[:15]:
    #     logger.info(f"<{r['src_id']},{r['tgt_id']}>")

    for src, tgt in edge_name[:20]:
        logger.info(f"<{src},{tgt}>")


    # edge_datas = sorted(
    #     edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    # )
    len_edge_datas = len(edge_datas)
    # edge_datas = truncate_list_by_token_size(
    #     edge_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_global_context,
    # )
    logger.debug(
        f"Truncate relations from {len_edge_datas} to {len(edge_datas)} (max tokens:{query_param.max_token_for_global_context})"
    )

    if query_param.retrieval_nodes:
        use_entities, (_,text_unit_per_edges) = await asyncio.gather(
            _find_most_related_entities_from_relationships(
                edge_datas, query_param, knowledge_graph_inst
            ),
            _find_related_text_unit_from_relationships(
                edge_datas, query_param, text_chunks_db, knowledge_graph_inst
            ),
        )
    else:
        text_unit_per_edges = {}
        use_entities = await _find_most_related_entities_from_relationships(
                edge_datas, query_param, knowledge_graph_inst
            )

    # use_entities, (use_text_units,text_unit_per_edges) = await asyncio.gather(
    #     _find_most_related_entities_from_relationships(
    #         edge_datas, query_param, knowledge_graph_inst
    #     ),
    #     _find_related_text_unit_from_relationships(
    #         edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    #     ),
    # )

    
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} chunks"
    )

    relations_section_list = [
        [
            "id",
            "source",
            "target",
            "description",
            "keywords",
            "weight",
            "rank",
            "created_at",
        ]
    ]
    for i, e in enumerate(edge_datas):
        created_at = e.get("created_at", "Unknown")
        # Convert timestamp to readable format
        if isinstance(created_at, (int, float)):
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
                created_at,
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content", "score"]]
    for i, (t,s,d,e) in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"],s,d,e])

    # text_units_section_list = [["id", "content", "score"]]
    # for i, (t,s,e) in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"],s,e])
    # text_units_section_list = [["id", "content"]]
    # for i, t in enumerate(use_text_units):
    #     text_units_section_list.append([i, t["content"]])
 # text_units_section_list = [["id", "content", "score"]]
    # for i, (t,s,e) in enumerate(use_text_units):
    #     text_units_sectio
    text_units_context = list_of_list_to_csv([[x[0], x[1]] for x in text_units_section_list])
    return use_entities, entities_context, relations_context, text_units_context,text_units_section_list,text_unit_per_edges

async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()
    # point = len(edge_datas) + 1

    for e in edge_datas:
        if e["src_id"] not in seen:
            # entity_names[e["src_id"]] = point
            seen.add(e["src_id"])
            entity_names.append(e["src_id"])
        # else:
            # entity_names[e["src_id"]] += point
        if e["tgt_id"] not in seen:
            # entity_names[e["tgt_id"]] = point
            seen.add(e["tgt_id"])
            entity_names.append(e["tgt_id"])
        # else:
            # entity_names[e["tgt_id"]] += point
        
        # point -= 1

    # entity_names = list(entity_names.items())
    # entity_names = sorted(entity_names, key = lambda x : x[1], reverse = True)

    # entity_names = [x[0] for x in entity_names]

    node_datas, node_degrees = await asyncio.gather(
        asyncio.gather(
            *[
                knowledge_graph_inst.get_node(entity_name)
                for entity_name in entity_names
            ]
        ),
        asyncio.gather(
            *[
                knowledge_graph_inst.node_degree(entity_name)
                for entity_name in entity_names
            ]
        ),
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    len_node_datas = len(node_datas)
    # node_datas = truncate_list_by_token_size(
    #     node_datas,
    #     key=lambda x: x["description"],
    #     max_token_size=query_param.max_token_for_local_context,
    # )
    logger.debug(
        f"Truncate entities from {len_node_datas} to {len(node_datas)} (max tokens:{query_param.max_token_for_local_context})"
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage,
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    edge_name = [edge_data["src_id"] + " <-> " + edge_data["tgt_id"] for edge_data in edge_datas]

    all_text_units_lookup = {}

    async def fetch_chunk_data(c_id, index):
        if c_id not in all_text_units_lookup:
            chunk_data = await text_chunks_db.get_by_id(c_id)
            # Only store valid data
            if chunk_data is not None and "content" in chunk_data:
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                }

    tasks = []
    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            tasks.append(fetch_chunk_data(c_id, index))

    await asyncio.gather(*tasks)

    if not all_text_units_lookup:
        logger.warning("No valid text chunks found")
        return []

    all_text_units = [{"id": k, **v} for k, v in all_text_units_lookup.items()]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])

    # Ensure all text chunks have content
    valid_text_units = [
        t for t in all_text_units if t["data"] is not None and "content" in t["data"]
    ]

    if not valid_text_units:
        logger.warning("No valid text chunks after filtering")
        return []

    # truncated_text_units = truncate_list_by_token_size(
    #     valid_text_units,
    #     key=lambda x: x["data"]["content"],
    #     max_token_size=query_param.max_token_for_text_unit,
    # )
    truncated_text_units = valid_text_units

    text_unit_per_edges = {}
    # for t in truncated_text_units:
    #     edge_data = edge_datas[t["order"]]
    #     edge_name = edge_data["src_id"] + " <-> " + edge_data["tgt_id"]
    #     text_unit_per_edges[edge_name] = text_unit_per_edges.get(edge_name, []) + [t["data"]["content"]]

    for i in range(len(edge_name)):
        results_text_chunks = await asyncio.gather(
        *[text_chunks_db.get_by_id(c_id) for c_id in text_units[i]]
    )
        text_unit_per_edges[edge_name[i]] = text_unit_per_edges.get(edge_name[i], []) + [text_chunk["content"] for text_chunk in results_text_chunks]


    logger.debug(
        f"Truncate chunks from {len(valid_text_units)} to {len(truncated_text_units)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    all_text_units: list[TextChunkSchema] = [t["data"] for t in truncated_text_units]

    return all_text_units,text_unit_per_edges


def combine_contexts(entities, relationships, sources):
    # Function to extract entities, relationships, and sources from context strings
    hl_entities, ll_entities = entities[0], entities[1]
    hl_relationships, ll_relationships = relationships[0], relationships[1]
    hl_sources, ll_sources = sources[0], sources[1]
    # Combine and deduplicate the entities
    combined_entities = process_combine_contexts(hl_entities, ll_entities)

    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )

    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)

    return combined_entities, combined_relationships, combined_sources

async def naive_retrieval(query: str,
                          query_param: QueryParam,
                          global_config: dict[str, str],
                          chunks_vdb: BaseVectorStorage,
                          text_chunks_db: BaseKVStorage):

    # language = global_config["addon_params"].get(
    #     "language", PROMPTS["DEFAULT_LANGUAGE"]
    # )
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # if not len(results):
    #     return get_prompt("fail_response", language)

    chunks_ids = [r["id"] for r in results]
    chunks_distance = [float(r["distance"]) for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    valid_chunks = [chunk["content"] for chunk in valid_chunks]

    # if not valid_chunks:
    #     logger.warning("No valid chunks found after filtering")
    #     return get_prompt("fail_response", language)
    chunk_scores_dict = {}
    for i in range(len(valid_chunks)):
        chunk_scores_dict[valid_chunks[i]] = chunks_distance[i]

    return chunk_scores_dict

async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    # Handle cache
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]

    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # Filter out invalid chunks
    valid_chunks = [
        chunk for chunk in chunks if chunk is not None and "content" in chunk
    ]

    if not valid_chunks:
        logger.warning("No valid chunks found after filtering")
        return PROMPTS["fail_response"]

    maybe_trun_chunks = truncate_list_by_token_size(
        valid_chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    if not maybe_trun_chunks:
        logger.warning("No chunks left after truncation")
        return PROMPTS["fail_response"]

    logger.debug(
        f"Truncate chunks from {len(chunks)} to {len(maybe_trun_chunks)} (max tokens:{query_param.max_token_for_text_unit})"
    )

    section = "\n--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    if query_param.only_need_context:
        return section

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[naive_query]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt) :]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )

    return response


async def kg_query_with_keywords(
    query: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
) -> str | AsyncIterator[str]:
    """
    Refactored kg_query that does NOT extract keywords by itself.
    It expects hl_keywords and ll_keywords to be set in query_param, or defaults to empty.
    Then it uses those to build context and produce a final LLM response.
    """

    # ---------------------------
    # 1) Handle potential cache for query results
    # ---------------------------
    use_model_func = global_config["llm_model_func"]
    args_hash = compute_args_hash(query_param.mode, query, cache_type="query")
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # ---------------------------
    # 2) RETRIEVE KEYWORDS FROM query_param
    # ---------------------------

    # If these fields don't exist, default to empty lists/strings.
    hl_keywords = getattr(query_param, "hl_keywords", []) or []
    ll_keywords = getattr(query_param, "ll_keywords", []) or []

    # If neither has any keywords, you could handle that logic here.
    if not hl_keywords and not ll_keywords:
        logger.warning(
            "No keywords found in query_param. Could default to global mode or fail."
        )
        return PROMPTS["fail_response"]
    if not ll_keywords and query_param.mode in ["local", "hybrid"]:
        logger.warning("low_level_keywords is empty, switching to global mode.")
        query_param.mode = "global"
    if not hl_keywords and query_param.mode in ["global", "hybrid"]:
        logger.warning("high_level_keywords is empty, switching to local mode.")
        query_param.mode = "local"

    # Flatten low-level and high-level keywords if needed
    ll_keywords_flat = (
        [item for sublist in ll_keywords for item in sublist]
        if any(isinstance(i, list) for i in ll_keywords)
        else ll_keywords
    )
    hl_keywords_flat = (
        [item for sublist in hl_keywords for item in sublist]
        if any(isinstance(i, list) for i in hl_keywords)
        else hl_keywords
    )

    # Join the flattened lists
    ll_keywords_str = ", ".join(ll_keywords_flat) if ll_keywords_flat else ""
    hl_keywords_str = ", ".join(hl_keywords_flat) if hl_keywords_flat else ""

    # ---------------------------
    # 3) BUILD CONTEXT
    # ---------------------------
    context = await _build_query_context(
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
    )
    if not context:
        return PROMPTS["fail_response"]

    # If only context is needed, return it
    if query_param.only_need_context:
        return context

    # ---------------------------
    # 4) BUILD THE SYSTEM PROMPT + CALL LLM
    # ---------------------------

    # Process conversation history
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
    )

    if query_param.only_need_prompt:
        return sys_prompt

    len_of_prompts = len(encode_string_by_tiktoken(query + sys_prompt))
    logger.debug(f"[kg_query_with_keywords]Prompt Tokens: {len_of_prompts}")

    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # Save to cache
    await save_to_cache(
        hashing_kv,
        CacheData(
            args_hash=args_hash,
            content=response,
            prompt=query,
            quantized=quantized,
            min_val=min_val,
            max_val=max_val,
            mode=query_param.mode,
            cache_type="query",
        ),
    )
    return response
