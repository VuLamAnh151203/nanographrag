# import nest_asyncio
from nano_graphrag import GraphRAG, QueryParam
import numpy as np
from nano_graphrag._utils import wrap_embedding_func_with_attrs
import os
import pandas as pd
import pyarrow.parquet as pa
import argparse
from tqdm import tqdm
import json 
import shutil

from dotenv import load_dotenv

from evaluate import average_four_metrics

# Load the .env file
load_dotenv()


# def get_text_chunks(df: pd.DataFrame, n_queries = 20):
#     # Lấy hết câu hỏi khác nhau
#     all_distinct_queries = df["query"].unique()

#     # Obtain the small subset of questions (20 first questions)
#     small_queries_list = all_distinct_queries[:n_queries]

#     # Take the all chunks related to that subset of questions
#     df_small = df[df["query"].isin(small_queries_list)]
#     list_contexts = df_small["pos"].to_list()
#     list_contexts = list(set([a[0] for a in list_contexts]))

#     #Build the dictionary with key is the query and value is list of indexes of relevant chunks
#     true_pos = {}
#     map_contexts_to_id = {contexts: i for i, contexts in enumerate(list_contexts)}
#     for query in tqdm(small_queries_list):
#         pos_list = []
#         df_query_specific = df[df["query"] == query]
#         df_query_specific.reset_index(inplace = True)
#         for i in range(len(df_query_specific)):
#             pos_list.append(map_contexts_to_id[df_query_specific["pos"][i][0]])
#         true_pos[query] = list(set(pos_list))

#     return list_contexts, true_pos

def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def init_graph(WORKING_DIR:str, enable_log = True):
    # nest_asyncio.apply()
    rag = GraphRAG(
            working_dir=WORKING_DIR,
            enable_naive_rag=True,
            enable_local= True,
            enable_log = enable_log
    )

    return rag

def build_graph(WORKING_DIR: str, DATA_DIR: str):
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)  # Remove existing folder
    os.makedirs(WORKING_DIR, exist_ok=True)
    remove_if_exist(f"{WORKING_DIR}/milvus_lite.db")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")
    # access data
    # table  = pa.read_table("/home/vulamanh/Documents/GRAPHRAG_DATN/src/data/synthetic/cross_queries.parquet")
    # df = table.to_pandas()

    # # get text chunks
    # list_context, true_pos = get_text_chunks(df, n_queries=int(os.getenv("N_QUERIES")))

    # with open(os.path.join(WORKING_DIR, "true_pos.json"), "w") as f:
    #     json.dump(true_pos, f)

    with open(DATA_DIR, 'r', encoding='utf-8') as f:
            data = json.load(f)
    data = [item[0] for item in data]

    # init graph
    graghrag = init_graph(WORKING_DIR)

    # insert data:
    graghrag.insert(data[:30])

def query(root:str, method: str, query: str):
    graphrag = init_graph(root,enable_log = False)

    query_param = QueryParam(mode = method, top_k = int(os.getenv("TOP_K")))
    return graphrag.query(query, param = query_param)

def evaluate(rag, true_pos, root:str, method:str, evaluate_k: int):
    # rag = init_graph(root, enable_log=False)
    # # load true_pos:
    # with open(os.path.join(root, "true_pos.json"), "r") as f:
    #     true_pos = json.load(f)
    directory_prediction = os.path.join(root, f"{method}_query")
    if os.path.exists(directory_prediction):
        shutil.rmtree(directory_prediction)
    os.makedirs(directory_prediction, exist_ok=True)
    list_queries = list(true_pos.keys())
    for query in tqdm(list_queries):
        result = rag.query(query, param=QueryParam(mode=method, top_k = int(os.getenv("TOP_K"))))
    
    # load prediction
    path_prediction = os.path.join(directory_prediction, "chosen_text_chunks.json")
    with open(path_prediction, "r") as f:
        prediction = json.load(f)

    # load the text chunks:
    text_chunk_path = os.path.join(root, "kv_store_text_chunks.json")
    with open(text_chunk_path, "r") as f:
        text_chunk = json.load(f)

    list_context = [v["content"] for k, v in text_chunk.items()]

    prediction_id = {}
    for k, v in prediction.items():
        ids = []
        for text_chunk in v:
            ids.append(list_context.index(text_chunk["content"]))
        
        prediction_id[k] = ids
    
    # evaluate
    list_actual = list(true_pos.values())
    list_predicted = list(prediction_id.values())
    return average_four_metrics(list_actual, list_predicted, k=int(evaluate_k))

def evaluate_all(root: str,evaluate_k):
    rag = init_graph(root, enable_log=False)
    # load true_pos:
    with open(os.path.join(root, "true_pos.json"), "r") as f:
        true_pos = json.load(f)
    final_results = ""
    for method in ["local", "naive"]:
        final_results += method.capitalize() + ":\n"
        final_results += evaluate(rag, true_pos, root, method, evaluate_k)

    return final_results[:-1]

def main():
    parser = argparse.ArgumentParser(description="GraphRAG CLI")
    parser.add_argument("command", choices=["query", "index", "evaluate"], help="Command to run")
    parser.add_argument("--root", required=True, help="Root directory")
    parser.add_argument("--data_dir")
    parser.add_argument("--method", help="Processing method")
    parser.add_argument("--query", help="Query text")
    parser.add_argument("--evaluate_k", help = "Top k considered chunks")

    args = parser.parse_args()

    if args.command == "index":
        print("CREATE THE GRAPHRAG...")
        # build the graph
        build_graph(args.root, args.data_dir)
    elif args.command == "query": 
        print(query(args.root, args.method, args.query))

    elif args.command == "evaluate":
        print(evaluate_all(root = args.root, evaluate_k=args.evaluate_k))




if __name__ == "__main__":
    main()







