python3 dual_graph_query.py \
    --query "ai đã viết phần độc tấu guitar trong beat it" \
    --graph1_path "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding" \
    --graph2_path "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding" \
    --mapping_file "/home/hungpv/projects/train_embedding/nanographrag/mapping_nq/element_mapping_bgem3_nq_vi_bgem3_nq_en_thresh0.9.json" \
    --mapping_sym_file "/home/hungpv/projects/train_embedding/nanographrag/mapping_nq/element_mapping_bgem3_nq_en_bgem3_nq_en_thresh0.95.json" \
    --output_dir "test" \
    --graph1_name "GRAPH1_NAME" \
    --graph2_name "GRAPH2_NAME" \
    --embedding_func_name "bge-m3-list_des" \
    --embedding_path "BAAI/bge-m3" \
    --devices "cuda:0" \
    --threshold_sym 0.8 \
        # --queries_file "QUERIES_FILE" \