# python entity_mapping.py --graph1_name "tuned_nq_vi" \
#                          --graph2_name "tuned_nq_en" \
#                          --g1_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_entities_bge-nq-tuned.json" \
#                          --g2_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_entities_bge-nq-tuned.json" \
#                          --g1_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_relationships_bge-nq-tuned.json" \
#                          --g2_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_relationships_bge-nq-tuned.json" \
#                          --output "/home/hungpv/projects/train_embedding/nanographrag/mapping_nq" \
#                          --threshold 0.9 \
#                         #  --max_threshold "" \



# python entity_mapping.py --graph2_name "tuned_nq_vi" \
#                          --graph1_name "tuned_nq_en" \
#                          --g2_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_entities_bge-nq-tuned.json" \
#                          --g1_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_entities_bge-nq-tuned.json" \
#                          --g2_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_relationships_bge-nq-tuned.json" \
#                          --g1_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_relationships_bge-nq-tuned.json" \
#                          --output "/home/hungpv/projects/train_embedding/nanographrag/mapping_nq" \
#                          --threshold 0.9 \
#                         #  --max_threshold "" \



python entity_mapping.py --graph1_name "tuned_nq_vi" \
                         --graph2_name "tuned_nq_vi" \
                         --g1_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_entities_bge-nq-tuned.json" \
                         --g2_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_entities_bge-nq-tuned.json" \
                         --g1_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_relationships_bge-nq-tuned.json" \
                         --g2_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_vi_without_embedding/vdb_relationships_bge-nq-tuned.json" \
                         --output "/home/hungpv/projects/train_embedding/nanographrag/mapping_nq" \
                         --threshold 0.94 \
                         --max_threshold 0.98 \



python entity_mapping.py --graph2_name "tuned_nq_en" \
                         --graph1_name "tuned_nq_en" \
                         --g2_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_entities_bge-nq-tuned.json" \
                         --g1_entity_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_entities_bge-nq-tuned.json" \
                         --g2_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_relationships_bge-nq-tuned.json" \
                         --g1_edge_vdb "/home/hungpv/projects/TN/LIGHTRAG/new_nq_en_without_embedding/vdb_relationships_bge-nq-tuned.json" \
                         --output "/home/hungpv/projects/train_embedding/nanographrag/mapping_nq" \
                         --threshold 0.94 \
                         --max_threshold 0.98 \



