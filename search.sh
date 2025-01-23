# baseline
# python -m colbert.scripts.search \
# --index_name neuclir-csl \
# --passage_mapping /home/jju/datasets/neuclir-csl/csl_mapping.tsv \
# --query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
# --metrics nDCG@20 MAP R@100 R@1000 Judged@10 \
# --qrel neuclir/tech_final_qrels.txt  \
# --experiment test

# checkpoint=experiments/test2/none/test2/8bat.6way/checkpoints/colbert
# python -m colbert.scripts.search \
# --index_name test_index \
# --checkpoint_path  ${checkpoint} \
# --passage_mapping /home/jju/datasets/neuclir-csl/csl_mapping.tsv \
# --query_file neuclir/neuclir-2023-technical_topics.0719.tsv  \
# --metrics nDCG@20 MAP R@100 R@1000 Judged@10 \
# --qrel neuclir/tech_final_qrels.txt  \
# --experiment test
