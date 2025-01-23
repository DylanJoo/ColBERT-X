for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir /home/jju/datasets/neuclir-csl/csl.tsv \
--index_name test_index \
--dataset_name test_coll \
--nbits 1 \
--step $step \
--checkpoint eugene-yang/plaidx-xlmr-large-mlir-neuclir \
--experiment test  
done
