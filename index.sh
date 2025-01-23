for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir ./test_coll \
--index_name test_index \
--dataset_name test_coll \
--nbits 1 \
--step $step \
--checkpoint experiments/test2/none/test2/8bat.6way/checkpoints/colbert/ \
--experiment test 
done
# --checkpoint eugene-yang/plaidx-xlmr-large-mlir-neuclir \
