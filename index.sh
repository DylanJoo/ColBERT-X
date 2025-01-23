for step in prepare encode finalize; do
python -m colbert.scripts.index \
--coll_dir /home/jju/datasets/neuclir-csl/csl.tsv \
--index_name neuclir-csl \
--dataset_name test_coll \
--nbits 1 \
--step $step \
--checkpoint hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng \
--experiment test  
done
