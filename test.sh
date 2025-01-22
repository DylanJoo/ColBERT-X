conda activate plaid
python -m colbert.scripts.train \
--model_name xlm-roberta-large \
--training_triples /home/jju/datasets/hltcoe/t53b-monot5-msmarco-engeng.sample.jsonl \
--training_irds_id neumarco/zh/train \
--maxsteps 100 \
--learning_rate 5e-6 \
--kd_loss KLD \
--only_top \
--per_device_batch_size 8 \
--nway 6 \
--run_tag test \
--experiment test \
--other_args lite_query_encoder=True lite_num_hidden_layers=1 lite_num_attention_heads=1
