#!/bin/sh
#SBATCH --job-name=debug
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

source ~/.bashrc
conda activate plaid

cd ~/ColBERT-X

python -m colbert.scripts.train \
--model_name xlm-roberta-large \
--training_triples /home/jju/datasets/hltcoe/t53b-monot5-msmarco-engeng.sample.jsonl \
--training_irds_id neumarco/zh/train \
--maxsteps 100 \
--learning_rate 5e-6 \
--kd_loss KLD \
--only_top \
--lite_query_encoder true \
--lite_document_encoder false \
--encoder_num_hidden_layers 1 \
--encoder_num_attention_heads 1 \
--per_device_batch_size 8 \
--nway 6 \
--run_tag test \
--experiment test
