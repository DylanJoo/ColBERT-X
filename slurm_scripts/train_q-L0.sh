#!/bin/sh
#SBATCH --job-name=l0
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X

# Train from scratch

dataset=/home/dju/datasets/hltcoe/t53b-monot5-msmarco-engeng.jsonl
pretrained_base=xlm-roberta-large
# pretrained_base=hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng

python -m colbert.scripts.train \
--model_name ${pretrained_base} \
--training_triples ${dataset} \
--training_irds_id neumarco/zh/train \
--maxsteps 100000 \
--learning_rate 5e-6 \
--kd_loss KLD \
--only_top \
--per_device_batch_size 16 \
--nway 6 \
--run_tag fromscratch \
--experiment colbert-lite-q-L0 \
--other_args lite_query_encoder=True lite_document_encoder=False lite_num_hidden_layers=0 lite_num_attention_heads=16 lite_encoder_init=${pretrained_base} freeze_document_encoder=False
