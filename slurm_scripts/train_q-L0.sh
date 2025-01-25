#!/bin/sh
#SBATCH --job-name=l0
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/jju/temp/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X

# [Train from scratch]
pretrained_base=xlm-roberta-large
python -m colbert.scripts.train \
--model_name ${pretrained_base} \
--training_triples /home/jju/datasets/hltcoe/t53b-monot5-msmarco-engeng.jsonl \
--training_irds_id neumarco/zh/train \
--maxsteps 10000 \
--learning_rate 5e-6 \
--kd_loss KLD \
--only_top \
--per_device_batch_size 8 \
--nway 6 \
--run_tag fromscratch \
--experiment colbert-lite \
--other_args lite_query_encoder=True lite_document_encoder=False lite_num_hidden_layers=0 lite_num_attention_heads=16 lite_encoder_init=${pretrained_base} freeze_document_encoder=False

# [Frozen docuemnt encoder] And query encoder is initialized from it as well
# pretrained_base=hltcoe/plaidx-large-zho-tdist-mt5xxl-engeng
# python -m colbert.scripts.train \
# --model_name ${pretrained_base} \
# --training_triples /home/jju/datasets/hltcoe/t53b-monot5-msmarco-engeng.jsonl \
# --training_irds_id neumarco/zh/train \
# --maxsteps 10000 \
# --learning_rate 5e-6 \
# --kd_loss KLD \
# --only_top \
# --per_device_batch_size 8 \
# --nway 6 \
# --run_tag fromscratch \
# --experiment colbert-lite \
# --other_args lite_query_encoder=True lite_document_encoder=False lite_num_hidden_layers=0 lite_num_attention_heads=16 lite_encoder_init=${pretrained_base} freeze_document_encoder=True
