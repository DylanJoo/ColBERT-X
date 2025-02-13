#!/bin/sh
#SBATCH --job-name=l0-glove-cos
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_p40:4
#SBATCH --ntasks-per-node=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=log/%x.out
#SBATCH --error=log/%x.err

. /home/dju/miniconda3/etc/profile.d/conda.sh
conda activate plaid

cd ~/ColBERT-X

# Train from scratch

dataset=/home/dju/datasets/hltcoe/t53b-monot5-msmarco-engeng.jsonl
pretrained_base=xlm-roberta-large

python -m colbert.scripts.train \
--model_name ${pretrained_base} \
--training_triples ${dataset} \
--training_irds_id neumarco/zh/train \
--maxsteps 50000 \
--learning_rate 5e-5 \
--kd_loss KLD \
--per_device_batch_size 8 \
--nway 6 \
--run_tag frozen \
--experiment q-L0-glove \
--other_args lite_query_encoder=True lite_document_encoder=False lite_num_hidden_layers=0 lite_encoder_init=DylanJHJ/glove.6B.300d model_lite_name=DylanJHJ/glove.6B.300d freeze_document_encoder=False shared_linear_lite=True freeze_query_word_embeddings=True do_normalization=True lite_hidden_size=1024 weight_decay=0
