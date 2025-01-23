#!/bin/sh
#SBATCH --job-name=download_data
#SBATCH --partition rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

. /home/jju/temp/miniconda3/etc/profile.d/conda.sh
conda activate plaid

python -m colbert.scripts.collection_utils create_passage_collection \
--root /home/jju/datasets/neuclir1 --corpus neuclir/neuclir1:data/zho-00000-of-00001.jsonl.gz
