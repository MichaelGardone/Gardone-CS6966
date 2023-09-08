#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1000771@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_2-%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs6966

mkdir -p /scratch/general/vast/u1000771/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1000771/huggingface_cache"

OUT_DIR=/scratch/general/vast/u1000771/cs6966/assignment2/models
mkdir -p ${OUT_DIR}
python assignment2.py --output_dir ${OUT_DIR}