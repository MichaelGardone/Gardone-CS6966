#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=60GB
#SBATCH --mail-user=u1000771@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o asg4-%j

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cs6966

mkdir -p /scratch/general/vast/u1000771/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1000771/huggingface_cache"

python assignment4.py