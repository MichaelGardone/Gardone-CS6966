# Assignment 4 - Visualizing Attention Matrices

## Project Structure

data/ - Contains all the input data for the project.

old/ - Prior work on another attempt for Assignment 4.

out/ - All outputs. Each example (of the 10 provided in data/) is labeled as ex_{X}. Each of these subfolders has atten_matrix_[EN/ST], heatmap, t2t. t2t is the token2token comparisons. atten_matrix_[ST/EN] is the start/end matrices from the model's interpretations. Heatmap is a visualization of the heatmaps of the attention matrices.

## To Run

If using a system with SLURM:

`sbatch assignment4.sh`

If you have a powerful enough machine:

`python assignment4.py`
