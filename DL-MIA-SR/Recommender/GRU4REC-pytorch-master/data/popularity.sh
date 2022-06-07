#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=popularity
#SBATCH -o out_popularity
#SBATCH -e err_popularity
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:0
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7

python popularity.py


EOT