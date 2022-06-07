#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=Attack
#SBATCH -o out_attack
#SBATCH -e err_attack
#SBATCH -p debug
#SBATCH --nodelist gpu03
#SBATCH --gres=gpu:0
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python amazon-itembase_movielens-lfm.py
python amazon-lfm_movielens-itembase.py

python movielens-itembase_amazon-lfm.py
python movielens-lfm_amazon-itembase.py



EOT