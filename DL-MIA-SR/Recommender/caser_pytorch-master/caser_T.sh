#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=beauty_caser_Target
#SBATCH -o out_beauty_caser_Target
#SBATCH -e err_beauty_caser_Target
#SBATCH -p debug
#SBATCH --nodelist gpu01
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python train_caser_T.py

EOT