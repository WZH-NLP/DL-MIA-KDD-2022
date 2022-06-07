#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=beauty_gru4rec_T
#SBATCH -o out_beauty_gru4rec_T
#SBATCH -e err_beauty_gru4rec_T
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python main_T.py

# python main_T.py --is_eval --load_model beauty_checkpoint_T/01041700/model_00009.pt


EOT