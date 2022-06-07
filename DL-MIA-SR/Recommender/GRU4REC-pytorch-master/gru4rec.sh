#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=beauty_gru4rec
#SBATCH -o out_beauty_gru4rec_S
#SBATCH -e err_beauty_gru4rec_S
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python main.py
# python main.py --is_eval --load_model beauty_checkpoint_S/01041659/model_00009.pt


EOT