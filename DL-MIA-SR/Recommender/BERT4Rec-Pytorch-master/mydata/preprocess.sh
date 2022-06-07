#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=process
#SBATCH -o out_process_beauty
#SBATCH -e err_process_beauty
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python preprocess.py


EOT