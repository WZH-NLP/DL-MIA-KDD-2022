#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=preprocess
#SBATCH -o out_preprocess
#SBATCH -e err_preprocess
#SBATCH -p debug
#SBATCH --nodelist gpu07
#SBATCH --gres=gpu:0
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python preprocess.py

EOT