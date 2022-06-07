#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=SMDD
#SBATCH -o out_SMDD
#SBATCH -e err_SMDD
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7

python ACMC.py
python AGMG.py

EOT