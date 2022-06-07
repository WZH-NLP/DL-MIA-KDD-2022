#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=MCAG
#SBATCH -o out_MCAG
#SBATCH -e err_MCAG
#SBATCH -p debug
#SBATCH --nodelist gpu01
#SBATCH --gres=gpu:0
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7

python MCAG.py


EOT