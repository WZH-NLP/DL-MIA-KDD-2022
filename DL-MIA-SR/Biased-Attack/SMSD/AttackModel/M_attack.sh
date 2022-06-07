#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=SMSD_M
#SBATCH -o out_SMSD_M
#SBATCH -e err_SMSD_M
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python MBMB.py
python MCMC.py
python MGMG.py


EOT