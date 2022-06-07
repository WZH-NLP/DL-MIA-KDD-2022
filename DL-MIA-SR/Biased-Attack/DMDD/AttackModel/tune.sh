#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=fine_tune
#SBATCH -o out_fine_tune
#SBATCH -e err_fine_tune
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:0
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7


python ACMB.py
python AGMB.py
python MCAB.py
python MCAG.py


EOT