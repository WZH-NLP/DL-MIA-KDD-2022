#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=SMSD_A
#SBATCH -o out_SMSD_A
#SBATCH -e err_SMSD_A
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python ABAB.py
python ACAC.py
python AGAG.py



EOT