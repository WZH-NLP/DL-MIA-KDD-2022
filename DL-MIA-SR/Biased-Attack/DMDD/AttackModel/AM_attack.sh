#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=DMDD_AM_attack
#SBATCH -o out_DMDD_AM
#SBATCH -e err_DMDD_AM
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:0
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python ABMC.py
python ABMG.py
python ACMB.py
python ACMG.py
python AGMB.py
python AGMC.py

EOT