#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=DMDD_MA_attack
#SBATCH -o out_DMDD_MA
#SBATCH -e err_DMDD_MA
#SBATCH -p debug
#SBATCH --nodelist gpu01
#SBATCH --gres=gpu:0
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.4

python MBAC.py
python MBAG.py
python MCAB.py
python MCAG.py
python MGAB.py
python MGAC.py

EOT