#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=dataPreProcess
#SBATCH -o out_dataPreProcess
#SBATCH -e err_dataPreProcess
#SBATCH -p debug
#SBATCH --nodelist gpu06
#SBATCH --gres=gpu:0
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7

python dataPreProcess.py

EOT