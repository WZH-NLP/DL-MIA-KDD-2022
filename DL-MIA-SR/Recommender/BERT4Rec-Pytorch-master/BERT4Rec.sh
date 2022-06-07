#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=BERT4Rec
#SBATCH -o out_BERT4Rec_Target_beauty
#SBATCH -e err_BERT4Rec_Target_beauty
#SBATCH -p debug
#SBATCH --nodelist gpu02
#SBATCH --gres=gpu:1
#SBATCH --partition=edu
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7

python main.py --template train_bert

EOT