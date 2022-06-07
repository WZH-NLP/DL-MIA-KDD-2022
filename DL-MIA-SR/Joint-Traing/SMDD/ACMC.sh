#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=ACMC
#SBATCH -o out_ACMC
#SBATCH -e err_ACMC
#SBATCH -p debug
#SBATCH --nodelist gpu06
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00

# Set-up the environment.
source ${HOME}/.bashrc
conda activate torch1.7

python ../train.py \
    --debug 1 \
    --save_prefix vgvae-exp \
    --checkpointD_dir CheckPoint/ACMC/Disentangle \
    --checkpointC_dir CheckPoint/ACMC/Classifier \
    --checkpoint_dir CheckPoint/ACMC/DCR \
    --decoder_type lstm \
    --yencoder_type bilstm \
    --zencoder_type bilstm \
    --D_epoch 200 \
    --C_epoch 20 \
    --n_epoch 100 \
    --alpha 0.7 \
    --beta 0.0 \
    --pre_train_emb 1 \
    --learning_rate 0.001 \
    --vocab_size 1092735 \
    --pair_size 9137 \
    --batch_size 2 \
    --dropout 0.0 \
    --l2 0.0 \
    --max_vmf_kl_temp 1e-4 \
    --max_gauss_kl_temp 1e-3 \
    --zmlp_n_layer 0 \
    --ymlp_n_layer 0 \
    --mlp_n_layer 3 \
    --mega_batch 20 \
    --para_logloss_ratio 1.0 \
    --ploss_ratio 1.0 \
    --disc_ratio 1.0 \
    --mlp_hidden_size 100 \
    --ysize 50 \
    --zsize 50 \
    --embed_dim 1 \
    --encoder_size 50 \
    --decoder_size 100 \
    --p_scramble 0.0 \
    --print_every 1000 \
    --eval_every 1000 \
    --summarize 0 \
    --is_eval 0 \
    --train_file ../dataset/PairVectorToTrain/ACMCpairvector.txt \
    --embed_file ../dataset/EmbeddingFile/ACMCembedding.txt \
    --vocab_file ../dataset/vocab_file/ACMC \
    --raw_feature1 ../../Biased-Attack/SMDD/AttackData/trainForClassifier_ACMC.txt \
    --raw_feature2 ../../Biased-Attack/SMDD/AttackData/testForClassifier_ACMC.txt \
    --shadow_size 9137 \
    --target_size 1939

EOT