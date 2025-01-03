#!/bin/bash

cd ../..
GPU_IDS=(1 2 3 4 5 6 7) # 1 2 3 4 5 6 7)
IDX=0

TRAINING_SCRIPT="run_crl.py"

DATASET='mgsm'
DIM_EMB=4096
DIM_LATENT=1024
EMB_MODEL='llama'

SEED=0
EPOCHS=500
INPUT_MODE='add'

BATCH_SIZE=(256)
BETA_RANGE=(1e-3 1e-1)
OPTIM=("adamw")
LR_RANGE=(1e-3)
WD_RANGE=(1e-4 1e-2)
KL_UPDATE=(-1 50)
LR_SCHEDULER=(True)
LR_WARMUP=(False True)
LOSS_WEIGHT=(True False)
COND_VAR_OPTIONS=(xyte)
REG_MMD_OPTIONS=(True)
MMD_GAMMA_OPTIONS=(0.001)

for BS in "${BATCH_SIZE[@]}"
do
for BETA in "${BETA_RANGE[@]}"
do
for OPT in "${OPTIM[@]}"
do
for LR in "${LR_RANGE[@]}"
do
for WD in "${WD_RANGE[@]}"
do
for LR_SCH in "${LR_SCHEDULER[@]}"
do
for LR_WARM in "${LR_WARMUP[@]}"
do
for LOWE in "${LOSS_WEIGHT[@]}"
do
for KLU in "${KL_UPDATE[@]}"
do
for VAR in "${COND_VAR_OPTIONS[@]}"
do
for MMD in "${REG_MMD_OPTIONS[@]}"
do
for GAMMA in "${MMD_GAMMA_OPTIONS[@]}"
do

CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python $TRAINING_SCRIPT \
--emb_model $EMB_MODEL \
--batch_size $BS \
--beta $BETA \
--num_epochs $EPOCHS \
--input_mode $INPUT_MODE \
--dataset $DATASET \
--dim_emb $DIM_EMB \
--dim_latent $DIM_LATENT \
--optim $OPT \
--cond_vars $VAR \
--lr $LR \
--wd $WD \
--call_kl $KLU \
--use_mmd_reg $MMD \
--gamma_ze $GAMMA \
--dim_h '[1024,2048,2048,2048,4096]' \
--dim_h_aux '[1024,2048,2048,4096]' \
--dim_h_q '[4096,2048,2048,1024]' \
--dim_h_t '[1024,1024]' \
--use_lr_scheduler $LR_SCH \
--use_lr_warmup $LR_WARM \
--use_loss_weight $LOWE 2>&1 &

IDX=$(( ($IDX + 1) % ${#GPU_IDS[@]} ))
if [ $IDX -eq 0 ]; then
wait
fi
done
done
done
done
done
done
done
done
done
done
done
done
wait