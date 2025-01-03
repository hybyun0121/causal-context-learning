#!/bin/bash

cd ../..

GPU_IDS=(1 2 3 4 5 6 7)
IDX=0

TRAINING_SCRIPT="run_crl.py"

DATASET='mgsm'
EMB_MODEL='gpt'

DIM_EMB=1536

SEED=0
EPOCHS=1000
INPUT_MODE='concat'
BETA_RANGE=-1
KL_UPDATE=-1
BATCH_SIZE=512

LATENT_DIM_RANGE=(128)
OPTIM=("adamw")
LR_RANGE=(1e-3 1e-6 1e-2)
WD_RANGE=(1e-4 1e-2 1e-3)

LR_SCHEDULER=(True)
LOSS_WEIGHT=(False)
EPRO=(True)
COND_VAR_OPTIONS=(xyte xyt xye)
REG_MMD_OPTIONS=(True)
MMD_GAMMA_OPTIONS=(0.001 0.01)
BETA_MMD_OPTIONS=(5 1)
MMD_REG_MODE_OPTIONS=(sigmoid)
KL_BETA_OPTIONS=(1e-4)
KL_BETA_MODE_OPTIONS=(linear)
LR_SCHEDULER_OPTIONS=(cosine none step)
DIM_H_ENCODER=('[1024,512,256]')
DIM_H_DECODER=('[256,512,1024]')
DIM_H_AUX=('[1024,1024]')
DIM_H_T=('[128,512,1024]')
DO_RECON_T=(False True)
ENV_PROJECTION=(True)
ALPHA=(10 5 1)

for BS in "${BATCH_SIZE[@]}"; do
for BETA in "${BETA_RANGE[@]}"; do
for OPT in "${OPTIM[@]}"; do
for LR in "${LR_RANGE[@]}"; do
for WD in "${WD_RANGE[@]}"; do
for EP in "${ENV_PROJECTION[@]}"; do
for LR_SCH in "${LR_SCHEDULER[@]}"; do
for LOWE in "${LOSS_WEIGHT[@]}"; do
for KLU in "${KL_UPDATE[@]}"; do
for VAR in "${COND_VAR_OPTIONS[@]}"; do
for MMD in "${REG_MMD_OPTIONS[@]}"; do
for GAMMA in "${MMD_GAMMA_OPTIONS[@]}"; do
for LATENT_DIM in "${LATENT_DIM_RANGE[@]}"; do
for ENC in "${DIM_H_ENCODER[@]}"; do
for DEC in "${DIM_H_DECODER[@]}"; do
for AUX in "${DIM_H_AUX[@]}"; do
for T in "${DIM_H_T[@]}"; do
for KB in "${KL_BETA_OPTIONS[@]}"; do
for KBM in "${KL_BETA_MODE_OPTIONS[@]}"; do
for BM in "${BETA_MMD_OPTIONS[@]}"; do
for MRM in "${MMD_REG_MODE_OPTIONS[@]}"; do
for LS in "${LR_SCHEDULER_OPTIONS[@]}"; do
for DRT in "${DO_RECON_T[@]}"; do
for RA in "${ALPHA[@]}"; do

    CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python $TRAINING_SCRIPT \
    --emb_model $EMB_MODEL \
    --batch_size $BS \
    --beta $BETA \
    --num_epochs $EPOCHS \
    --input_mode $INPUT_MODE \
    --dataset $DATASET \
    --dim_emb $DIM_EMB \
    --dim_latent $LATENT_DIM \
    --optim $OPT \
    --cond_vars $VAR \
    --lr $LR \
    --wd $WD \
    --recon_alpha $RA \
    --pro_e $EPRO \
    --kl_beta_k $KB \
    --kl_beta_mode $KBM \
    --do_recon_t $DRT \
    --call_kl $KLU \
    --use_reg $MMD \
    --beta_mmd $BM \
    --mmd_reg_mode $MRM \
    --gamma_ze $GAMMA \
    --dim_h $DEC \
    --dim_h_aux $AUX \
    --dim_h_q $ENC \
    --dim_h_t $T \
    --use_lr_scheduler $LR_SCH \
    --lr_scheduler $LS 2>&1 &

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