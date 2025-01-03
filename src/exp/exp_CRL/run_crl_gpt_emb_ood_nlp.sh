#!/bin/bash

cd ../..
GPU_IDS=(7) # 1 2 3 4 5 6 7)
IDX=0

TRAINING_SCRIPT="run_crl.py"

DATASET='ood_nlp'
DIM_EMB=1536
LATENT_DIM_RANGE=(512)
EMB_MODEL='gpt'

SEED=0
EPOCHS=50
INPUT_MODE='add'

BATCH_SIZE=(256)
BETA_RANGE=(-1)
OPTIM=("adamw")
LR_RANGE=(1e-3 1e-4)
WD_RANGE=(1e-4 1e-2)
KL_UPDATE=(-1)
LR_SCHEDULER=(True False)
LR_WARMUP=(False True)
LOSS_WEIGHT=(True False)
COND_VAR_OPTIONS=(xyte)
REG_MMD_OPTIONS=(True)
MMD_GAMMA_OPTIONS=(0.001)
BETA_MMD_OPTIONS=(10 1e-1)
KL_BETA_OPTIONS=(1e-2 1e-6)
KL_BETA_MODE_OPTIONS=(logistic linear)

DIM_H_ENCODER=('[1536,1024,512,512]' '[1536,768,512]' '[1536,1024,1024,512]')
DIM_H_DECODER=('[512,1024,1024,1024,1536]' '[512,768,1024,1536]' '[512,1024,1536]')
DIM_H_AUX=('[512,1024,1024,1536]' '[512,768,1536]' '[512,1024,1536]')
DIM_H_T=('[512,512]' '[768,512]' '[1024,512]')

for BS in "${BATCH_SIZE[@]}"; do
for BETA in "${BETA_RANGE[@]}"; do
for OPT in "${OPTIM[@]}"; do
for LR in "${LR_RANGE[@]}"; do
for WD in "${WD_RANGE[@]}"; do
for LR_SCH in "${LR_SCHEDULER[@]}"; do
for LR_WARM in "${LR_WARMUP[@]}"; do
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

CUDA_VISIBLE_DEVICES=$gpu python $TRAINING_SCRIPT \
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
--kl_beta_k $KB \
--kl_beta_mode $KBM \
--call_kl $KLU \
--use_mmd_reg $MMD \
--beta_mmd $BM \
--gamma_ze $GAMMA \
--dim_h $DEC \
--dim_h_aux $AUX \
--dim_h_q $ENC \
--dim_h_t $T \
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