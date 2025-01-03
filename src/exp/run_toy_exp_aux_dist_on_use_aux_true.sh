#!/bin/bash

cd ..
GPU_IDS=(0 1 2 3 4 5 6 7)  # Adjust GPU IDs as needed
IDX=0

TRAINING_SCRIPT="run_toy.py"

# Fixed parameters
SEED=0
EPOCHS=500
DATASET='data_v10'
EXP_MODE='True'
SAVE_BOOL='True'
IS_DEPLOYMENT='False'
DIM_Z=32

hidden_q_z_values=([64,32] [64])
hidden_q_y_values=([64] [128,64])
hidden_q_t_values=([64] [128] [128,64])
hidden_q_e_values=([32] [16] [32,16])

hidden_p_x_values=([64] [64,64] [64,64,64])
hidden_p_y_values=([64] [64,64] [64,64,64])

ALPHA_RANGE=([1.0,100.0] [0.1,1.0])
GAMMA_RANGE=([10.0,1.0,1.0] [10.0,1.0,0.1])
BETA_RANGE=(1e-0 10 1e-3)

# Hyperparameter ranges
BATCH_SIZE=(512)
OPTIM=("adam" "adamw")
LR_RANGE=(1e-6 1e-4 1e-2)
WD_RANGE=(1e-6 1e-2 1e-4)
DIM_PROJ=(32 16)
PROJ_GRAD=(True False)

for BS in "${BATCH_SIZE[@]}"
do
for OPT in "${OPTIM[@]}"
do
for LR in "${LR_RANGE[@]}"
do
for WD in "${WD_RANGE[@]}"
do
for DIM_P in "${DIM_PROJ[@]}"
do
for PROJ_G in "${PROJ_GRAD[@]}"
do
for HQZ in "${hidden_q_z_values[@]}"
do
for HQY in "${hidden_q_y_values[@]}"
do
for HQT in "${hidden_q_t_values[@]}"
do
for HQE in "${hidden_q_e_values[@]}"
do
for HPX in "${hidden_p_x_values[@]}"
do
for HPY in "${hidden_p_y_values[@]}"
do
for ALPHA in "${ALPHA_RANGE[@]}"
do
for GAMMA_W in "${GAMMA_RANGE[@]}"
do
for BETA in "${BETA_RANGE[@]}"
do

CUDA_VISIBLE_DEVICES=${GPU_IDS[$IDX]} python $TRAINING_SCRIPT \
--seed $SEED \
--gpu_id ${GPU_IDS[$IDX]} \
--exp_mode $EXP_MODE \
--dataset $DATASET \
--batch_size $BS \
--num_epochs $EPOCHS \
--save_results $SAVE_BOOL \
--optim $OPT \
--lr $LR \
--wd $WD \
--dim_projection $DIM_P \
--projection_grad $PROJ_G \
--is_deployment_mode $IS_DEPLOYMENT \
--dim_z $DIM_Z \
--hidden_q_z "$HQZ" \
--hidden_q_y "$HQY" \
--hidden_q_t "$HQT" \
--hidden_q_e "$HQE" \
--hidden_p_x "$HPX" \
--hidden_p_y "$HPY" \
--alpha "$ALPHA" \
--gamma "$GAMMA_W" \
--beta $BETA 2>&1 &

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
wait