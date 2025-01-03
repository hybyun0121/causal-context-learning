#!/bin/bash

cd ..

SEED=0
NUM_SHOTS=3
DATASET='mgsm'
MAX_TOKENS_LENGTH=10
IS_OOP=(True False)
ICL_MODE=("ccl" "icl")
API=("gpt-4o-mini-2024-07-18")

for oop in "${IS_OOP[@]}"
do
for mode in "${ICL_MODE[@]}"
do
for api in "${API[@]}"
do

python run_icl.py \
--seed $SEED \
--num_shots $NUM_SHOTS \
--save_results True \
--use_system_prompt True \
--max_tokens 64 \
--model_name $api \
--is_oop $oop \
--icl_mode $mode
sleep 1

done
done
done