#!/bin/bash
cd ..

# text-small-emb
CUDA_VISIBLE_DEVICES=1 nohup python run_crl.py --exp_id gpt_emb --emb_model gpt --save_weights True --train_w_all True &
sleep 5

# Llama-3.1-8B-instruct
CUDA_VISIBLE_DEVICES=2 nohup python run_crl.py --exp_id llama_emb --emb_model llama --save_weights True --train_w_all True &
sleep 5

# RoBERTa-base
CUDA_VISIBLE_DEVICES=3 nohup python run_crl.py --exp_id bert_emb --emb_model bert --save_weights True --train_w_all True &

sleep 5
wait

CUDA_VISIBLE_DEVICES=1 nohup python infer_latent_z.py --dataset mgsm --batch_size 512 --save_results True --is_oop False --emb_model gpt --exp_id gpt_emb &
CUDA_VISIBLE_DEVICES=1 nohup python infer_latent_z.py --dataset mgsm --batch_size 512 --save_results True --is_oop True --emb_model gpt --exp_id gpt_emb &

CUDA_VISIBLE_DEVICES=2 nohup python infer_latent_z.py --dataset mgsm --batch_size 512 --save_results True --is_oop False --emb_model llama --exp_id llama_emb &
CUDA_VISIBLE_DEVICES=2 nohup python infer_latent_z.py --dataset mgsm --batch_size 512 --save_results True --is_oop True --emb_model llama --exp_id llama_emb &

CUDA_VISIBLE_DEVICES=3 nohup python infer_latent_z.py --dataset mgsm --batch_size 512 --save_results True --is_oop False --emb_model bert --exp_id bert_emb &
CUDA_VISIBLE_DEVICES=3 nohup python infer_latent_z.py --dataset mgsm --batch_size 512 --save_results True --is_oop True --emb_model bert --exp_id bert_emb &
wait

nohup python run_icl.py --dataset mgsm --is_oop True --icl_mode ccl --num_shots 3 --batch_size 1 --model_name gpt-4o-mini --max_tokens 1024 --use_system_prompt True --save_results True --emb_model gpt --exp_id gpt_emb &
nohup python run_icl.py --dataset mgsm --is_oop True --icl_mode ccl --num_shots 3 --batch_size 1 --model_name gpt-4o-mini --max_tokens 1024 --use_system_prompt True --save_results True --emb_model llama --exp_id llama_emb &
nohup python run_icl.py --dataset mgsm --is_oop True --icl_mode ccl --num_shots 3 --batch_size 1 --model_name gpt-4o-mini --max_tokens 1024 --use_system_prompt True --save_results True --emb_model bert --exp_id bert_emb &
wait

nohup python run_icl.py --dataset mgsm --is_oop True --icl_mode ccl --num_shots 3 --batch_size 1 --model_name gpt-3.5-turbo-0125 --max_tokens 1024 --use_system_prompt True --save_results True --emb_model gpt --exp_id gpt_emb &
nohup python run_icl.py --dataset mgsm --is_oop True --icl_mode ccl --num_shots 3 --batch_size 1 --model_name gpt-3.5-turbo-0125 --max_tokens 1024 --use_system_prompt True --save_results True --emb_model llama --exp_id llama_emb &
nohup python run_icl.py --dataset mgsm --is_oop True --icl_mode ccl --num_shots 3 --batch_size 1 --model_name gpt-3.5-turbo-0125 --max_tokens 1024 --use_system_prompt True --save_results True --emb_model bert --exp_id bert_emb &