# Run In-context Learning
import os
import sys
sys.path.append('./utils')
import time
import pickle
import argparse

import torch
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas  as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import set_seed_all, str2bool
from dataset import ICLDataset
from icl_utils import constructPrompt, KATE, save_path_selected_indices, save_icl_prompt, save_real_time, prompt_format, call_llms, prob_based_selection

parser = argparse.ArgumentParser()

# Experiment settings
parser.add_argument('--seed',              type=int,                                         default=0        )
parser.add_argument('--gpu_id',            type=int,                                         default=1        )
parser.add_argument('--dataset',           type=str,                                         default='mgsm'   )
parser.add_argument('--is_oop',            type=str2bool,  choices=[True, False],            default=False    )
parser.add_argument('--icl_mode',          type=str, choices=['ccl', 'icl', 'zs'],           default='ccl'    )
parser.add_argument('--num_shots',         type=int,                                         default=5        )
parser.add_argument('--batch_size',        type=int,                                         default=256      )
parser.add_argument('--emb_model',         type=str, choices=['llama', 'gpt', 'bert'],       default='llama'  )
parser.add_argument('--dataset_name',      type=str,                                         default='sst5',
                    choices=['dynasent', 'semeval', 'sst5', 'toxigen', 'anli', 'wanli', 'conll', 'wnut', 'advqa', 'newsqa'])
parser.add_argument('--task_id',           type=str, choices=['sa','nli','eqa','ner','td'],  default='sa')
parser.add_argument('--get_idx_only',      type=str2bool,   choices=[True, False],           default=False     )
parser.add_argument('--in_same_task',      type=str2bool,   choices=[True, False],           default=False     )
parser.add_argument('--exp_id',            type=str,                                         default='gogo'    )

# LLMs
parser.add_argument('--model_name',        type=str,                                         default=None     )
parser.add_argument('--cache_dir',         type=str,                                         default=None     )
parser.add_argument('--max_tokens',        type=int,                                         default=1024     )
parser.add_argument('--use_system_prompt', type=str2bool,   choices=[True, False],           default=True     )

# ICL method [KATE]
parser.add_argument('--icl_method',        type=str,        choices=['kate','prob'],               default='kate'   )
parser.add_argument('--metric',            type=str,        choices=['cosine','mmd','kl','mean'],   default='cosine' )
parser.add_argument('--num_neighbors',     type=int,                                                default=100      )

# etc.
parser.add_argument('--save_results',      type=str2bool,   choices=[True, False],  default=True    )

def retrive_samples(args, emb_demo, emb_target, train_indices, num_neighbors):
    direc_demo_indices, path_demo_indices = save_path_selected_indices(args)

    # if not os.path.exists(path_demo_indices):
    # print("No exist the saved selected indices")
    # print(f"Get examples based on {args.icl_mode} by using {args.icl_method}")

    if args.icl_method == 'kate':
        data = KATE(metric=args.metric, emb_demo=emb_demo, emb_target=emb_target, 
                    train_indices=train_indices, num_neighbors=num_neighbors)
    elif args.icl_method == 'prob':
        data = prob_based_selection(args.metric, emb_demo, emb_target, train_indices, num_neighbors=num_neighbors)

    if args.save_results:
        print(f"Save example indices in {path_demo_indices}")
        if not os.path.exists(direc_demo_indices):
            os.mkdir(direc_demo_indices)
        with open(path_demo_indices, "wb") as f:
            pickle.dump(data, f)
    # else:
    #     print(f"The saved selected indices exist in {path_demo_indices}")
    #     with open(path_demo_indices, 'rb') as f:
    #         data = pickle.load(f)

    return data['selected_idx']

def retrival_examples(args):
    if args.is_oop:
        df_pool = pd.read_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_inpool_{args.exp_id}.pkl")
        df_target = pd.read_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_oop_{args.exp_id}.pkl")
    else:
        df_pool = pd.read_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_inpool_{args.exp_id}.pkl")
        total_len = range(len(df_pool))
        train_indices, test_indices = train_test_split(total_len, test_size=0.3)
        df_target = df_pool.iloc[test_indices].reset_index(drop=True)
        df_pool = df_pool.iloc[train_indices].reset_index(drop=True)

    if args.icl_mode == 'ccl':
        if args.icl_method == "prob":
            q_demo = torch.load(f"../results/{args.dataset}/posterior_q_{args.emb_model}_emb_inpool_{args.exp_id}.pth", map_location='cpu', weights_only=True)
            q_target = torch.load(f"../results/{args.dataset}/posterior_q_{args.emb_model}_emb_oop_{args.exp_id}.pth", map_location='cpu', weights_only=True)
            demo_idx = retrive_samples(args, q_demo, q_target, train_indices=range(len(df_pool)), num_neighbors=args.num_neighbors)
        else:
            emb_demo = np.array(df_pool['Z_hat'].tolist())
            emb_target = np.array(df_target['Z_hat'].tolist())
            demo_idx = retrive_samples(args, emb_demo, emb_target, train_indices=range(len(df_pool)), num_neighbors=args.num_neighbors)
    else:
        emb_demo = np.array(df_pool['X'].tolist())
        emb_target = np.array(df_target['X'].tolist())
        demo_idx = retrive_samples(args, emb_demo, emb_target, train_indices=range(len(df_pool)), num_neighbors=args.num_neighbors)

    icl_prompt = constructPrompt(dataset=args.dataset, df=df_pool, indices=demo_idx[:, :args.num_shots], template_idx=0)

    direc_demo_indices, path_demo_indices = save_icl_prompt(args)
    # if not os.path.exists(path_demo_indices):
    if args.save_results:
        direc_demo_indices, path_demo_indices = save_icl_prompt(args)
        if not os.path.exists(path_demo_indices):
            print(f"No exist the saved {args.icl_mode} prompt")
            print(f"Save {args.icl_mode} prompt in {path_demo_indices}")
            if not os.path.exists(direc_demo_indices):
                os.mkdir(direc_demo_indices)
            with open(path_demo_indices, "wb") as f:
                pickle.dump(icl_prompt, f)
            if args.is_oop:
                df_target.to_pickle(os.path.join(direc_demo_indices, f"data_target_seed_{args.seed}_{args.emb_model}_emb_oop_{args.exp_id}.pkl"))
            else:
                df_target.to_pickle(os.path.join(direc_demo_indices, f"data_target_seed_{args.seed}_{args.emb_model}_emb_inpool_{args.exp_id}.pkl"))
    # else:
    #     print(f"Exists: {path_demo_indices}")
    #     if args.is_oop:
    #         PATH = os.path.join(direc_demo_indices, f"data_target_seed_{args.seed}_{args.emb_model}_emb_oop.pkl")
    #     else:
    #         PATH = os.path.join(direc_demo_indices, f"data_target_seed_{args.seed}_{args.emb_model}_emb_inpool.pkl")
    #     print(f"Load {PATH}")
    #     df_target = pd.read_pickle(PATH)

    #     with open(path_demo_indices, "rb") as f:
    #         icl_prompt = pickle.load(f)

    return df_target, icl_prompt

def get_response(args, dataset):
    llm = call_llms(args)
    results = []
    total_samples = len(dataset)
    batch_size = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, total_samples, batch_size)):
            messages, answer = dataset[i]

            start_time = time.time()
            responses = llm.get_response(messages, max_tokens=args.max_tokens)
            print(f"Get {batch_size} Responses in {round((time.time() - start_time)/60, 4)}min")

            result = [{
                'input':messages[0]['content'],
                'response': responses[0],
                'reference': answer
            }]

            if args.save_results:
                save_real_time(args, result)
            results.append(result)

    return result

def retrival_and_save_idx_only(args):
    df_pool = pd.read_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_inpool.pkl")
    df_target = pd.read_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_{args.task_id}_{args.dataset_name}_oop.pkl")

    if args.in_same_task:
        df_pool = df_pool[df_pool['Task']==args.task_id]

    emb_demo = np.array(df_pool['Z_hat'].tolist())
    emb_target = np.array(df_target['Z_hat'].tolist())

    retrive_samples(args, emb_demo, emb_target, train_indices=range(len(df_pool)), num_neighbors=args.num_neighbors)
    print("Save indeces of selected samles")
    
if __name__=='__main__':
    args = parser.parse_args()
    set_seed_all(args.seed)
    args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    if args.model_name is None:
        raise ValueError("Model name is None")
    
    if args.get_idx_only:
        retrival_and_save_idx_only(args)
    
    else:
        df_target, icl_prompt = retrival_examples(args)

        if args.icl_mode=='zs':
            is_zeroshot = True
        else:
            is_zeroshot = False

        dataset = ICLDataset(df_target, icl_prompt, prompt_format[args.dataset.upper()], is_zeroshot=is_zeroshot)

        LLM = args.model_name.split("-")[0]
        if LLM == 'api':
            LLM='llama'
        elif LLM == 'gpt':
            LLM = args.model_name.split("-")[2]
        elif LLM == 'meta':
            LLM = 'llama'

        if args.is_oop:
            PATH = f"../results/{args.dataset}/icl/response_results_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}_oop_{args.exp_id}.pkl"
        else:
            PATH = f"../results/{args.dataset}/icl/response_results_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}_inpool_{args.exp_id}.pkl"

        if os.path.exists(PATH):
            with open(PATH, 'rb') as f:
                logs = pickle.load(f)
            start_idx = len(logs['results'])
            dataset = Subset(dataset, list(range(start_idx, len(dataset))))
            print("Number of remain samples : ", len(dataset))

        results = get_response(args, dataset)