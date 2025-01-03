import os
import sys
sys.path.append('../')
import pickle
import numpy as np

import torch

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise

# from gcr import mmd_compute

prompt_format = dict()
prompt_format = {
    "MGSM": {
            "columns": {
                       "X": "question",
                       "Y": "answer"
                       },
            "templates": {
                         "X": ["Question: {}\n"],
                         "Y": ["Answer: {}\n\n"],
                         "A": ["Answer: "]
                         }
            }
}

def save_path_selected_indices(args):
    if args.dataset == 'ood_nlp':
        directory = f"../results/{args.dataset}/icl/{args.task_id}/{args.dataset_name}/"
    else:
        directory = f"../results/{args.dataset}/icl/"
    LLM = args.model_name.split("-")[0]
    if LLM == 'api':
        LLM='llama'
    elif LLM == 'gpt':
        LLM = args.model_name.split("-")[2]
    elif LLM == 'meta':
        LLM = 'llama'

    if args.metric is not None:
        PIK = directory + f"selected_indices_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}_{args.metric}"
    else:
        PIK = directory + f"selected_indices_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}"

    if args.in_same_task:
        PIK += "_task_pool"

    if args.is_oop:
        PIK += f"_oop_{args.exp_id}.dat"
    else:
        PIK += f"_inpool_{args.exp_id}.dat"
    return directory, PIK

def save_icl_prompt(args):
    directory = f"../results/{args.dataset}/icl/"
    LLM = args.model_name.split("-")[0]
    if LLM == 'api':
        LLM='llama'
    elif LLM == 'gpt':
        LLM = args.model_name.split("-")[2]
    elif LLM == 'meta':
        LLM = 'llama'

    if args.metric is not None:
        PIK = directory + f"icl_prompt_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}_{args.metric}"
    else:
        PIK = directory + f"icl_prompt_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}"

    if args.is_oop:
        PIK += f"_oop_{args.exp_id}.pkl"
    else:
        PIK += f"_inpool_{args.exp_id}.pkl"
    return directory, PIK

def constructPrompt(dataset, df, indices, template_idx=0):
    x_col = prompt_format[dataset.upper()]['columns']['X']
    y_col = prompt_format[dataset.upper()]['columns']['Y']
    X_templates = prompt_format[dataset.upper()]['templates']['X']
    Y_templates = prompt_format[dataset.upper()]['templates']['Y']
    
    if template_idx == -1:
        i = np.random.choice(len(X_templates), 1)
    else:
        i = template_idx

    prompt_set = []
    for k in range(len(indices)):
        prompt = ''
        for tmp_index in indices[k]:
            prompt += X_templates[i].format(df.loc[tmp_index, x_col])
            prompt += Y_templates[i].format(df.loc[tmp_index, y_col])
        prompt_set.append(prompt)
    return prompt_set

def save_real_time(args, result):
    LLM = args.model_name.split("-")[0]
    if LLM == 'api':
        LLM='llama'
    elif LLM == 'gpt':
        LLM = args.model_name.split("-")[2]
    elif LLM == 'meta':
        LLM = 'llama'
        
    if args.is_oop:
        PATH = f"../results/{args.dataset}/icl/response_results_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}_{args.metric}_oop_{args.exp_id}.pkl"
    else:
        PATH = f"../results/{args.dataset}/icl/response_results_{args.seed}_{args.emb_model}_emb_LLM_{LLM}_{args.icl_mode}_{args.icl_method}_{args.metric}_inpool_{args.exp_id}.pkl"
    
    if not os.path.exists(PATH):
        logs={
            'results':result,
            'arguments':args
        }
        with open(PATH, "wb") as f:
            pickle.dump(logs, f)
    else:
        with open(PATH, 'rb') as f:
            logs = pickle.load(f)
        logs['results'].append(result)

        with open(PATH, "wb") as f:
            pickle.dump(logs, f)

def call_llms(args):
    if args.model_name.split("-")[0] == 'gpt':
        '''
        [gpt-3.5-turbo-0125,
        gpt-4o-mini]
        '''
        from llms import gpt
        setattr(args, 'batch_size', 1)
        llm_engine=gpt.GPTEngine(args.model_name, args.use_system_prompt)

    elif args.model_name.split("-")[0] == 'meta':
        '''
        [meta-llama/Meta-Llama-3.1-8B-Instruct,]
        '''
        from llms import llama
        llm_engine=llama.LLama(args.cache_dir, args.model_name, args.use_system_prompt)
    elif args.model_name.split("-")[0] == 'api':
        '''
        [api-llama-3.1-70b-versatile,
        api-llama-3.1-8b-instant]
        '''
        from llms import api
        setattr(args, 'batch_size', 1)
        llm_engine = api.LLM_API("-".join(args.model_name.split("-")[1:]), args.use_system_prompt)
    
    return llm_engine

def KATE(metric, emb_demo, emb_target, train_indices, num_neighbors, reversed=False):
    data = dict()
    if metric == "euclidean":
        nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_demo)
        distances, indices = nbrs.kneighbors(emb_target)
        data['distances'] = distances
    elif metric == "cosine":
        dist_matrix = pairwise.cosine_similarity(X=emb_target, Y=emb_demo)
        if reversed:
            distances, indices = torch.topk(-torch.from_numpy(dist_matrix), k=num_neighbors, dim=-1)
        else:
            distances, indices = torch.topk(torch.from_numpy(dist_matrix), k=num_neighbors, dim=-1)
        indices = indices.numpy()
        data['distances'] = distances.numpy()

    train_indices_np = np.asarray(train_indices)
    selected_idx = [train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))]
    selected_idx = np.concatenate(selected_idx, axis=0)

    data["selected_idx"] = selected_idx
    return data

def prob_based_selection(metric, q_demo, q_target, train_indices, num_neighbors):
    data = dict()

    q_demo_loc = q_demo['loc']
    q_demo_scale = q_demo['scale']
    q_target_loc = q_target['loc']
    q_target_scale = q_target['scale']

    if metric == 'kl':
        q_demo_loc = q_demo_loc.unsqueeze(0)
        q_demo_scale = q_demo_scale.unsqueeze(0)
        q_target_loc = q_target_loc.unsqueeze(1)
        q_target_scale = q_target_scale.unsqueeze(1)

        kl_diffs = (
            torch.log(q_demo_scale) - torch.log(q_target_scale)
            + (q_target_scale ** 2 + (q_target_loc - q_demo_loc) ** 2) / (2 * q_demo_scale ** 2)
            - 0.5
        )

        kl_diffs = kl_diffs.sum(dim=-1)
        distances, indices = torch.topk(-kl_diffs, k=num_neighbors, dim=1)
        
        indices = indices.numpy()
        data['distances'] = distances.numpy()

    elif metric == 'mean':
        dist_matrix = pairwise.cosine_similarity(X=q_target_loc, Y=q_demo_loc)
        distances, indices = torch.topk(torch.from_numpy(dist_matrix), k=10, dim=-1)
        indices = indices.numpy()
        data['distances'] = distances.numpy()

    elif metric == 'mmd':
        pass

    train_indices_np = np.asarray(train_indices)
    selected_idx = [train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))]
    selected_idx = np.concatenate(selected_idx, axis=0)

    data["selected_idx"] = selected_idx
    return data
