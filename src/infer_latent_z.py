import os
import sys
import pickle
sys.path.append('./gcr')
sys.path.append('./utils')

import torch
import torch.nn as nn
import torch.distributions as dist

import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from LCRL import GenCausalRepresentation
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, LambdaLR
from dataset import RealWorldDataset, RealWorldLargeDataset

parser = argparse.ArgumentParser()
# Overall settings
parser.add_argument('--seed',              type=int,                              default=0      )
parser.add_argument('--gpu_id',            type=int,                              default=1      )
parser.add_argument('--exp_id',            type=str,                              default='gogo' )

# Experiments settings
parser.add_argument('--dataset',           type=str,                              default='mgsm'   )
parser.add_argument('--dataset_name',      type=str,                              default='amazon' )
parser.add_argument('--batch_size',        type=int,                              default=32     )
parser.add_argument('--save_results',      type=str2bool,  choices=[True, False], default=False  )
parser.add_argument('--is_oop',            type=str2bool,  choices=[True, False], default=True  )
parser.add_argument('--emb_model',         type=str,       choices=['llama', 'gpt', 'bert'], default='gpt')
parser.add_argument('--do_recon_t',        type=str2bool,  choices=[True, False], default=True  )
parser.add_argument('--pro_e',             type=str2bool,  choices=[True, False], default=True  )
parser.add_argument('--dist_reg',          type=str,       choices=['mmd', 'hsic'], default='mmd')

def infer(gcr, dataloader, do_recon_t):
    gcr.eval()
    all_z_samples, all_idx, loc_list, scale_list = [], [], [], []
    all_x_hat, all_y_hat = [], []
    if do_recon_t:
        all_t_hat = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            xs = batch['X'].to(gcr.device, dtype=torch.float32)
            envs = batch['E'].to(gcr.device, dtype=torch.float32)
            ts = batch['T'].to(gcr.device, dtype=torch.float32)
            idx = batch['index'].to(gcr.device, dtype=torch.long)

            # Forward pass (Experiment mode)
            if do_recon_t:
                y_hat, z_sample, q_z, x_hat, t_hat, reg_mmd_ze = gcr.inference(xs, envs, ts)
            else:
                y_hat, z_sample, q_z, x_hat, reg_mmd_ze = gcr.inference(xs, envs, ts)

            all_z_samples.append(z_sample.detach().cpu().numpy())
            all_idx.append(idx.detach().cpu().numpy())
            loc_list.append(q_z.loc.detach().cpu())
            scale_list.append(q_z.scale.detach().cpu())

            if do_recon_t:
                all_t_hat.append(t_hat.detach().cpu().numpy())
            all_y_hat.append(y_hat.detach().cpu().numpy())
            all_x_hat.append(x_hat.detach().cpu().numpy())

    all_z_samples = np.concatenate(all_z_samples, axis=0)
    all_idx = np.concatenate(all_idx, axis=0)
    q_z_loc = torch.concatenate(loc_list, dim=0)
    q_z_scale = torch.concatenate(scale_list, dim=0)

    all_y_hat = np.concatenate(all_y_hat, axis=0)
    all_x_hat = np.concatenate(all_x_hat, axis=0)
    if do_recon_t:
        all_t_hat = np.concatenate(all_t_hat, axis=0)
        return all_z_samples, all_idx, q_z_loc, q_z_scale, all_x_hat, all_y_hat, all_t_hat
    else:
        return all_z_samples, all_idx, q_z_loc, q_z_scale, all_x_hat, all_y_hat

def plot_results(args, df_inpool, df_oop):
    z_ip = np.stack(df_inpool['Z_hat'].values)
    z_oop = np.stack(df_oop['Z_hat'].values)

    K=3
    num_same_in_topk=0
    dist_matrix = pairwise.cosine_similarity(X=torch.from_numpy(z_oop), Y=torch.from_numpy(z_ip))
    topk_indices = torch.topk(torch.from_numpy(dist_matrix), K, dim=-1).indices
    
    for p in range(0, 250):
        for i in range(0, 2000, 250):
            k = p + i
            if len(np.where(topk_indices[p,:K]==k)[0]) != 0:
                num_same_in_topk += 1
            if len(np.where(topk_indices[p+250,:K]==k)[0]) != 0:
                num_same_in_topk += 1
            if len(np.where(topk_indices[p+500,:K]==k)[0]) != 0:
                num_same_in_topk += 1

    df_oop['Index_E'] = df_oop['Environment'].apply(lambda x: x.split(" ")[-1])
    df_oop['Index_T'] = df_oop['Task'].apply(lambda x: 0)

    df_tmp = pd.concat([df_oop[(df_oop['Index_T']==0)&(df_oop['Index_E']=='Thai')].sample(100),
                        df_oop[(df_oop['Index_T']==0)&(df_oop['Index_E']=='Telugu')].sample(100),
                        df_oop[(df_oop['Index_T']==0)&(df_oop['Index_E']=='Bengali')].sample(100)],axis=0)
    emb = np.array(df_tmp['Z_hat'].to_list())
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(emb)
    df_tmp['TSNE-1'] = tsne_result[:, 0]
    df_tmp['TSNE-2'] = tsne_result[:, 1]

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x='TSNE-1',
        y='TSNE-2',
        hue='Index_E',
        data=df_tmp,
        s=100,
        marker='o',
        legend='full'
    )

    plt.title(f'Num of same problem in TopK: {num_same_in_topk}')
    plt.xlabel('TSNE-1')
    plt.ylabel('TSNE-2')
    plt.savefig(f"../results/mgsm/plots/Xc_tSNE_{args.emb_model}_emb_oop_{args.exp_id}.png")

def extract_inpool(args, model, do_recon_t):
    if not args.dataset in ['ood_nlp', 'exp_5']:
        df = pd.read_pickle(f"../data/{args.dataset}/dataset_{args.emb_model}_emb_inpool.pkl")
        dataset = RealWorldDataset(df)
    else:
        df = pd.read_csv(f"../data/{args.dataset}/df_inpool.csv")
        X_emb = torch.load(f"../data/{args.dataset}/embeddings/input_emb_gpt_inpool.pt", map_location='cpu', weights_only=True)
        Y_emb = torch.load(f"../data/{args.dataset}/embeddings/output_emb_gpt_inpool.pt", map_location='cpu', weights_only=True)
        with open(f'../data/{args.dataset}/embeddings/task2emb.pkl', 'rb') as f:
            task2emb = pickle.load(f)
        with open(f'../data/{args.dataset}/embeddings/env2emb.pkl', 'rb') as f:
            env2emb = pickle.load(f)
        dataset = RealWorldLargeDataset(df, X_emb=X_emb, Y_emb=Y_emb, task2emb=task2emb, env2emb=env2emb, is_oop=args.is_oop)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if do_recon_t:
        z_infered, idx, q_z_loc, q_z_scale, all_x_hat, all_y_hat, all_t_hat = infer(model, dataloader, do_recon_t)
    else:
        z_infered, idx, q_z_loc, q_z_scale, all_x_hat, all_y_hat = infer(model, dataloader, do_recon_t)

    assert len(df) == len(z_infered), f"len(df): {len(df)} != len(z_infered): {len(z_infered)}"
    missing_idx = set(idx) - set(df.index)
    assert len(missing_idx) == 0, f"Some indices in `idx` are not present in `df.index`: {missing_idx}"

    df_sorted = df.loc[idx].copy()
    df_sorted['Z_hat'] = z_infered.tolist()
    df_sorted['Z_hat'] = df_sorted['Z_hat'].apply(lambda x: np.array(x))

    df_sorted['X_hat'] = all_x_hat.tolist()
    df_sorted['X_hat'] = df_sorted['X_hat'].apply(lambda x: np.array(x))
    df_sorted['Y_hat'] = all_y_hat.tolist()
    df_sorted['Y_hat'] = df_sorted['Y_hat'].apply(lambda x: np.array(x))
    
    if args.do_recon_t:
        df_sorted['T_hat'] = all_t_hat.tolist()
        df_sorted['T_hat'] = df_sorted['T_hat'].apply(lambda x: np.array(x))

    df_sorted.to_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_inpool_{args.exp_id}.pkl")
    state_dict={'loc': q_z_loc, 'scale': q_z_scale}
    torch.save(state_dict, f'../results/{args.dataset}/posterior_q_{args.emb_model}_emb_inpool_{args.exp_id}.pth')

    return df_sorted

def extract_oop(args, model, do_recon_t):
    if not args.dataset in ['ood_nlp', 'exp_5']:
        df = pd.read_pickle(f"../data/{args.dataset}/dataset_{args.emb_model}_emb_oop.pkl")
        dataset = RealWorldDataset(df, is_oop=args.is_oop)
    else:
        if args.dataset_name in ['dynasent', 'semeval', 'sst5']:
            data_folder = 'SentimentAnalysis'
            task_id = 'sa'
        elif args.dataset_name in ['anli','wanli']:
            data_folder = 'NaturalLanguageInference'
            task_id = 'nli'
        elif args.dataset_name in ['advqa','newsqa']:
            data_folder = 'QuestionAnswering'
            task_id = 'eqa'
        elif args.dataset_name in ['conll', 'wnut']:
            data_folder = 'NameEntityRecognition'
            task_id = 'ner'
        elif args.dataset_name in ['tg']:
            data_folder = 'ToxicDetection'
            task_id = 'td'

        df = pd.read_csv(f"../data/{args.dataset}/{data_folder}/{args.dataset_name}/df_oop.csv")
        X_emb = torch.load(f"../data/{args.dataset}/embeddings/input_emb_gpt_{task_id}_{args.dataset_name}_oop.pt", map_location='cpu', weights_only=True)
        with open(f'../data/{args.dataset}/embeddings/task2emb.pkl', 'rb') as f:
            task2emb = pickle.load(f)
        with open(f'../data/{args.dataset}/embeddings/env2emb.pkl', 'rb') as f:
            env2emb = pickle.load(f)
        T_emb = task2emb[task_id]
        E_emb = env2emb[args.dataset_name]
        dataset = RealWorldLargeDataset(df, X_emb=X_emb, Y_emb=None, task2emb=None, env2emb=None, T_emb=T_emb, E_emb=E_emb, is_oop=args.is_oop)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if do_recon_t:
        z_infered, idx, q_z_loc, q_z_scale, all_x_hat, all_y_hat, all_t_hat = infer(model, dataloader, do_recon_t)
    else:
        z_infered, idx, q_z_loc, q_z_scale, all_x_hat, all_y_hat = infer(model, dataloader, do_recon_t)

    assert len(df) == len(z_infered), f"len(df): {len(df)} != len(z_infered): {len(z_infered)}"
    missing_idx = set(idx) - set(df.index)
    assert len(missing_idx) == 0, f"Some indices in `idx` are not present in `df.index`: {missing_idx}"

    df_sorted = df.loc[idx].copy()
    df_sorted['Z_hat'] = z_infered.tolist()
    df_sorted['Z_hat'] = df_sorted['Z_hat'].apply(lambda x: np.array(x))

    df_sorted['X_hat'] = all_x_hat.tolist()
    df_sorted['X_hat'] = df_sorted['X_hat'].apply(lambda x: np.array(x))
    df_sorted['Y_hat'] = all_y_hat.tolist()
    df_sorted['Y_hat'] = df_sorted['Y_hat'].apply(lambda x: np.array(x))
    
    if args.do_recon_t:
        df_sorted['T_hat'] = all_t_hat.tolist()
        df_sorted['T_hat'] = df_sorted['T_hat'].apply(lambda x: np.array(x))

    if not args.dataset in ['ood_nlp', 'exp_5']:
        df_sorted.to_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_oop_{args.exp_id}.pkl")
        state_dict={'loc': q_z_loc, 'scale': q_z_scale}
        torch.save(state_dict, f'../results/{args.dataset}/posterior_q_{args.emb_model}_emb_oop_{args.exp_id}.pth')
    else:
        df_sorted.to_pickle(f"../data/{args.dataset}/icl_dataset/data_{args.emb_model}_emb_oop_{args.exp_id}.pkl")
        state_dict={'loc': q_z_loc, 'scale': q_z_scale}
        torch.save(state_dict, f'../results/{args.dataset}/posterior_q_{args.emb_model}_emb_oop_{args.exp_id}.pth')
    return df_sorted

def main(args):
    # Model define
    model_parameters = torch.load(f'../results/{args.dataset}/gcr_{args.emb_model}_emb_weights_{args.exp_id}.pth', weights_only=True, map_location='cpu')

    model_config = model_parameters['config']
    try: setattr(args, 'do_recon_t', model_config['do_recon_t'])
    except: pass

    set_seed_all(args.seed)

    gcr = GenCausalRepresentation(
        dim_h_q = model_config['dim_h_q'],
        dim_h = model_config['dim_h'],
        dim_h_t = model_config['dim_h_t'],
        dim_h_aux = model_config['dim_h_aux'],
        dim_emb = model_config['dim_emb'],
        dim_z = model_config['dim_latent'],
        input_mode = model_config['input_mode'],
        cond_vars = model_config['cond_vars'],
        use_reg = model_config['use_reg'],
        device = args.device,
        pro_e = args.pro_e, # model_config['pro_e'],
        dist_reg = args.dist_reg, # model_config['dist_reg'],
        gamma_ze = model_config['gamma_ze'],
        do_recon_t = model_config['do_recon_t']
    )

    gcr.load_state_dict(model_parameters['model_state_dict'], strict=False)
    gcr.to(args.device)
    
    print("Extract IN-POOL Xc")
    df_z_inpool = extract_inpool(args, gcr, args.do_recon_t)
    print("Extract OOP Xc")
    df_z_oop = extract_oop(args, gcr, args.do_recon_t)

    plot_results(args, df_z_inpool, df_z_oop)

if __name__=='__main__':
    args = parser.parse_args()
    if not os.path.exists(f"../results/{args.dataset}"):
        os.mkdir(f"../results/{args.dataset}")

    args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    main(args)