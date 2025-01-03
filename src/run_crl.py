# Run Toy experiments
import os

import torch
import torch.nn as nn
import torch.distributions as dist

import tempfile
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

from utils.utils import *

from LCRL import LatentCausalRepresentationLearner, get_model_config
from torch.utils.data import random_split, DataLoader
from utils.dataset import RealWorldDataset, RealWorldLargeDataset

import wandb

parser = argparse.ArgumentParser()
# Overall settings
parser.add_argument('--seed',              type=int,                             default=0      )
parser.add_argument('--gpu_id',            type=int,                             default=1      )

# Experiments settings
parser.add_argument('--dataset',                type=str,                             default='mgsm' )
parser.add_argument('--batch_size',             type=int,                             default=32     )
parser.add_argument('--num_epochs',             type=int,                             default=10     )
parser.add_argument('--save_results',           type=str2bool,  choices=[True, False], default=False )
parser.add_argument('--save_inference_results', type=str2bool,  choices=[True, False], default=False )

# Learning Settings
parser.add_argument('--optim',             type=str,                             default='adam' )
parser.add_argument('--lr',                type=float,                           default=1e-4   )
parser.add_argument('--lr_gamma',          type=float,                           default=0.98   )
parser.add_argument('--wd',                type=float,                           default=1e-2   )
parser.add_argument('--loss_type',         type=str,                             default='fixed', help='fixed or dynamic')
parser.add_argument('--emb_model',         type=str,                             default='gpt' )

# Model settings
parser.add_argument('--dim_projection',     type=int,                              default=64      )
parser.add_argument('--projection_grad',    type=str2bool,  choices=[True, False], default=False  )
parser.add_argument('--is_deployment_mode', type=str2bool,  choices=[True, False], default=False  )
parser.add_argument('--aux_dist',           type=str2bool,  choices=[True, False], default=True   )
parser.add_argument('--use_aux',            type=str2bool,  choices=[True, False], default=True   )
parser.add_argument('--dim_z',              type=int,                              default=32      )
parser.add_argument('--dim_projection_z',   type=int,                              default=32      )
parser.add_argument('--hidden_q_z',         type=str_to_list,         default=[64,32], help='Hidden dims for q_z')
parser.add_argument('--hidden_q_y',         type=str_to_list,         default=[64,64], help='Hidden dims for q_y')
parser.add_argument('--hidden_q_t',         type=str_to_list,         default=[64,32], help='Hidden dims for q_t')
parser.add_argument('--hidden_q_e',         type=str_to_list,         default=[32,32], help='Hidden dims for q_e')
parser.add_argument('--hidden_p_x',         type=str_to_list,         default=[32,64], help='Hidden dims for p_x')
parser.add_argument('--hidden_p_y',         type=str_to_list,         default=[32,64], help='Hidden dims for p_y')
parser.add_argument('--alpha',              type=str_to_list_float,         default=[1.0,1.0], help='Loss weights for recon')
parser.add_argument('--gamma',              type=str_to_list_float,         default=[1.0,1.0,1.0], help='Loss weights for aux')
parser.add_argument('--beta',               type=float,               default=1e-0   )

def train(LCRL, args, train_loader, opt, recon_loss, epoch):
    LCRL.train()

    running_loss = 0.0
    loss_x, loss_y, loss_aux_y, loss_aux_t, loss_aux_e, loss_kl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss_xc, total_kl_diff = 0.0, 0.0
    
    for k, batch in enumerate(train_loader):
        xs = batch['X'].to(args.device, dtype=torch.float32)
        ys = batch['Y'].to(args.device, dtype=torch.float32)
        envs = batch['E'].to(args.device, dtype=torch.float32)
        ts = batch['T'].to(args.device, dtype=torch.float32)
        tr_idx = batch['index'].to(args.device, dtype=torch.long)

        # Forward pass (Experiments mode)
        losses, z_sample, q_z = LCRL.step(xs, ys, ts, envs, aux_dist=args.aux_dist, use_aux=args.use_aux)

        # Calculate weighted loss components
        if args.loss_type == 'fixed':
            loss = (
                (args.alpha[0] * losses['recon_x_loss']) +
                (args.alpha[1] * losses['recon_y_loss']) +
                (args.gamma[0] * losses['aux_y_loss']) +
                (args.gamma[1] * losses['aux_t_loss']) +
                (args.gamma[2] * losses['aux_e_loss']) +
                (args.beta * losses['kl_loss'])
            ).mean()
        else:  # Dynamic weighting based on relative loss magnitudes
            # Calculate mean losses for normalization
            recon_x_mean = losses['recon_x_loss'].mean()
            recon_y_mean = losses['recon_y_loss'].mean()
            aux_y_mean = losses['aux_y_loss'].mean()
            aux_t_mean = losses['aux_t_loss'].mean() 
            aux_e_mean = losses['aux_e_loss'].mean()
            
            # Calculate relative weights
            total_recon = recon_x_mean + recon_y_mean
            total_aux = aux_y_mean + aux_t_mean + aux_e_mean
            
            # Avoid division by zero
            eps = 1e-8
            recon_x_weight = recon_x_mean / (total_recon + eps)
            recon_y_weight = recon_y_mean / (total_recon + eps)
            aux_y_weight = aux_y_mean / (total_aux + eps)
            aux_t_weight = aux_t_mean / (total_aux + eps)
            aux_e_weight = aux_e_mean / (total_aux + eps)
            
            loss = (
                (recon_x_weight * losses['recon_x_loss']) +
                (recon_y_weight * losses['recon_y_loss']) +
                (aux_y_weight * losses['aux_y_loss']) +
                (aux_t_weight * losses['aux_t_loss']) + 
                (aux_e_weight * losses['aux_e_loss']) +
                (args.beta * losses['kl_loss'])
            ).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item()
        loss_x += losses['recon_x_loss'].sum().detach().cpu().item()
        loss_y += losses['recon_y_loss'].sum().detach().cpu().item()
        loss_aux_y += losses['aux_y_loss'].sum().detach().cpu().item()
        loss_aux_t += losses['aux_t_loss'].sum().detach().cpu().item()
        loss_aux_e += losses['aux_e_loss'].sum().detach().cpu().item()
        loss_kl += losses['kl_loss'].sum().detach().cpu().item()

    current_lr = opt.param_groups[0]['lr']
    print("============")
    print(f'Train Epoch [{epoch + 1}] Current Learning Rate: {current_lr:.6f}')
    print(f'Train Epoch [{epoch + 1}] Total   backward Loss: {round(running_loss / len(train_loader), 8)}')
    print(f'Train Epoch [{epoch + 1}] [X]     Recon.   Loss: {round(loss_x / len(train_loader.dataset), 8)}')
    print(f'Train Epoch [{epoch + 1}] [Y]     Recon.   Loss: {round(loss_y / len(train_loader.dataset), 8)}')
    print(f'Train Epoch [{epoch + 1}] [Aux Y] Recon.   Loss: {round(loss_aux_y / len(train_loader.dataset), 8)}')
    print(f'Train Epoch [{epoch + 1}] [Aux T] Recon.   Loss: {round(loss_aux_t / len(train_loader.dataset), 8)}')
    print(f'Train Epoch [{epoch + 1}] [Aux E] Recon.   Loss: {round(loss_aux_e / len(train_loader.dataset), 8)}')
    print(f'Train Epoch [{epoch + 1}] [KL]    Dive.    Loss: {round(loss_kl / len(train_loader.dataset), 8)}')
    print("@@@@@@@@@@@@")
    print(f'Train Epoch [{epoch + 1}] [Z-Xc]  Recon.   Loss: {round(loss_xc / len(train_loader.dataset), 8)}')
    print(f'Train Epoch [{epoch + 1}] KL-Div  (q_z, p_z) : {round(total_kl_diff / len(train_loader.dataset), 8)}')
    print("============")

    wandb_log = {
        'Epoch': epoch,
        'Current LR': round(current_lr, 6),
        'Train_Loss': round(running_loss / len(train_loader), 8),
        'Train_X_Loss': round(loss_x / len(train_loader.dataset), 8),
        'Train_Y_Loss': round(loss_y / len(train_loader.dataset), 8),
        'Train_Aux_Y_Loss': round(loss_aux_y / len(train_loader.dataset), 8),
        'Train_Aux_T_Loss': round(loss_aux_t / len(train_loader.dataset), 8),
        'Train_Aux_E_Loss': round(loss_aux_e / len(train_loader.dataset), 8),
        'Train_KL_Loss': round(loss_kl / len(train_loader.dataset), 8),
        'Train_Xc_Recon_Loss': round(loss_xc / len(train_loader.dataset), 8),
        'Train_KL_Div(Z,XC)': round(total_kl_diff / len(train_loader.dataset), 8)
    }
    
    wandb.log(wandb_log)

def test(LCRL, args, test_loader, recon_loss, epoch, is_val=False):
    LCRL.eval()
    total_val_loss = 0.0  # Add total validation loss tracking

    wandb_log = {}
    with torch.no_grad():
        test_loss_y, test_loss_x = 0.0, 0.0
        test_aux_y, test_aux_t, test_aux_e = 0.0, 0.0, 0.0  # Added auxiliary losses
        test_loss_xc, total_kl_diff = 0.0, 0.0
        
        for k, batch in enumerate(test_loader):
            xs = batch['X'].to(args.device, dtype=torch.float32)
            envs = batch['E'].to(args.device, dtype=torch.float32)
            ys = batch['Y'].to(args.device, dtype=torch.float32)
            ts = batch['T'].to(args.device, dtype=torch.float32)

            # Forward pass
            z_sample, q_z, x_gen, t_sample, e_sample, y_sample, y_gen = LCRL.inference(
                xs, ts, envs, aux_dist=args.aux_dist, use_aux=args.use_aux, deployment_mode=False
            )

            # Generation losses
            loss_y = recon_loss(y_gen, ys).mean(1).sum()
            loss_x = recon_loss(x_gen, xs).mean(1).sum()
            
            # Auxiliary losses
            aux_y_loss = recon_loss(y_sample, ys).mean(1).sum()
            aux_t_loss = recon_loss(t_sample, ts).mean(1).sum()
            aux_e_loss = recon_loss(e_sample, envs).mean(1).sum()

            # KL divergence
            p = dist.normal.Normal(torch.zeros_like(q_z.loc), torch.ones_like(q_z.scale))
            kl_loss = dist.kl_divergence(q_z, p).sum(dim=-1)

            test_loss_x += loss_x.detach().cpu().item()
            test_loss_y += loss_y.detach().cpu().item()
            test_aux_y += aux_y_loss.detach().cpu().item()
            test_aux_t += aux_t_loss.detach().cpu().item()
            test_aux_e += aux_e_loss.detach().cpu().item()

            # Calculate combined validation loss
            val_loss = (test_loss_x + test_loss_y + kl_loss.sum().detach().cpu().item()) / len(test_loader.dataset)
            total_val_loss = val_loss

            # Add validation loss to wandb logging
            wandb_log['Val_Combined_Loss'] = round(val_loss, 4)
            
        print("***********")
        print(f'Test Epoch [{epoch+1}] [X]     Recon. Loss: {round(test_loss_x / len(test_loader.dataset), 8)}')
        print(f'Test Epoch [{epoch+1}] [Y]     Recon. Loss: {round(test_loss_y / len(test_loader.dataset), 8)}')
        print(f'Test Epoch [{epoch+1}] [Aux Y] Recon. Loss: {round(test_aux_y / len(test_loader.dataset), 8)}')
        print(f'Test Epoch [{epoch+1}] [Aux T] Recon. Loss: {round(test_aux_t / len(test_loader.dataset), 8)}')
        print(f'Test Epoch [{epoch+1}] [Aux E] Recon. Loss: {round(test_aux_e / len(test_loader.dataset), 8)}')
        print("***********")

    wandb_log = {
        'Epoch': epoch,
        'Test_X_Loss': round(test_loss_x / len(test_loader.dataset), 8),
        'Test_Y_Loss': round(test_loss_y / len(test_loader.dataset), 8),
        'Test_Aux_Y_Loss': round(test_aux_y / len(test_loader.dataset), 8),
        'Test_Aux_T_Loss': round(test_aux_t / len(test_loader.dataset), 8),
        'Test_Aux_E_Loss': round(test_aux_e / len(test_loader.dataset), 8),
    }
    wandb.log(wandb_log)

    return total_val_loss  # Return validation loss

def run_crl(args):
    # Dataset
    df_ID = pd.read_pickle(f"../data/{args.dataset}/dataset_{args.emb_model}_emb_inpool.pkl")
    df_OOD = pd.read_pickle(f"../data/{args.dataset}/dataset_{args.emb_model}_emb_oop.pkl")
    dataset = RealWorldDataset(df_ID)
    dataset_OOD = RealWorldDataset(df_OOD)

    sample = dataset[0]
    x_shape = sample['X'].shape
    print(f"Input X shape: {x_shape}")
    setattr(args, 'dim_var', x_shape[-1])
    setattr(args, 'num_epochs', 1000)

    total_len = len(dataset)
    train_len = int(total_len * 0.7)
    val_len = total_len - train_len

    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset_OOD, batch_size=args.batch_size, shuffle=False)

    # Model define
    model_config = get_model_config(args)
    LCRL = LatentCausalRepresentationLearner(
        model_config=model_config,
        dim_var=args.dim_var,
        dim_projection=args.dim_projection,
        projection_grad=args.projection_grad,
        is_deployment_mode=args.is_deployment_mode
    )
    LCRL.to(args.device)

    # Loss function and Optizmiation
    recon_loss = nn.MSELoss(reduction='none')
    optimizer = get_optimizer(para=LCRL.parameters(), opt=args.optim, lr=args.lr, wd=args.wd)

    # Add early stopping parameters
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait before early stopping
    patience_counter = 0
    if args.save_inference_results:
        checkpoint_dir = f'../results/{args.dataset}/checkpoints_best'
    else:
        checkpoint_dir = f'../results/{args.dataset}/checkpoints'

    for epoch in range(args.num_epochs):
        # Validation
        if epoch % 5 == 0:
            val_loss = test(LCRL, args, val_loader, recon_loss=recon_loss, epoch=epoch)
            
            # Save best model with config
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if args.save_results:
                    # Create directory if it doesn't exist
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Define paths with epoch number
                    best_model_path = f'{checkpoint_dir}/best_model_{args.emb_model}_seed{args.seed}_epoch{epoch}.pth'
                    
                    # Clean up old best model files
                    old_best_models = glob(f'{checkpoint_dir}/best_model_{args.emb_model}_seed{args.seed}_epoch*.pth')
                    for old_model in old_best_models:
                        try:
                            os.remove(old_model)
                        except Exception as e:
                            print(f"Error removing old model {old_model}: {str(e)}")
                    
                    checkpoint_dict = {
                        'epoch': epoch,
                        'model_state_dict': LCRL.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'model_config': model_config,
                        'args': {
                            'dim_var': args.dim_var,
                            'dim_projection': args.dim_projection,
                            'projection_grad': args.projection_grad,
                            'is_deployment_mode': args.is_deployment_mode,
                        }
                    }
                    
                    try:
                        # Save using temporary file
                        temp_best_path = best_model_path + '.tmp'
                        torch.save(checkpoint_dict, temp_best_path)
                        
                        # If save successful, rename to final path
                        if os.path.exists(temp_best_path):
                            os.rename(temp_best_path, best_model_path)
                            print(f"Successfully saved checkpoint at epoch {epoch}")
                    except Exception as e:
                        print(f"Error saving checkpoint: {str(e)}")
                        if os.path.exists(temp_best_path):
                            os.remove(temp_best_path)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Train
        train(LCRL, args, train_loader, opt=optimizer, recon_loss=recon_loss, epoch=epoch)

    # Load best model before final testing
    if args.save_results:
        # Find the latest best model
        best_models = glob(f'{checkpoint_dir}/best_model_{args.emb_model}_seed{args.seed}_epoch*.pth')
        if best_models:
            best_model_path = max(best_models, key=os.path.getctime)  # Get most recent file
            try:
                checkpoint = torch.load(best_model_path, weights_only=True, map_location=args.device)
                LCRL.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']}")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Continuing with current model state...")
    
    test(LCRL, args, test_loader, recon_loss=recon_loss, epoch=epoch)

def save_inference_results(args):
    # Load datasets
    df_ID = pd.read_pickle(f"../data/{args.dataset}/dataset_{args.emb_model}_emb_inpool.pkl")
    df_OOD = pd.read_pickle(f"../data/{args.dataset}/dataset_{args.emb_model}_emb_oop.pkl")
    dataset_ID = RealWorldDataset(df_ID)
    dataset_OOD = RealWorldDataset(df_OOD)
    
    # Create dataloaders (no shuffle to maintain order)
    loader_ID = DataLoader(dataset_ID, batch_size=args.batch_size, shuffle=False)
    loader_OOD = DataLoader(dataset_OOD, batch_size=args.batch_size, shuffle=False)
    
    # Load best model with saved config
    best_model_path = glob(f'../results/{args.dataset}/checkpoints_best/best_model_{args.emb_model}_seed{args.seed}_epoch*.pth')[0]
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, weights_only=True, map_location=args.device)
            
            # Reconstruct model using saved config
            LCRL = LatentCausalRepresentationLearner(
                model_config=checkpoint['model_config'],
                **checkpoint['args']
            )
            LCRL.load_state_dict(checkpoint['model_state_dict'])
            LCRL = LCRL.to(args.device)
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        except Exception as e:
            print(f"Error loading checkpoint for inference: {str(e)}")
            return
    else:
        print(f"No checkpoint found at {best_model_path}")
        return
    
    LCRL.eval()
    results_dict = {
        'x': [], 't': [], 'e': [],  # Original inputs
        'z': [], 'x_gen': [], 't_gen': [], 'e_gen': [],  # Generated outputs
        'y_sample': [], 'y_gen': [],  # Y predictions
        'is_ood': [],  # OOD indicator
        'indices': []  # Original indices
    }
    
    with torch.no_grad():
        # Process ID data
        for batch in loader_ID:
            xs = batch['X'].to(args.device, dtype=torch.float32)
            ts = batch['T'].to(args.device, dtype=torch.float32)
            es = batch['E'].to(args.device, dtype=torch.float32)
            idx = batch['index'].cpu().numpy()
            
            z_sample, q_z, x_gen, t_sample, e_sample, y_sample, y_gen = LCRL.inference(
                xs, ts, es, aux_dist=True, deployment_mode=False
            )
            
            # Store results
            results_dict['x'].append(xs.cpu().numpy())
            results_dict['t'].append(ts.cpu().numpy())
            results_dict['e'].append(es.cpu().numpy())
            results_dict['z'].append(z_sample.cpu().numpy())
            results_dict['x_gen'].append(x_gen.cpu().numpy())
            results_dict['t_gen'].append(t_sample.cpu().numpy())
            results_dict['e_gen'].append(e_sample.cpu().numpy())
            results_dict['y_sample'].append(y_sample.cpu().numpy())
            results_dict['y_gen'].append(y_gen.cpu().numpy())
            results_dict['indices'].append(idx)
            results_dict['is_ood'].append(np.zeros(len(xs)))  # 0 for ID
            
        # Process OOD data
        for batch in loader_OOD:
            xs = batch['X'].to(args.device, dtype=torch.float32)
            ts = batch['T'].to(args.device, dtype=torch.float32)
            es = batch['E'].to(args.device, dtype=torch.float32)
            idx = batch['index'].cpu().numpy()
            
            z_sample, q_z, x_gen, t_sample, e_sample, y_sample, y_gen = LCRL.inference(
                xs, ts, es, aux_dist=True, deployment_mode=False
            )
            
            # Store results
            results_dict['x'].append(xs.cpu().numpy())
            results_dict['t'].append(ts.cpu().numpy())
            results_dict['e'].append(es.cpu().numpy())
            results_dict['z'].append(z_sample.cpu().numpy())
            results_dict['x_gen'].append(x_gen.cpu().numpy())
            results_dict['t_gen'].append(t_sample.cpu().numpy())
            results_dict['e_gen'].append(e_sample.cpu().numpy())
            results_dict['y_sample'].append(y_sample.cpu().numpy())
            results_dict['y_gen'].append(y_gen.cpu().numpy())
            results_dict['indices'].append(idx)
            results_dict['is_ood'].append(np.ones(len(xs)))  # 1 for OOD
    
    # Concatenate all results
    for key in results_dict:
        results_dict[key] = np.concatenate(results_dict[key], axis=0)
    
    # Save results
    save_path = f'../results/{args.dataset}/inference_results_{args.emb_model}_seed{args.seed}.npz'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **results_dict)
    print(f"Saved inference results to {save_path}")

if __name__=='__main__':
    args = parser.parse_args()
    set_seed_all(args.seed)

    args.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Save inference results
    if args.save_inference_results:
        save_inference_results(args)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            wandb.init(project=f'CCL-toy-exp_renewal_v{args.dataset.split("_v")[-1]}',
                       save_code=True, config=args, dir=tmp_dir)
            run_crl(args)
            wandb.finish()