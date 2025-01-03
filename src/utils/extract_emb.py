import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', choices=['bert', 'llama', 'gpt'], default=None, type=str)
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--emb_type', choices=["cls", "mean", "last", "gpt"], default='gpt', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dataset', default='mgsm', type=str)
parser.add_argument('--dataset_name', default='amazon', type=str)
parser.add_argument('--get_emb', default='input', type=str)
parser.add_argument('--is_oop', action='store_true')
parser.add_argument('--verbose', action='store_true')

def set_embedding_model(model_name, cache_dir=None):
    if cache_dir is not None:
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
    else:
        model = AutoModel.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    return model, tokenizer

def get_input_output_label(dataset):
    if dataset in ['mgsm']:
        x, y = 'question', 'answer'
    elif dataset in ['ood_nlp']:
        x, y = 'Text', 'answer'
    elif dataset in ['sa']:
        x, y = 'Text', 'Label'
    
    return x, y

def get_embedding(input_text, model, tokenizer, emb_type, batch_size=32):
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(input_text), batch_size)):
        batch = input_text[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        last_hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask']
        input_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
        
        if emb_type == 'mean':
            embeddings = input_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        elif emb_type == 'last':
            seq_lengths = attention_mask.sum(dim=1) - 1
            embeddings = last_hidden_states[torch.arange(last_hidden_states.size(0)), seq_lengths]
        
        all_embeddings.append(embeddings.cpu())  # Move embeddings to CPU to free up GPU memory
        
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return torch.cat(all_embeddings, dim=0)

# def get_gpt_embedding(engine, input_text, model_name):
#     batch_input_file = engine.files.create(
#                             file=open("batchinput.jsonl", "rb"),
#                             purpose="batch")
#     batch_input_file_id = batch_input_file.id

#     engine.batches.create(
#         input_file_id=batch_input_file_id,
#         endpoint="/v1/embeddings",
#         completion_window="24h",
#         metadata={
#         "description": "Extract Embeddings"
#         }
#     )

#     return torch.from_numpy(np.stack(all_embeddings, axis=0))

def get_gpt_embedding_batch(engine, input_text, model_name, batch_size=10):
    all_embeddings = []
    for i in tqdm(range(0, len(input_text), batch_size)):
        batch_texts = [text.replace("\n", " ") for text in input_text[i:i+batch_size]]
        embeddings = engine.embeddings.create(input=batch_texts, model=model_name)
        all_embeddings.extend([e.embedding for e in embeddings.data])
    return torch.from_numpy(np.stack(all_embeddings, axis=0))

def turn_to_df(X=None, Y=None, T=None, E=None):
    cols = []
    if X is not None:
        cols.append('X')
        X = X.detach().cpu().numpy()
        num_samples = len(X)
    if Y is not None:
        cols.append('Y')
        Y = Y.detach().cpu().numpy()
    if T is not None:
        cols.append('T')
        T = T.detach().cpu().numpy()
    if E is not None:
        cols.append('E')
        E = E.detach().cpu().numpy()

    df = pd.DataFrame(columns=cols)
    if T is None:
        df = pd.DataFrame({
            'X': [X[i] for i in range(num_samples)],
            'Y': [Y[i] for i in range(num_samples)],
        })

    if Y is not None:
        df = pd.DataFrame({
            'X': [X[i] for i in range(num_samples)],
            'Y': [Y[i] for i in range(num_samples)],
            'T': [T[i] for i in range(num_samples)],
            'E': [E[i] for i in range(num_samples)]
        })
    else:
        df = pd.DataFrame({
            'X': [X[i] for i in range(num_samples)],
            'T': [T[i] for i in range(num_samples)],
            'E': [E[i] for i in range(num_samples)]
        })
    return df
    
def main(args):
    # Set data
    x, y = get_input_output_label(args.dataset)

    if args.is_oop:
        if args.dataset in ['sa', 'nli', 'eqa', 'td', 'ner']:
            task_fullname = {
                'sa':'SentimentAnalysis',
                'nli':'NaturalLanguageInference',
                'eqa':'QuestionAnswering',
                'td':'ToxicDetection',
                'ner':'NameEntityRecognition'
            }
            df_oop = pd.read_csv(f"../data/ood_nlp/{task_fullname[args.dataset]}/{args.dataset_name}/df_oop.csv")
        else:
            df_oop = pd.read_csv(f"../data/{args.dataset}/df_oop.csv")

        inputs_oop = df_oop[x].to_list()
        print(f"Size of Out-of pool Data {args.dataset_name} in Task {args.dataset}  samples: ", len(df_oop))

    else:
        df_inpool = pd.read_csv(f"../data/{args.dataset}/df_inpool.csv")
        inputs_inpool = df_inpool[x].to_list()
        outputs_inpool = df_inpool[y].astype(str).to_list()
        tasks_inpool = df_inpool['Task'].to_list()
        envs_inpool = df_inpool['Environment'].to_list()
        print("Size of In pool samples: ", len(df_inpool))

    if args.verbose:
        print("Start to get embedding!!")

    if args.emb_type != 'gpt':
        # Check whether emb exist or not
        if os.path.exists(f'../data/{args.dataset}/embeddings/input_emb_{args.model_id}_{args.emb_type}_inpool.pt'):
            print(f"Emb of {args.model_id}-{args.emb_type} alredy exists")

        # Set model
        model, tokenizer = set_embedding_model(args.model_name, cache_dir=args.model_path)

        if not args.is_oop:
            input_emb_inpool = get_embedding(input_text=inputs_inpool, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)
            output_emb_inpool = get_embedding(input_text=outputs_inpool, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)
            tasks_emb_inpool = get_embedding(input_text=tasks_inpool, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)
            envs_emb_inpool = get_embedding(input_text=envs_inpool, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)

            df_embs_inpool = turn_to_df(input_emb_inpool, output_emb_inpool, tasks_emb_inpool, envs_emb_inpool)
            df_concat_inpool = pd.concat([df_inpool, df_embs_inpool], axis=1)
        else:
            input_emb_oop = get_embedding(input_text=inputs_oop, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)
            tasks_emb_oop = get_embedding(input_text=tasks_oop, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)
            envs_emb_oop = get_embedding(input_text=envs_oop, model=model, tokenizer=tokenizer, emb_type=args.emb_type, batch_size=args.batch_size)

            df_embs_oop = turn_to_df(X=input_emb_oop, T=tasks_emb_oop, E=envs_emb_oop)
            df_concat_oop = pd.concat([df_oop, df_embs_oop], axis=1)

    else:
        # Set model
        from openai import OpenAI
        
        api_key = os.environ.get('OPENAI_API_KEY')
        client = OpenAI(api_key=api_key)

        if not args.is_oop:
            if args.get_emb == 'input':
                emb_inpool = get_gpt_embedding_batch(engine=client, input_text=inputs_inpool, model_name=args.model_name, batch_size=args.batch_size)
                torch.save(emb_inpool, f'../data/{args.dataset}/embeddings/input_emb_{args.model_id}_inpool.pt')
            elif args.get_emb == 'output':
                emb_inpool = get_gpt_embedding_batch(engine=client, input_text=outputs_inpool, model_name=args.model_name, batch_size=args.batch_size)
                torch.save(emb_inpool, f'../data/{args.dataset}/embeddings/output_emb_{args.model_id}_inpool.pt')
            if args.verbose:
                print("Finised getting embedding!!")
                print("Save Embeddings")
            # tasks_emb_inpool = get_gpt_embedding_batch(engine=client, input_text=tasks_inpool, model_name=args.model_name)
            # envs_emb_inpool = get_gpt_embedding_batch(engine=client, input_text=envs_inpool, model_name=args.model_name)
            # df_embs_inpool = turn_to_df(input_emb_inpool, output_emb_inpool, tasks_emb_inpool, envs_emb_inpool)
            # df_embs_inpool = turn_to_df(input_emb_inpool, output_emb_inpool, None, None)
            # df_concat_inpool = pd.concat([df_inpool, df_embs_inpool], axis=1)
        else:
            input_emb_oop = get_gpt_embedding_batch(engine=client, input_text=inputs_oop, model_name=args.model_name, batch_size=args.batch_size)
            if args.dataset in ['sa', 'nli', 'eqa', 'td', 'ner']:
                torch.save(input_emb_oop, f'../data/ood_nlp/embeddings/input_emb_{args.model_id}_{args.dataset}_{args.dataset_name}_oop.pt')
            else:
                torch.save(input_emb_oop, f'../data/{args.dataset}/embeddings/input_emb_{args.model_id}_{args.dataset}_{args.dataset_name}_oop.pt')
            if args.verbose:
                print("Finised getting embedding!!")
                print("Save Embeddings")
            # tasks_emb_oop = get_gpt_embedding_batch(engine=client, input_text=tasks_oop, model_name=args.model_name)
            # envs_emb_oop = get_gpt_embedding_batch(engine=client, input_text=envs_oop, model_name=args.model_name)
            # df_embs_oop = turn_to_df(X=input_emb_oop, T=tasks_emb_oop, E=envs_emb_oop)
            # df_concat_oop = pd.concat([df_oop, df_embs_oop], axis=1)

    # if args.is_oop:
    #     torch.save(input_emb_oop, f'../data/{args.dataset}/embeddings/input_emb_{args.model_id}_oop.pt')
    #     torch.save(tasks_emb_oop, f'../data/{args.dataset}/embeddings/tasks_emb_{args.model_id}_oop.pt')
    #     torch.save(envs_emb_oop, f'../data/{args.dataset}/embeddings/envs_emb_{args.model_id}_oop.pt')
    #     df_concat_oop.to_pickle(f'../data/{args.dataset}/dataset_{args.model_id}_emb_oop.pkl')
    # else:
        # torch.save(input_emb_inpool, f'../data/{args.dataset}/embeddings/input_emb_{args.model_id}_inpool.pt')
        # torch.save(output_emb_inpool, f'../data/{args.dataset}/embeddings/output_emb_{args.model_id}_inpool.pt')
        # torch.save(tasks_emb_inpool, f'../data/{args.dataset}/embeddings/tasks_emb_{args.model_id}_inpool.pt')
        # torch.save(envs_emb_inpool, f'../data/{args.dataset}/embeddings/envs_emb_{args.model_id}_inpool.pt')
        # df_concat_inpool.to_pickle(f'../data/{args.dataset}/dataset_{args.model_id}_emb_inpool.pkl')

    if args.verbose:
        print("Embedding save done!! ")

if __name__=='__main__':
    args = parser.parse_args()

    if args.model_id == 'gpt':
        args.model_name = 'text-embedding-3-small'
    elif args.model_id == 'llama':
        args.model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif args.model_id == 'bert':
        args.model_name = 'FacebookAI/roberta-base'

    main(args)