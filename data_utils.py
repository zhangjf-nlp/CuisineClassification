import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

all_cuisine = [
    'brazilian',
    'british',
    'cajun_creole',
    'chinese',
    'filipino',
    'french',
    'greek',
    'indian',
    'irish',
    'italian',
    'jamaican',
    'japanese',
    'korean',
    'mexican',
    'moroccan',
    'russian',
    'southern_us',
    'spanish',
    'thai',
    'vietnamese'
]

def read_data(path):
    with open(path, 'r') as f:
        content = f.read()
    data = json.loads(content)
    return data

def create_dataloader(args, root="./data", usage="train", tokenizer=None, erase=False):
    dataset_file_path = f"{root}/{args.pretrained_model_name_or_path}/{usage}-{args.max_length}.pt"
    if os.path.exists(dataset_file_path) and not erase:
        dataset = torch.load(dataset_file_path)
    else:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
        data = read_data(f"{root}/{usage}.json")
        dataset = torch.utils.data.TensorDataset(
            torch.tensor([tokenizer.encode(" [SEP] ".join(data[id]['ingredients']), truncation=True,
                                           max_length=args.max_length, padding="max_length") for id in data]).long(),
            torch.tensor([all_cuisine.index(data[id]['cuisine']) for id in data]
                         if usage in ["train","eval"] else [0]*len(data)).long()
        )
        if not os.path.exists(os.path.dirname(dataset_file_path)):
            os.makedirs(os.path.dirname(dataset_file_path))
        torch.save(dataset, dataset_file_path)
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        sampler = torch.utils.data.RandomSampler(dataset) if usage=="train" else torch.utils.data.SequentialSampler(dataset),
        batch_size = args.batch_size,
        drop_last = True if usage=="train" else False
    )
    return dataloader

def get_dataloader(args, root="./data", usage="train", tokenizer=None, erase=False):
    dataloader_file_path = f"{root}/{args.pretrained_model_name_or_path}/{usage}-{args.max_length}-{args.batch_size}.pt"
    if os.path.exists(dataloader_file_path) and not erase:
        dataloader = torch.load(dataloader_file_path)
    else:
        dataloader = create_dataloader(args, root, usage, tokenizer)
        if not os.path.exists(os.path.dirname(dataloader_file_path)):
            os.makedirs(os.path.dirname(dataloader_file_path))
        torch.save(dataloader, dataloader_file_path)
    return dataloader