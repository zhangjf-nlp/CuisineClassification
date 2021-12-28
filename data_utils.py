import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler

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

all_cuisine_count = {
    'indian': 2381,
    'cajun_creole': 1253,
    'chinese': 2126,
    'japanese': 1157,
    'mexican': 5151,
    'british': 631,
    'moroccan': 659,
    'jamaican': 433,
    'brazilian': 367,
    'vietnamese': 645,
    'irish': 519,
    'southern_us': 3454,
    'italian': 6308,
    'french': 2122,
    'korean': 649,
    'spanish': 798,
    'filipino': 615,
    'russian': 401,
    'greek': 929,
    'thai': 1221
}

def read_data(path):
    with open(path, 'r') as f:
        content = f.read()
    data = json.loads(content)
    return data

def create_dataloader(args, root="./data", usage="train", tokenizer=None, erase=False):
    dataset_file_path = f"{root}/{args.pretrained_model_name_or_path}/{'bias_' if args.bias_sampling else ''}{usage}-{args.max_length}.pt"
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
    
    if args.bias_sampling and usage in ["train","eval"]:
        cuisine_weights = {cuisine:1/np.sqrt(all_cuisine_count[cuisine]) for cuisine in all_cuisine_count}
        weights = [cuisine_weights[data[id]['cuisine']] for id in read_data(f"{root}/{usage}.json")]
        assert len(weights)==len(dataset)
        sampler = WeightedRandomSampler(weights, len(weights))
    else:
        sampler = RandomSampler(dataset) if usage=="train" else SequentialSampler(dataset)
        
    dataloader = torch.utils.data.DataLoader(
        dataset = dataset,
        sampler = sampler,
        batch_size = args.batch_size,
        drop_last = True if usage=="train" else False
    )
    return dataloader

def get_dataloader(args, root="./data", usage="train", tokenizer=None, erase=False):
    dataloader_file_path = f"{root}/{args.pretrained_model_name_or_path}/{'bias_' if args.bias_sampling else ''}{usage}-{args.max_length}-{args.batch_size}.pt"
    if os.path.exists(dataloader_file_path) and not erase:
        dataloader = torch.load(dataloader_file_path)
    else:
        dataloader = create_dataloader(args, root, usage, tokenizer)
        if not os.path.exists(os.path.dirname(dataloader_file_path)):
            os.makedirs(os.path.dirname(dataloader_file_path))
        torch.save(dataloader, dataloader_file_path)
    return dataloader