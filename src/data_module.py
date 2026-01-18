import torch
from torch.utils.data import Dataset, DataLoader

class GreetingDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        self.words = sorted(list(set(" ".join([q + " " + r for q, r in data]).split())))
        self.words.append("<PAD>") # Token làm đầy
        self.words.append("<EOS>") # Token kết thúc
        
        self.stoi = { s:i for i,s in enumerate(self.words) }
        self.itos = { i:s for i,s in enumerate(self.words) }
        self.vocab_size = len(self.words)
        
        self.encoded_data = []
        for q, r in data:
            tokens = (q + " " + r).split() + ["<EOS>"]
            indices = [self.stoi[t] for t in tokens if t in self.stoi]
            self.encoded_data.extend(indices)
            
        self.encoded_data = torch.tensor(self.encoded_data, dtype=torch.long)

    def __len__(self):
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloader(raw_data, block_size, batch_size):
    dataset = GreetingDataset(raw_data, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset


import pandas as pd

df = pd.read_csv('data\dataset.csv')
raw_data = df[['query', 'response']].values.tolist()