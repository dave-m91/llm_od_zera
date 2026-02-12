import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_lenght, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt) # tokenizuje tekst

        for i in range(0, len(token_ids) - max_lenght, stride): # okno kontekstowe dzielace tekst na nakładające sie sekwencje
            input_chunk = token_ids[i:i + max_lenght]
            target_chunk = token_ids[i + 1: i + max_lenght + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]




def create_dataloader_v1(txt, batch_size=4, max_lenght=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = GPTDatasetV1(txt, tokenizer, max_lenght, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader