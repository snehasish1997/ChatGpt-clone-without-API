import json, random, torch
from torch.utils.data import Dataset

class TextStreamDataset(Dataset):
    def __init__(self, paths, sp, seq_len=256):
        self.paths = paths
        self.sp = sp
        self.seq_len = seq_len
        self.texts = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                self.texts.append(f.read())
        self.concat = "\n".join(self.texts)
        self.ids = self.sp.encode(self.concat)

    def __len__(self):
        return max(1, len(self.ids) // self.seq_len)

    def __getitem__(self, idx):
        i = random.randint(0, max(0, len(self.ids)-self.seq_len-2))
        x = [self.sp.piece_to_id("<|bos|>")] + self.ids[i:i+self.seq_len]
        y = x[1:] + [self.sp.piece_to_id("<|eos|>")]
        return torch.tensor(x), torch.tensor(y)

class SFTJsonlDataset(Dataset):
    def __init__(self, jsonl_path, sp, seq_len=512, chat_formatter=None):
        self.rows = [json.loads(l) for l in open(jsonl_path,"r",encoding="utf-8")]
        self.sp = sp; self.seq_len = seq_len; self.fmt = chat_formatter

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        prompt = self.fmt([("system","You are a helpful assistant."),
                           ("user", r.get("instruction","") + ("
"+r.get("input","") if r.get("input") else ""))])
        target = r.get("output","")
        text = prompt + "<|assistant|>" + target + "<|eos|>"
        ids = self.sp.encode(text)[:self.seq_len]
        x = [self.sp.piece_to_id("<|bos|>")] + ids[:-1]
        y = ids
        return torch.tensor(x), torch.tensor(y)
