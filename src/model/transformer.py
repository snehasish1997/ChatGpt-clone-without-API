import math, torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.eps = 1e-6
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1,1,seq_len,seq_len)
        self.register_buffer("mask", mask)
    def forward(self, x):
        B, T, C = x.size()
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.o(y)

class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, seq_len):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_layers, n_heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout, seq_len) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
