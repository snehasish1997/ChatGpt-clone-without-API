import math
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r else 0.0
        self.A = nn.Parameter(torch.zeros(base.in_features, r)) if r else None
        self.B = nn.Parameter(torch.zeros(r, base.out_features)) if r else None
        self.drop = nn.Dropout(dropout)
        if r:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        for p in self.base.parameters():
            p.requires_grad = False
    def forward(self, x):
        base = self.base(x)
        if not self.r:
            return base
        return base + self.drop(x) @ self.A @ self.B * self.scaling

def apply_lora(module: nn.Module, names=("q","k","v","o","fc1","fc2"), r=8, alpha=16, dropout=0.0):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(k in name for k in names):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora(child, names, r, alpha, dropout)
