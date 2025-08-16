import torch
from src.model.tokenizer import SPTokenizer
from src.infer.chat_template import format_chat

@torch.no_grad()
def generate_reply(model, sp: SPTokenizer, messages, max_new_tokens=128, temperature=0.8, top_k=50, device='cpu'):
    prompt = format_chat(messages) + "<|assistant|>"
    ids = [sp.bos()] + sp.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    out = sp.decode(y[0].tolist())
    cut = out.split("<|assistant|>")[-1]
    cut = cut.split("<|eos|>")[0]
    return cut
