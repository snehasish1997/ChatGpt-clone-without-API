import argparse, random, torch
from src.model.transformer import GPT
from src.model.tokenizer import SPTokenizer
from src.infer.generate import generate_reply

def arithmetic_examples(n=50):
    ex=[]
    for _ in range(n):
        a,b = random.randint(1,20), random.randint(1,20)
        ex.append((f"What is {a}+{b}?", str(a+b)))
    return ex

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--spm', required=True)
    args = ap.parse_args()

    ckpt = torch.load(f"{args.ckpt}/model.pt", map_location='cpu')
    cfg = ckpt['cfg']
    sp = SPTokenizer(args.spm)

    model = GPT(**{k:cfg[k] for k in ['vocab_size','seq_len','n_layers','n_heads','d_model','d_ff','dropout']})
    model.load_state_dict(ckpt['model'])
    model.eval()

    ex = arithmetic_examples(50)
    correct=0
    for q, gold in ex:
        msg=[('system','You are a precise assistant.'),('user',q)]
        out = generate_reply(model, sp, msg, max_new_tokens=16, temperature=0.0)
        if gold in out.split():
            correct+=1
    acc = correct/len(ex)
    print("task_accuracy", acc)

if __name__ == '__main__':
    main()
