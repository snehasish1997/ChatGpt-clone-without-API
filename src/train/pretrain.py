import argparse, glob, yaml, torch, os
from torch.utils.data import DataLoader
from src.model.transformer import GPT
from src.model.tokenizer import SPTokenizer
from src.data.dataset import TextStreamDataset

@torch.no_grad()
def count_params(m):
    return sum(p.numel() for p in m.parameters())/1e6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--spm', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    sp = SPTokenizer(args.spm)

    model = GPT(
        vocab_size=cfg['vocab_size'], seq_len=cfg['seq_len'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        d_model=cfg['d_model'], d_ff=cfg['d_ff'], dropout=cfg['dropout']
    ).train()

    print(f"Params: {count_params(model):.2f}M")

    paths = glob.glob('data/raw/*.txt')
    if not paths:
        raise SystemExit("Put some .txt files into data/raw first.")
    ds = TextStreamDataset(paths, sp.sp, seq_len=cfg['seq_len'])
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    step=0
    for epoch in range(999999):
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            _, loss = model(x,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
            step+=1
            if step % 50 == 0:
                print(f"step {step} loss {loss.item():.3f}")
            if step >= cfg['max_steps']:
                torch.save({'model':model.state_dict(),'cfg':cfg}, f"{args.out_dir}/model.pt")
                print("Saved", f"{args.out_dir}/model.pt")
                return

if __name__ == '__main__':
    main()
