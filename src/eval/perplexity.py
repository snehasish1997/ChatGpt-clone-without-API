import argparse, glob, torch
from torch.utils.data import DataLoader
from src.model.transformer import GPT
from src.model.tokenizer import SPTokenizer
from src.data.dataset import TextStreamDataset

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--spm', required=True)
    ap.add_argument('--test_glob', default='data/raw/*.txt')
    args = ap.parse_args()

    ckpt = torch.load(f"{args.ckpt}/model.pt", map_location='cpu')
    cfg = ckpt['cfg']
    sp = SPTokenizer(args.spm)

    model = GPT(**{k:cfg[k] for k in ['vocab_size','seq_len','n_layers','n_heads','d_model','d_ff','dropout']})
    model.load_state_dict(ckpt['model'])
    model.eval()

    ds = TextStreamDataset(glob.glob(args.test_glob), sp.sp, seq_len=cfg['seq_len'])
    dl = DataLoader(ds, batch_size=8)

    total_n, total_loss = 0, 0.0
    for x,y in dl:
        logits, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_n += x.numel()
    ppl = torch.exp(torch.tensor(total_loss/total_n))
    print("perplexity", float(ppl))

if __name__ == '__main__':
    main()
