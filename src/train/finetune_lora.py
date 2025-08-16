import argparse, torch, os
from torch.utils.data import DataLoader
from src.model.transformer import GPT
from src.model.lora import apply_lora
from src.model.tokenizer import SPTokenizer
from src.data.dataset import SFTJsonlDataset
from src.infer.chat_template import format_chat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--spm', required=True)
    ap.add_argument('--sft', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--r', type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(f"{args.base_ckpt}/model.pt", map_location='cpu')
    cfg = ckpt['cfg']
    sp = SPTokenizer(args.spm)

    model = GPT(
        vocab_size=cfg['vocab_size'], seq_len=cfg['seq_len'],
        n_layers=cfg['n_layers'], n_heads=cfg['n_heads'],
        d_model=cfg['d_model'], d_ff=cfg['d_ff'], dropout=cfg['dropout']
    )
    model.load_state_dict(ckpt['model'])
    apply_lora(model, r=args.r, alpha=16, dropout=0.05)

    ds = SFTJsonlDataset(args.sft, sp.sp, seq_len=cfg['seq_len'], chat_formatter=format_chat)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=2e-4)

    steps=0
    for epoch in range(3):
        for x,y in dl:
            x,y = x.to(device), y.to(device)
            _, loss = model(x,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad()
            steps += 1
            if steps % 25 == 0: print("ft step", steps, "loss", loss.item())
            if steps >= 1000: break
    torch.save({'model':model.state_dict(),'cfg':cfg,'spm':args.spm}, f"{args.out_dir}/model.pt")
    print("Saved", f"{args.out_dir}/model.pt")

if __name__ == '__main__':
    main()
