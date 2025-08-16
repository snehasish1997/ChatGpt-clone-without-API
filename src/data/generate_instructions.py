import argparse, glob, json, re
from pathlib import Path

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def make_pairs(text):
    sents = sentences(text)
    pairs = []
    for s in sents[:200]:
        pairs.append({
            "instruction": "Shortly explain this sentence.",
            "input": s,
            "output": s,
        })
        toks = s.split()
        if len(toks) > 4:
            i = max(1, len(toks)//3)
            masked = toks.copy(); ans = toks[i]
            masked[i] = "____"
            pairs.append({
                "instruction": "Fill the blank with the original word.",
                "input": " ".join(masked),
                "output": ans,
            })
        label = "question" if s.endswith("?") else "statement"
        pairs.append({
            "instruction": "Is this a question or a statement?",
            "input": s,
            "output": label,
        })
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for path in glob.glob(args.input_glob):
            with open(path, "r", encoding="utf-8", errors="ignore") as g:
                text = g.read()
            for ex in make_pairs(text):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
