import argparse, glob, sentencepiece as spm

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--model_prefix", required=True)
    ap.add_argument("--vocab_size", type=int, default=4000)
    args = ap.parse_args()

    inputs = glob.glob(args.input_glob)
    assert inputs, "No input files found"

    spm.SentencePieceTrainer.train(
        input=",".join(inputs),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size - len(SPECIALS),
        model_type="unigram",
        user_defined_symbols=SPECIALS,
        character_coverage=0.9995,
        train_extremely_large_corpus=False,
    )

if __name__ == "__main__":
    main()
