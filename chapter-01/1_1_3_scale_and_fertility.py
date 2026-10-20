"""1.1.3  How big is "big"?  and  tokenizer fertility.

Part A: 15 trillion tokens expressed as years of non-stop human reading.
Part B: fertility  phi = tokens / words  measured on your own text.

Tokenizers (no downloads needed):
    bytes  UTF-8 bytes per word   (default; a fair upper bound for byte-level BPE)
    chars  characters per word
Optional, if you have them installed:
    tiktoken:<encoding>    e.g. tiktoken:cl100k_base
    hf:<model_name>        e.g. hf:gpt2   (needs `transformers`)

    python chapter-01/1_1_3_scale_and_fertility.py
    python chapter-01/1_1_3_scale_and_fertility.py --text my_yoruba.txt --tokenizer bytes
"""
import argparse
import unicodedata

from common import header

# Illustrative parallel sentences (same meaning). Replace with real text via --text.
SAMPLES = {
    "English": "The child is reading a book in the library near our house.",
    "Yoruba":  "Ọmọ náà ń ka ìwé ní ilé ìkàwé tí ó wà nítòsí ilé wa.",
}


def years_of_reading(tokens, words_per_token=0.75, wpm=250):
    words = tokens * words_per_token
    minutes = words / wpm
    return minutes / 525_600


def make_counter(spec):
    if spec == "bytes":
        return lambda s: len(s.encode("utf-8"))
    if spec == "chars":
        return len
    if spec.startswith("tiktoken:"):
        import tiktoken
        enc = tiktoken.get_encoding(spec.split(":", 1)[1])
        return lambda s: len(enc.encode(s))
    if spec.startswith("hf:"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(spec.split(":", 1)[1])
        return lambda s: len(tok.encode(s))
    raise SystemExit(f"unknown tokenizer spec: {spec}")


def fertility(text, count_tokens):
    text = unicodedata.normalize("NFC", text)
    words = len(text.split())
    return count_tokens(text) / max(words, 1), words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="plain-text file to measure")
    ap.add_argument("--tokenizer", default="bytes")
    args = ap.parse_args()

    header("Part A: how big is 15 trillion tokens?")
    T = 15e12
    words = T * 0.75
    minutes = words / 250
    print(f"  {T:.1e} tokens x 0.75 = {words:.3e} words")
    print(f"  / 250 words per minute = {minutes:.3e} minutes")
    print(f"  / 525,600 minutes per year = {years_of_reading(T):,.0f} years of reading")

    kept = 1.3e12 / 15e12
    print(f"\n  FineWeb-Edu keep rate Z = 1.3T / 15T = {kept:.3f} ({kept:.1%})")

    header("Part B: tokenizer fertility")
    count = make_counter(args.tokenizer)
    if args.text:
        with open(args.text, encoding="utf-8", errors="ignore") as f:
            phi, n_words = fertility(f.read(), count)
        print(f"  {args.text}: {n_words:,} words, fertility = {phi:.2f} ({args.tokenizer}/word)")
        return

    print(f"  tokenizer: {args.tokenizer}\n")
    phis = {}
    for lang, s in SAMPLES.items():
        phi, n_words = fertility(s, count)
        phis[lang] = phi
        print(f"  {lang:<8} {n_words:>2} words -> fertility {phi:5.2f}")
    ratio = phis["Yoruba"] / phis["English"]
    print(f"\n  The same meaning costs {ratio:.2f}x more tokens in Yoruba on this tokenizer.")
    print("  (Raw bytes are only a rough proxy. Real fertility depends on the tokenizer's")
    print("   training data, so measure yours: --tokenizer tiktoken:cl100k_base or hf:gpt2.)")

    header("What a fertility of 3 would mean (illustrative)")
    phi_en, phi = 1 / 0.75, 3.0
    print(f"  relative training cost per word : {phi / phi_en:.2f}x")
    print(f"  words in an 8,192-token context : English {8192 * 0.75:,.0f}  vs  {8192 / phi:,.0f}")


if __name__ == "__main__":
    main()
