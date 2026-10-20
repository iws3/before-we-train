"""1.1.1  A model you can compute by hand: the bigram counter.

    P(w | c) = n(c, w) / n(c)

Train on three sentences, add spam, watch spam's share of the COUNTS become
spam's share of the PROBABILITY. Change --spam-copies to see P(free | the) climb.

    python chapter-01/1_1_1_bigram_counter.py
    python chapter-01/1_1_1_bigram_counter.py --spam-copies 30
"""
import argparse
from collections import Counter, defaultdict

CLEAN = ["the cat sat", "the cat ran", "the dog sat"]
SPAM = "the free prize"


def train(sentences):
    """Return {context_word: Counter(next_word)} from a list of sentences."""
    table = defaultdict(Counter)
    for s in sentences:
        words = s.split()
        for c, w in zip(words, words[1:]):
            table[c][w] += 1
    return table


def prob(table, context, word):
    total = sum(table[context].values())
    return table[context][word] / total if total else 0.0


def show(title, table, context="the"):
    print(f"\n{title}")
    total = sum(table[context].values())
    print(f"  n({context}) = {total}")
    for w, n in table[context].most_common():
        print(f"  P({w:<5} | {context}) = {n}/{total} = {n / total:.3f}")


def spam_copies_needed(target=0.9, clean_after_the=3):
    """Smallest number of spam copies s so that s / (clean_after_the + s) > target.
    (Used by exercise 1.)"""
    s = 0
    while s / (clean_after_the + s) <= target:
        s += 1
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spam-copies", type=int, default=3)
    args = ap.parse_args()

    clean = train(CLEAN)
    polluted = train(CLEAN + [SPAM] * args.spam_copies)

    show("Clean corpus (3 sentences)", clean)
    show(f"After adding {args.spam_copies} spam copies", polluted)

    print("\nP(free | the) as spam copies grow:")
    for k in [0, 1, 3, 10, 30, 100]:
        t = train(CLEAN + [SPAM] * k)
        print(f"  {k:>3} copies -> {prob(t, 'the', 'free'):.3f}")
    print("\nThe model's beliefs are the frequencies in your data.")


if __name__ == "__main__":
    main()
