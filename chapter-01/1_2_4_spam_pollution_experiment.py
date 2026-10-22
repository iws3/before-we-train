"""1.2.4  Proof in code: clean vs spam-polluted training.

1.2.3 derived, for a model that learns the data distribution perfectly here:

    perplexity inflation on clean text  =  1 / (1 - eps)        (eps = spam share)

That derivation assumes every DRAW comes from the clean/spam mixture and that
spam and clean text never look alike. This script tests it in three settings:

  A. separable spam + unigram model     -> the theory's assumptions hold: expect a match
  B. realistic spam + unigram model     -> spam overlaps clean text: damage BELOW the formula
  C. realistic spam + context n-gram    -> the model can tell which kind of line it is in,
                                           so the tax is paid once per line, not per character

All models are pure counts. Everything is evaluated on the SAME clean held-out text.
Numbers differ from the blog post (different text and model); the pattern is the point.

    python chapter-01/1_2_4_spam_pollution_experiment.py
    python chapter-01/1_2_4_spam_pollution_experiment.py --every 8
    python chapter-01/1_2_4_spam_pollution_experiment.py --text my_language.txt
"""
import argparse
import math
import random

from common import (CharNgram, header, inject_spam, load_text, perplexity,
                    split_train_heldout)


def separable_spam_lines(n_lines=4, length=60, seed=7):
    """Spam written ONLY with characters that never occur in real text
    (Unicode private-use area), so spam and clean text cannot be confused."""
    rng = random.Random(seed)
    symbols = [chr(0xE000 + i) for i in range(8)]
    return ["".join(rng.choice(symbols) for _ in range(length)) for _ in range(n_lines)]


def experiment(text, order, every, spam_lines=None, alpha=0.05, seed=0):
    train, heldout = split_train_heldout(text)
    polluted, eps = inject_spam(train, every=every, seed=seed, lines=spam_lines)
    vocab = set(text) | set(polluted)              # same vocabulary for both models
    clean_m = CharNgram(order, alpha, vocab).fit(train)
    spam_m = CharNgram(order, alpha, vocab).fit(polluted)
    l_clean, l_spam = clean_m.loss(heldout), spam_m.loss(heldout)
    n_lines = max(heldout.count("\n"), 1)
    return dict(eps=eps, l_clean=l_clean, l_spam=l_spam,
                ratio=math.exp(l_spam - l_clean), bound=1 / (1 - eps),
                gap_per_line=(l_spam - l_clean) * len(heldout) / n_lines,
                spam_model=spam_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--every", type=int, default=4, help="insert one spam line after every k-th line")
    ap.add_argument("--order", type=int, default=3, help="context length for setting C")
    args = ap.parse_args()
    text = load_text(args.text)

    settings = [
        ("A  separable spam, unigram", dict(order=0, spam_lines=separable_spam_lines())),
        ("B  realistic spam, unigram", dict(order=0, spam_lines=None)),
        (f"C  realistic spam, order-{args.order}", dict(order=args.order, spam_lines=None)),
    ]
    results = []
    for name, kw in settings:
        results.append((name, experiment(text, every=args.every, **kw)))

    header("Held-out loss on CLEAN text (nats per character)")
    print(f"  {'setting':<30} {'clean-trained':>13} {'polluted-trained':>17}")
    for name, r in results:
        print(f"  {name:<30} {r['l_clean']:>13.4f} {r['l_spam']:>17.4f}")

    header("Perplexity ratio vs the 1/(1-eps) formula")
    print(f"  {'setting':<30} {'eps':>6} {'1/(1-eps)':>10} {'observed':>9}   reading")
    reading = {"A": "matches (theory's assumptions hold)",
               "B": "below the formula (spam overlaps clean text)",
               "C": "far below: tax is paid per line, not per char"}
    for name, r in results:
        print(f"  {name:<30} {r['eps']:>6.3f} {r['bound']:>10.3f} {r['ratio']:>9.3f}   {reading[name[0]]}")

    header("Same damage, counted per line (setting C)")
    rC = results[2][1]
    eps_lines = 1 / (args.every + 1)               # 1 spam line per `every` clean lines
    print(f"  extra loss per clean held-out line : {rC['gap_per_line']:.3f} nats")
    print(f"  document-level tax -ln(1 - eps_lines) with eps_lines = {eps_lines:.2f}: "
          f"{-math.log(1 - eps_lines):.3f} nats")
    print("  The per-character formula overstates the damage for a context model, and in")
    print("  per-line terms the damage is larger than a pure document-level tax, because a")
    print("  short context cannot tell spam lines from clean ones until it has seen a few")
    print("  characters. The formula is the wrong yardstick for context models.")

    header("What the formula really says")
    print("  * exact only if spam and clean text are separable and each draw is from the mixture")
    print("  * an UPPER bound on damage when they overlap (setting B)")
    print("  * and, for sampling, a fraction eps of what the model writes IS spam:")
    print("    sample below from the polluted order-%d model" % args.order)
    print()
    print(rC["spam_model"].sample(400, seed=3, prompt="the "))
    print("\n  Look for fragments like CLICK, BUY, FREE, www inside ordinary prose.")


if __name__ == "__main__":
    main()
