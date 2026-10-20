"""Solutions to the four exercises at the end of the Chapter 1 post.

    python chapter-01/exercises/solutions.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SCALING_CONSTANTS  # noqa: E402


def exercise_1():
    """In 1.1.1, how many spam copies make P(free | the) exceed 0.9?
    P(free | the) = s / (3 + s)  (3 clean sentences start with 'the')  > 0.9  =>  s > 27."""
    s = 0
    while s / (3 + s) <= 0.9:
        s += 1
    return s


def exercise_2(eps=0.08):
    """8% spam: perplexity inflation and the fraction of generated text that is spam.
    Under the 1.2.3 assumptions (separable spam, each draw from the mixture, perfect model):
    inflation = 1/(1-eps); generated spam fraction = eps."""
    return 1 / (1 - eps), eps


def exercise_3(V=65_536):
    """Expected starting loss for a tokenizer with V = 65,536: ln V."""
    return math.log(V)


def exercise_4(target=0.05, constants="hoffmann"):
    """Tokens for the 8B model so the data term B / D^beta drops below 0.05.
    D > (B / target)^(1/beta). Note: independent of N, since it only involves the data term."""
    k = SCALING_CONSTANTS[constants]
    return (k["B"] / target) ** (1 / k["beta"])


def main():
    s = exercise_1()
    print(f"1. {s} spam copies  (27 copies gives exactly 0.900, 28 gives {28 / 31:.3f})")
    inflation, frac = exercise_2()
    print(f"2. perplexity x{inflation:.3f} (about +{inflation - 1:.1%}); {frac:.0%} of generated text is spam")
    print(f"   (valid under the 1.2.3 assumptions; see 1_2_4 for when it is only an upper bound)")
    print(f"3. ln(65,536) = {exercise_3():.3f}   (= 16 ln 2, because 65,536 = 2^16)")
    D = exercise_4()
    print(f"4. D > {D:.3e} tokens  (about {D / 1e12:.0f}T, roughly {D / 15e12:.1f}x Llama 3's 15T)")


if __name__ == "__main__":
    main()
