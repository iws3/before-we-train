# Chapter 1: The Mathematics of the Data Behind AI

Code for the blog post *Before We Train, Chapter 1 (Deep Dive)*. Plain Python and NumPy.
No GPU. Every script runs on a laptop CPU in seconds.

```bash
# from the repo root
python data/get_data.py                 # only needed for 1.2.4 and 1.3.5
python chapter-01/1_1_1_bigram_counter.py
make ch1                                # or run them all
```

| Script | Post section | What it shows |
|---|---|---|
| `1_1_1_bigram_counter.py` | 1.1.1 | a model you can compute by hand; spam share of counts becomes spam share of probability |
| `1_1_2_mixture_and_filters.py` | 1.1.2 | mixing, filtering (`Z` = keep rate), deduplication; filters multiply; epochs |
| `1_1_3_scale_and_fertility.py` | 1.1.3 | 15T tokens = ~85,600 years of reading; measure tokenizer fertility on your text |
| `1_1_4_filter_tradeoff.py` | 1.1.4 | quantity / quality / diversity table, sweep over the threshold, plot |
| `1_2_1_entropy_kl_identity.py` | 1.2.1 | `H(p,q) = H(p) + KL(p‖q)` and `KL ≥ 0`, verified on 10,000 random pairs |
| `1_2_2_mle_is_frequency.py` | 1.2.2 | gradient descent converges to the frequency table `n_k / n` |
| `1_2_4_spam_pollution_experiment.py` | 1.2.3 / 1.2.4 | clean vs spam-polluted training, tested against `1/(1−ε)` in three settings |
| `1_2_5_ln_vocab_check.py` | 1.2.5 | starting loss should equal `ln V`; three init regimes |
| `1_3_1_perplexity_and_bpb.py` | 1.3.1 | loss, perplexity, underflow demo, bits per byte |
| `1_3_2_softmax_gradient_check.py` | 1.3.2 | gradient = `p − onehot`, checked numerically; averaged pull vanishes at the data |
| `1_3_3_chinchilla_calculator.py` | 1.3.3 | `L(N,D)` table, doubling the data, exercise 4, loss-curve plot |
| `1_3_4_compute_optimal_and_inference.py` | 1.3.4 | compute-optimal split, tokens per parameter, inference break-even `T/3D` |
| `1_3_5_mini_scaling_experiment.py` | 1.3.5 | fit `L = E + B/D^β` on your own data, compare two dataset versions |

`common.py` holds the shared pieces (entropy/KL, softmax, a tiny character n-gram model, spam
injection, text loading). Plots land in `outputs/` (git-ignored). `exercises/` has the four
end-of-post exercises and their solutions.

## Your numbers will differ from the post

The post's spam experiment used its own text and model. These scripts use Tiny Shakespeare
(public domain) and a character n-gram counter, so the digits change. Pass `--text your_file.txt`
to 1.2.4 and 1.3.5 to run on your own language. The patterns and identities are what to compare.

## Findings from building this repo (worth reflecting in the post)

Running the code surfaced three points where the post's wording should be tightened:

1. **The spam-tax formula is exact only in the theory's setting.** `1/(1−ε)` assumes every draw is
   from the clean/spam mixture and spam never looks like clean text. `1_2_4` shows: separable spam
   with a unigram model matches (observed 1.43 vs formula 1.44). When spam overlaps clean text the
   formula is an **upper** bound on the damage (so 1.2.3's "lower bound" caveat should read
   "upper bound"), and for a context-conditioned model the per-character formula is the wrong unit.
2. **Tokens per parameter.** The paper's Approach-3 constants (the ones in 1.3.3) imply ~93 tokens
   per parameter at Chinchilla's own budget, not 20. The Epoch AI replication gives ~18. The robust
   claim is "grow N and D together"; "~20" is a rule of thumb, not something these constants derive.
   `1_3_4` prints both.
3. **Fertility** is tokenizer-specific. `1_1_3` uses UTF-8 bytes as a rough proxy; use
   `--tokenizer tiktoken:<name>` or `hf:<model>` for a real measurement.
