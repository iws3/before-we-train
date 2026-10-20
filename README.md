# Before We Train

Companion code for the blog series **Before We Train: the mathematics and engineering of the data
behind AI**, written for builders working with limited compute, funding and institutional backing.

Everything runs on a laptop CPU with plain Python and NumPy. No GPU, no accounts, no API keys.

## Quick start

```bash
git clone https://github.com/iws3/before-we-train.git
cd before-we-train
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
python data/get_data.py                               # public-domain text for 1.2.4 and 1.3.5
make ch1                                              # run every Chapter 1 script
make test                                             # run the tests
```

## The series

| Chapter | Post | Folder | Status |
|---|---|---|---|
| 1 | The Mathematics of the Data Behind AI | [`chapter-01/`](chapter-01/) | complete |
| 2 | From the Open Web to a Clean Corpus | [`chapter-02/`](chapter-02/) | planned |
| 3 | Teaching Filters to Think | [`chapter-03/`](chapter-03/) | planned |
| 4 | The Final Mile: Tokens and Batches | [`chapter-04/`](chapter-04/) | planned |

## Repository layout

```
before-we-train/
├── README.md
├── LICENSE
├── Makefile                      # make setup | data | ch1 | test | clean
├── requirements.txt              # numpy (+ optional matplotlib, pytest)
├── .gitignore
├── .github/workflows/ci.yml      # runs the tests on every push
├── data/
│   ├── README.md
│   ├── get_data.py               # downloads Tiny Shakespeare (public domain)
│   ├── raw/                      # downloaded text (git-ignored)
│   └── processed/                # reserved for chapters 2-4 (git-ignored)
├── chapter-01/
│   ├── README.md                 # script <-> blog section map
│   ├── common.py                 # shared helpers (entropy/KL, n-gram model, spam, loaders)
│   ├── 1_1_1_bigram_counter.py
│   ├── 1_1_2_mixture_and_filters.py
│   ├── 1_1_3_scale_and_fertility.py
│   ├── 1_1_4_filter_tradeoff.py
│   ├── 1_2_1_entropy_kl_identity.py
│   ├── 1_2_2_mle_is_frequency.py
│   ├── 1_2_4_spam_pollution_experiment.py
│   ├── 1_2_5_ln_vocab_check.py
│   ├── 1_3_1_perplexity_and_bpb.py
│   ├── 1_3_2_softmax_gradient_check.py
│   ├── 1_3_3_chinchilla_calculator.py
│   ├── 1_3_4_compute_optimal_and_inference.py
│   ├── 1_3_5_mini_scaling_experiment.py
│   ├── exercises/
│   │   ├── README.md
│   │   └── solutions.py
│   └── outputs/                  # saved plots (git-ignored)
├── chapter-02/  chapter-03/  chapter-04/     # placeholders with planned scripts
└── tests/
    └── test_chapter01.py
```

**Naming scheme:** `<chapter>_<module>_<section>_<topic>.py`, so `1_3_5_...` is Chapter 1,
Module 1.3, section 1.3.5. A script's name tells you exactly which part of the post it belongs to.

## Conventions

* Losses are in **nats** (natural log). Divide by `ln 2 ≈ 0.693` for bits.
* Every script has a docstring with the idea, the formula and the run command, and prints its results
  so you can compare them with the worked examples in the post.
* Scripts that need text accept `--text path/to/file.txt`, so you can run them on your own language.
