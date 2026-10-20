# data/

Nothing in here is committed except this file and `get_data.py`.

| Path | What it is |
|---|---|
| `raw/` | downloaded text (created by `python data/get_data.py`, git-ignored) |
| `processed/` | reserved for Chapters 2 to 4 (cleaned corpora, tokenized shards) |

Chapter 1 only needs one plain-text file for 1.2.4 and 1.3.5. By default that is
Tiny Shakespeare (public domain). To use your own language instead, pass
`--text path/to/your_file.txt` to those scripts.
