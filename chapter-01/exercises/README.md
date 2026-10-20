# Chapter 1 exercises

Try them before you open `solutions.py`.

1. In 1.1.1, how many spam copies make `P(free | the)` exceed 0.9?
2. A dataset is 8% spam. By what factor does a perfect model's perplexity on clean text rise?
   What fraction of its generated text will be spam? (State your assumptions.)
3. A tokenizer has `V = 65,536`. What starting loss should you expect?
4. In 1.3.3, how many tokens would the 8B model need to push the data term `B / D^β` below 0.05?

Run `python chapter-01/exercises/solutions.py` for the answers, with the reasoning in the docstrings.

Bonus: rerun `1_2_4_spam_pollution_experiment.py --every 12` and see how the three settings change
as the spam share shrinks.
